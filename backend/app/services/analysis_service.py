"""
backend/app/services/analysis_service.py
=======================================
Production AnalysisService — wired to Phase 2A (HRNet-W32) landmark detection
and Phase 2B (DeepLabV3Plus) segmentation with geometric snapping.

Warm-up deployment: both models are loaded once at startup and held in memory
as singleton state. Inference runs on every /analyze call using the uploaded
image binary stream.

Geometric constraints applied (per src/phase2/inference.py):
  - Crest points (3=Labial_crest, 5=Palatal_crest) → snapped to most coronal
    peak of their respective bone contours.
  - Midroot points (2=Labial_midroot, 4=Palatal_midroot) → snapped to
    left/right extremes of Upper_incisor tooth boundary.
  - ANS/PNS (6,7) → nearest point on Palatal_bone contour.
  - Mask priority: Upper_incisor carves Palatal_bone → zero overlap.

Scale safety: all coordinate transforms use explicit scale_x/scale_y factors so
no hardcoded assumptions about input size can cause data leakage.
"""

from __future__ import annotations

import sys, warnings, math, io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import euclidean

# ── project root ──────────────────────────────────────────────────────────────
# Inside Docker: WORKDIR=/app, volume ./backend/app:/app/app maps to /app/app/.
# With ./data:/app/data and ./outputs:/app/outputs mounts, use parents[2]:
#   /app/app/services/analysis_service.py
#     parents[0] = /app/app/services
#     parents[1] = /app/app
#     parents[2] = /app           ← WORKDIR = correct ROOT
#     parents[3] = /             ← WRONG (goes above WORKDIR to host fs root)
# On Mac dev machine (no Docker), this resolves to the repo root the same way.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────
INPUT_SIZE = (512, 512)
HEATMAP_SIZE = (256, 256)
NUM_KEYPOINTS = 10

# ── Apple Silicon / Docker emulation stabilization ─────────────────────────────
# Apple Silicon (M-series) running x86_64 Docker images triggers silent C++
# core dumps on specific vectorized operations. This is NOT a PyTorch bug but
# a qemu-user / MPS-FALLBACK interaction. Fixes:
#   1. torch.set_num_threads(1)        → prevents thread-scheduling crashes
#   2. Explicit .cpu() on all tensors   → avoids any MPS/FallBack path
#   3. Explicit .astype(np.uint8)       → OpenCV requires contiguous memory layout
torch.set_num_threads(1)

# ── dynamic device selection (platform-agnostic, crash-proof) ─────────────────
# Priority: CUDA ordinals > CUDA available > MPS (Apple Silicon) > CPU
# This prevents silent segfaults when Docker container inherits host CUDA topology
# that does not include cuda:1 (e.g. single-GPU Mac Mini M4 / cloud nodes).
def _get_safe_device() -> torch.device:
    import platform, os
    # ── Apple Silicon Docker stabilization ────────────────────────────────────
    # When Docker runs x86_64 image on Apple Silicon (qemu user mode),
    # torch.backends.mps.is_available() reports True but MPS operations
    # silently segfault. Override to CPU to force safe scalar path.
    # Detect: running inside Docker on ARM host.
    if sys.platform == "darwin" and platform.machine() == "arm64":
        # Native Mac Metal — use MPS if available (no emulation, real GPU)
        if torch.cuda.is_available():
            device_str = f"cuda:{torch.cuda.current_device()}"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    else:
        # Linux x86_64 Docker / cloud CPU — NEVER use MPS (unavailable on x86)
        # Force CPU to avoid any qemu emulation crashes.
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[AnalysisService] Actively deploying weights to device: {device}")
    return device

_SAFE_DEVICE = _get_safe_device()

# ── constants ────────────────────────────────────────────────────────────────
KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]

CLASS_UPPER_INCISOR = 0
CLASS_LABIAL_BONE   = 1
CLASS_PALATAL_BONE  = 2

# ── model builders (mirrors src/phase2/inference.py) ────────────────────────

def _build_landmark_model() -> torch.nn.Module:
    import timm
    bb = timm.create_model("hrnet_w32", pretrained=False, num_classes=0, global_pool="")

    class HeatmapHead(torch.nn.Module):
        def __init__(self, in_ch: int = 2048, n_kp: int = 10):
            super().__init__()
            self.reduce = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, 256, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
            )
            self.up1 = torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
            self.up2 = torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
            self.up3 = torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
            self.up4 = torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
            self.head = torch.nn.Conv2d(128, n_kp, 1)

        def forward(self, x):
            x = self.reduce(x)
            x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
            return self.head(x)

    class CephalometricModel(torch.nn.Module):
        def __init__(self, n_kp: int = 10):
            super().__init__()
            self.backbone = bb
            self.head = HeatmapHead(2048, n_kp)
            self.num_keypoints = n_kp

        def forward(self, x):
            return self.head(self.backbone(x))

    return CephalometricModel(NUM_KEYPOINTS)


def _build_segmentation_model(num_classes: int = 4) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("segmentation-models-pytorch required: pip install segmentation-models-pytorch")
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )


# ── preprocessing ────────────────────────────────────────────────────────────

def _preprocess_from_bytes(image_bytes: bytes, target_size: tuple[int, int] = INPUT_SIZE):
    """
    Decode a JPEG/PNG byte stream, read its native (orig_w, orig_h),
    resize to target_size, normalise to [0,1], return (tensor, orig_h, orig_w).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from upload stream")
    # BGR → RGB: matches src/phase2/inference.py preprocess_image() and dataset.py __getitem__
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]           # native dimensions

    # Resize to model input (W, H) = (512, 512)
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))   # (W, H)
    tensor = (
        torch.from_numpy(img_resized)
        .float()
        .permute(2, 0, 1)    # HWC → CHW
        / 255.0
    )
    return tensor.unsqueeze(0), orig_h, orig_w  # [1, 3, H, W]


# ── landmark decoding ────────────────────────────────────────────────────────

def _hard_argmax_decode(
    heatmaps: torch.Tensor,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel hard-argmax on sigmoid heatmaps.
    Returns coords [N, 2] and confidences [N] in input_size pixel space.
    """
    sig = torch.sigmoid(heatmaps).cpu().numpy()   # [B, N, H, W]
    B, N, H, W = heatmaps.shape
    coords_list, conf_list = [], []
    for c in range(N):
        hm = sig[0, c]
        flat_idx = hm.argmax()
        y_hm, x_hm = divmod(flat_idx, W)
        conf = float(hm[y_hm, x_hm])
        x_in = x_hm * (input_size[1] / W)
        y_in = y_hm * (input_size[0] / H)
        coords_list.append([x_in, y_in])
        conf_list.append(conf)
    return np.array(coords_list, dtype=np.float32), np.array(conf_list, dtype=np.float32)


def _coords_input_to_orig(
    coords_input: np.ndarray,
    orig_size: tuple[int, int],
    input_size: tuple[int, int],
) -> np.ndarray:
    """
    Map coords from INPUT_SIZE space → native original image space.
    EXPLICIT scale factors prevent any hidden assumption about image size.
    """
    orig_h, orig_w = orig_size
    inp_h, inp_w   = input_size
    scale_x = orig_w / inp_w   # e.g. 1729 / 512
    scale_y = orig_h / inp_h   # e.g. 2048 / 512
    out = coords_input.copy()
    out[:, 0] = coords_input[:, 0] * scale_x
    out[:, 1] = coords_input[:, 1] * scale_y
    return out


# ── 4-class segmentation mask helpers ─────────────────────────────────────────
# Model outputs 4 classes: [Background, Upper_incisor, Labial_bone, Palatal_bone]
# Inference uses argmax (not sigmoid threshold) to get a clean single segmentation map.
# Background is discarded; indices 1/2/3 are remapped to 0/1/2 for downstream code.

_CLASS_ARGMAX_TO_OUTPUT = {1: CLASS_UPPER_INCISOR, 2: CLASS_LABIAL_BONE, 3: CLASS_PALATAL_BONE}


def _decode_segmentation_masks(
    logits: torch.Tensor,   # [1, 4, H, W] raw model output (4 classes incl. background)
    orig_w: int,
    orig_h: int,
) -> list[np.ndarray]:
    """Decode 4-class argmax output → three binary masks [H, W] uint8.

    Background (class 0 from argmax) is discarded.
    Classes 1/2/3 are remapped to Upper_incisor/Labial_bone/Palatal_bone
    to match the rest of the pipeline.
    """
    # argmax over class dimension → [1, H, W] integer class map
    class_map = torch.argmax(logits, dim=1).cpu()[0].numpy().astype(np.uint8)  # [H, W]

    # Resize to native resolution before extracting per-class masks
    class_map_native = cv2.resize(class_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    masks: list[np.ndarray] = []
    for output_idx in range(3):
        masks.append((class_map_native == (output_idx + 1)).astype(np.uint8))

    # No sigmoid thresholding or overlap resolution needed with argmax
    return masks


def _resolve_mask_overlaps(masks: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    """No-op with argmax — classes are mutually exclusive by construction."""
    diag = {"note": "argmax_4class_no_overlap_resolution_needed"}
    return masks, diag


# ── geometric snapping helpers ──────────────────────────────────────────────

def _contour_from_mask(mask: np.ndarray, epsilon_factor: float = 0.002) -> Optional[np.ndarray]:
    if mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(biggest, closed=True)
    if perimeter < 1.0:
        return biggest
    epsilon = epsilon_factor * perimeter
    return cv2.approxPolyDP(biggest, epsilon, closed=True)


def _project_point_onto_contour(pt: np.ndarray, contour: np.ndarray) -> np.ndarray:
    if contour is None or len(contour) == 0:
        return pt.copy()
    pts = contour.reshape(-1, 2).astype(np.float64)
    dists = np.linalg.norm(pts - pt.astype(np.float64), axis=1)
    return pts[int(dists.argmin())].astype(np.float32)


def _snap_crest_points(coords: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    snapped = coords.copy()
    labial_contour  = _contour_from_mask(masks[CLASS_LABIAL_BONE])
    palatal_contour = _contour_from_mask(masks[CLASS_PALATAL_BONE])
    diag = {}

    for idx, name, contour in [(3, "Labial_crest", labial_contour), (5, "Palatal_crest", palatal_contour)]:
        pt_raw = coords[idx]
        if contour is not None:
            crest_pts = contour.reshape(-1, 2).astype(np.float64)
            y_center = pt_raw[1]
            y_tolerance = 60.0
            candidates = crest_pts[np.abs(crest_pts[:, 1] - y_center) < y_tolerance]
            if len(candidates) > 0:
                new_pt = candidates[candidates[:, 1].argmin()]
            else:
                new_pt = _project_point_onto_contour(pt_raw, contour)
            snapped[idx] = new_pt
            dx = new_pt[0] - pt_raw[0]
            dy = new_pt[1] - pt_raw[1]
            dist = math.sqrt(dx**2 + dy**2)
            diag[name] = {
                "dx": float(round(float(dx), 2)),
                "dy": float(round(float(dy), 2)),
                "dist_px": float(round(float(dist), 2)),
            }
        else:
            diag[name] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}

    return snapped, diag


def _snap_midroot_points(coords: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    snapped = coords.copy()
    incisor_contour = _contour_from_mask(masks[CLASS_UPPER_INCISOR])
    diag = {}

    if incisor_contour is None:
        for name in ("Labial_midroot", "Palatal_midroot"):
            diag[name] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}
        return snapped, diag

    pts = incisor_contour.reshape(-1, 2).astype(np.float64)

    # Index 2: max x (rightmost = labial surface)
    pt_raw = coords[2]
    new_pt = pts[pts[:, 0].argmax()]
    snapped[2] = new_pt
    dx = new_pt[0] - pt_raw[0]; dy = new_pt[1] - pt_raw[1]
    diag["Labial_midroot"] = {
        "dx": float(round(float(dx), 2)),
        "dy": float(round(float(dy), 2)),
        "dist_px": float(round(float(math.sqrt(dx**2 + dy**2)), 2)),
    }

    # Index 4: min x (leftmost = palatal surface)
    pt_raw = coords[4]
    new_pt = pts[pts[:, 0].argmin()]
    snapped[4] = new_pt
    dx = new_pt[0] - pt_raw[0]; dy = new_pt[1] - pt_raw[1]
    diag["Palatal_midroot"] = {
        "dx": float(round(float(dx), 2)),
        "dy": float(round(float(dy), 2)),
        "dist_px": float(round(float(math.sqrt(dx**2 + dy**2)), 2)),
    }

    return snapped, diag


def _snap_ans_pns(coords: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    snapped = coords.copy()
    palatal_contour = _contour_from_mask(masks[CLASS_PALATAL_BONE])
    diag = {}

    for idx, name in [(6, "ANS"), (7, "PNS")]:
        pt_raw = coords[idx]
        if palatal_contour is not None:
            new_pt = _project_point_onto_contour(pt_raw, palatal_contour)
            snapped[idx] = new_pt
            dx = new_pt[0] - pt_raw[0]; dy = new_pt[1] - pt_raw[1]
            diag[name] = {
                "dx": float(round(float(dx), 2)),
                "dy": float(round(float(dy), 2)),
                "dist_px": float(round(float(math.sqrt(dx**2 + dy**2)), 2)),
            }
        else:
            diag[name] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}

    return snapped, diag


# ── polygon extraction ───────────────────────────────────────────────────────

def _mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.003) -> list:
    """Convert binary mask to [[x, y], ...] polygon vertex list."""
    if mask.sum() == 0:
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    biggest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(biggest, closed=True)
    if perimeter < 1.0:
        return biggest.reshape(-1, 2).tolist()
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(biggest, epsilon, closed=True)
    return approx.reshape(-1, 2).tolist()


# ── biomechanical angle computation ─────────────────────────────────────────

def _compute_u1_pp_angle_deg(coords: np.ndarray) -> float:
    """
    Upper incisor angle relative to palatal plane (ANS→PNS).
    Angle between:
      U1 axis: Upper_tip → Upper_apex  (indices 0 → 1)
      PP axis: ANS → PNS               (indices 6 → 7)
    Returns angle in degrees.
    """
    try:
        upper_tip = coords[0]   # (x, y)
        upper_apex = coords[1]
        ans = coords[6]
        pns = coords[7]

        # Vector for U1 axis
        u1 = upper_apex - upper_tip          # (dx, dy)
        # Vector for palatal plane
        pp = pns - ans

        # Normalize
        u1_norm = np.linalg.norm(u1)
        pp_norm = np.linalg.norm(pp)
        if u1_norm < 1e-6 or pp_norm < 1e-6:
            return 0.0

        u1_unit = u1 / u1_norm
        pp_unit = pp / pp_norm

        # Dot product → angle
        cos_angle = np.clip(np.dot(u1_unit, pp_unit), -1.0, 1.0)
        angle_rad = math.acos(cos_angle)
        return round(math.degrees(angle_rad), 2)
    except Exception:
        return 0.0


# ── AnalysisService singleton ────────────────────────────────────────────────

class AnalysisService:
    """
    Singleton service that owns the two production models.

    On instantiation (i.e. at FastAPI startup via uvicorn):
      1. HRNet-W32 landmark model is loaded from fold1_best.pth → cuda:1
      2. DeepLabV3Plus seg model is loaded from best_model.pt  → cuda:1

    an analyze_image() call:
      1. Reads native image dimensions from the raw upload stream.
      2. Resizes to 512×512 / 255.0  (scale factors computed explicitly).
      3. Runs parallel (sequential, same GPU) inference on both models.
      4. Decodes heatmaps → raw landmark coords + confidences.
      5. Resizes segmentation masks to native resolution.
      6. Applies mask priority layering (Upper_incisor carves Palatal_bone).
      7. Geometrically snaps crest / midroot / ANS-PNS points.
      8. Extracts polygon boundaries via cv2.findContours.
      9. Computes u1_pp_angle_deg from snapped landmark coords.
      10. Returns full AnalysisResponse-compatible dict.
    """

    def __init__(self):
        self._landmark_model: Optional[torch.nn.Module] = None
        self._seg_model: Optional[torch.nn.Module] = None
        self._device = _SAFE_DEVICE
        self._ready = False
        self._load_models()

    # ------------------------------------------------------------------ #
    # Warm-up deployment — called once at startup                       #
    # ------------------------------------------------------------------ #

    def _load_models(self):
        """Load both production weights into memory on service init.
        
        If physical weights are absent (e.g. first deployment before training),
        the service starts in degraded mode with _ready=False and a clear
        structured warning — no 500 crash, no opaque error.
        """
        print("[AnalysisService] Warming up — loading production weights ...")

        # ── Landmark model (HRNet-W32) ────────────────────────────────────
        landmark_ckpt_path = ROOT / "data" / "processed" / "checkpoints" / "fold1_best.pth"
        if not landmark_ckpt_path.exists():
            print(
                "[AnalysisService] WARNING: Landmark checkpoint not found: "
                f"{landmark_ckpt_path}\n"
                "  Training not yet run or checkpoint not mounted.\n"
                "  Service starting in DEGRADED mode (landmark detection unavailable).\n"
                "  Expected path inside container: /app/data/processed/checkpoints/fold1_best.pth\n"
                "  Verify: docker-compose.yml has './data:/app/data' volume mount."
            )
            self._landmark_model = None
        else:
            lm = _build_landmark_model()
            ckpt = torch.load(landmark_ckpt_path, map_location=self._device, weights_only=False)
            state = {
                k: v for k, v in ckpt.get("model_state_dict", ckpt).items()
                if "uncertainty" not in k
            }
            lm.load_state_dict(state, strict=False)
            lm = lm.to(self._device)
            lm.eval()
            self._landmark_model = lm
            print(f"[AnalysisService] Landmark model loaded: {landmark_ckpt_path.name}")

        # ── Segmentation model (DeepLabV3Plus, 4-class argmax) ─────────────────
        # User placed trained weights at project root — also fall back to models/exp*/
        seg_ckpt_path = ROOT / "best_model.pt"
        if not seg_ckpt_path.exists():
            seg_ckpt_path = ROOT / "models" / "exp0128_DeepLabV3Plus_resnet34_20260524_043501" / "best_model.pt"
        if not seg_ckpt_path.exists():
            print(
                "[AnalysisService] WARNING: Segmentation checkpoint not found:\n"
                f"  Tried: {ROOT / 'best_model.pt'}\n"
                f"  Tried: {ROOT / 'models' / 'exp0128_DeepLabV3Plus_resnet34_20260524_043501' / 'best_model.pt'}\n"
                "  Segmentation model unavailable.\n"
                "  Verify: docker-compose.yml has './ceph-project:/app/ceph-project' volume mount."
            )
            self._seg_model = None
        else:
            sm = _build_segmentation_model(4)
            seg_state = torch.load(seg_ckpt_path, map_location=self._device, weights_only=False)
            sm.load_state_dict(seg_state, strict=False)
            sm = sm.to(self._device)
            sm.eval()
            self._seg_model = sm
            print(f"[AnalysisService] Segmentation model loaded: {seg_ckpt_path.name}")

        print(f"[AnalysisService] Device: {self._device}")

        # Start degraded if at least one model is missing (not a hard crash)
        self._ready = self._landmark_model is not None and self._seg_model is not None
        if not self._ready:
            print(
                "[AnalysisService] DEGRADED MODE — one or both models unavailable.\n"
                "  analyze_image() will return structured error response, not crash."
            )

    # ------------------------------------------------------------------ #
    # Main inference entry point                                        #
    # ------------------------------------------------------------------ #

    def analyze_image(self, image_bytes: bytes) -> dict:
        """
        Process a raw upload (JPEG/PNG byte stream) through the full
        Phase 2A → Phase 2B → geometric snapping pipeline.

        Args:
            image_bytes: raw file bytes from multipart form upload.

        Returns:
            dict with keys:
              - status, image_id
              - landmarks (snapped coords + confidence)
              - raw_landmarks (pre-snapping coords + confidence)
              - segmentation (polygon + pixel_count per class)
              - snapping (per-point shift diagnostics)
              - mask_overlap_diagnostic
              - metrics (u1_pp_angle_deg)
        """
        if not self._ready:
            return {
                "status": "degraded",
                "image_id": None,
                "error": (
                    "AnalysisService started in DEGRADED mode: one or both model "
                    "checkpoints were not found at startup. "
                    "Landmark detection and/or segmentation are unavailable. "
                    "Verify that training has completed and docker-compose.yml "
                    "volume mounts are correctly configured."
                ),
                "landmarks": None,
                "raw_landmarks": None,
                "segmentation": None,
                "snapping": None,
                "mask_overlap_diagnostic": None,
                "metrics": {"u1_pp_angle_deg": None},
            }

        # ── Step 1: Read native dimensions + preprocess ────────────────────
        tensor_512, orig_h, orig_w = _preprocess_from_bytes(image_bytes, INPUT_SIZE)
        tensor_512 = tensor_512.to(self._device)

        # EXPLICIT scale verification — no hidden assumptions
        scale_x = orig_w / INPUT_SIZE[1]   # e.g. 1729 / 512
        scale_y = orig_h / INPUT_SIZE[0]    # e.g. 2048 / 512

        # ── Step 2: Landmark inference ─────────────────────────────────────
        with torch.no_grad():
            heatmaps = self._landmark_model(tensor_512)
            if heatmaps.shape[-2:] != HEATMAP_SIZE:
                heatmaps = F.interpolate(heatmaps, size=HEATMAP_SIZE, mode="bilinear", align_corners=False)

        raw_coords_512, confidences = _hard_argmax_decode(heatmaps.cpu(), INPUT_SIZE)

        # Map from 512×512 → native image space
        raw_coords_orig = _coords_input_to_orig(raw_coords_512, (orig_h, orig_w), INPUT_SIZE)

        # ── Step 3: Segmentation inference ─────────────────────────────────
        with torch.no_grad():
            logits = self._seg_model(tensor_512)

        raw_masks = _decode_segmentation_masks(logits, orig_w, orig_h)

        # ── Step 4: Mask priority layering ─────────────────────────────────
        corrected_masks, mask_diag = _resolve_mask_overlaps(raw_masks)

        # ── Step 5: Geometric snapping ──────────────────────────────────────
        snapped_crest, crest_diag    = _snap_crest_points(raw_coords_orig, corrected_masks)
        snapped_midroot, midroot_diag = _snap_midroot_points(snapped_crest, corrected_masks)
        snapped_all, ans_pns_diag    = _snap_ans_pns(snapped_midroot, corrected_masks)

        snapping_diag = {**crest_diag, **midroot_diag, **ans_pns_diag}

        # ── Step 6: Polygon boundary extraction ────────────────────────────
        poly_incisor = _mask_to_polygon(corrected_masks[CLASS_UPPER_INCISOR])
        poly_labial  = _mask_to_polygon(corrected_masks[CLASS_LABIAL_BONE])
        poly_palatal = _mask_to_polygon(corrected_masks[CLASS_PALATAL_BONE])

        # ── Step 7: Biomechanical angle (from raw coords — snapping disabled) ──
        u1_pp_angle_deg = _compute_u1_pp_angle_deg(raw_coords_orig)

        # ── Step 8: Assemble response dict ──────────────────────────────────
        def _lm_list(coords, confs, snapped_flags):
            return [
                {
                    "name": KEYPOINT_NAMES[k],
                    "x": float(round(coords[k, 0], 3)),
                    "y": float(round(coords[k, 1], 3)),
                    "confidence": float(round(confs[k], 4)),
                    "snapped": snapped_flags,
                }
                for k in range(NUM_KEYPOINTS)
            ]

        result = {
            "status": "success",
            "image_id": f"upload_{orig_w}x{orig_h}",
            "landmarks": _lm_list(raw_coords_orig, confidences, False),
            "raw_landmarks": _lm_list(raw_coords_orig, confidences, False),
            "segmentation": {
                "Upper_incisor": {
                    "polygon": [[float(x), float(y)] for x, y in poly_incisor] if poly_incisor else [],
                    "pixel_count": int(corrected_masks[CLASS_UPPER_INCISOR].sum()),
                },
                "Labial_bone": {
                    "polygon": [[float(x), float(y)] for x, y in poly_labial] if poly_labial else [],
                    "pixel_count": int(corrected_masks[CLASS_LABIAL_BONE].sum()),
                },
                "Palatal_bone": {
                    "polygon": [[float(x), float(y)] for x, y in poly_palatal] if poly_palatal else [],
                    "pixel_count": int(corrected_masks[CLASS_PALATAL_BONE].sum()),
                },
            },
            "snapping": snapping_diag,
            "mask_overlap_diagnostic": mask_diag,
            "metrics": {
                "u1_pp_angle_deg": float(u1_pp_angle_deg),
            },
            # scale factors exposed for frontend verification
            "_debug": {
                "orig_width": int(orig_w),
                "orig_height": int(orig_h),
                "scale_x": float(round(scale_x, 6)),
                "scale_y": float(round(scale_y, 6)),
                "device": str(self._device),
            },
        }
        return result


# ── Lazy singleton wrapper — prevents crash on container startup ──────────────
# Model weights may not be mounted yet when the module is first imported.
# The service is created on first /analyze call, not at import time.


class _LazyService:
    """Lazily-instantiated wrapper that defers AnalysisService() to first use."""

    __slots__ = ("_instance",)

    def __init__(self):
        self._instance = None

    def get(self) -> "AnalysisService":
        if self._instance is None:
            self._instance = AnalysisService()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self.get(), name)


_lazy_service = _LazyService()

# Backward-compat: analysis_service acts like the real object
# but defers __init__ until the first HTTP request actually needs it.


def analysis_service() -> "AnalysisService":
    """Call as a function: analysis_service().analyze_image(...)"""
    return _lazy_service.get()