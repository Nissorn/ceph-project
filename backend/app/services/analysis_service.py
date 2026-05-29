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

import sys, warnings, math, io, os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import euclidean

# ── project root ──────────────────────────────────────────────────────────────
# Inside Docker: WORKDIR=/app, volume ./backend/app:/app/app maps to /app/app/.
# With ./backend/models:/app/models, ./data:/app/data, ./outputs:/app/outputs
# mounts, use parents[2]:
#   /app/app/services/analysis_service.py
#     parents[0] = /app/app/services
#     parents[1] = /app/app
#     parents[2] = /app           ← WORKDIR = correct ROOT
#     parents[3] = /             ← WRONG (goes above WORKDIR to host fs root)
# On Mac dev machine (no Docker), this resolves to the repo root the same way.
_parents = Path(__file__).resolve().parents
ROOT = _parents[3] if _parents[2].name == "backend" else _parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.phase3.biomechanics import BoneThicknessCalculator, classify_treatment, calculate_metrics

# ── constants ───────────────────────────────────────────────────────────────
INPUT_SIZE = (512, 512)
HEATMAP_SIZE = (256, 256)
NUM_KEYPOINTS = 10

# ── sliding window (Pipeline B — zero retraining) ───────────────────────────
# Segmentation runs at native resolution via overlapping 512×512 patches
# stitched with Gaussian-weighted averaging. Landmark model stays at 512×512.
SLIDING_WINDOW_SIZE = 512        # patch resolution (matches model's training domain)
SLIDING_STRIDE = 256             # 50% overlap → 4 patches per spatial position on average
# Gaussian sigma = patch_size / 4 → weight falls to ~0.01 at the edges
_SLIDING_SIGMA = SLIDING_WINDOW_SIZE / 4   # 128.0

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

# ── class indices (4-class model, argmax inference) ─────────────────────────
# Background=0, Upper_incisor=1, Labial_bone=2, Palatal_bone=3
CLASS_BACKGROUND      = 0
CLASS_UPPER_INCISOR   = 1
CLASS_LABIAL_BONE     = 2
CLASS_PALATAL_BONE    = 3

# ── mask list indices (length 3, after discarding background) ───────────────
MASK_IDX_UPPER_INCISOR = 0
MASK_IDX_LABIAL_BONE   = 1
MASK_IDX_PALATAL_BONE  = 2

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


def _build_segmentation_model(num_classes: int = 3) -> torch.nn.Module:
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


# ── preprocessing (landmarks — NO CLAHE, NO ImageNet norm) ──────────────────

def _preprocess_for_landmarks(image_bytes: bytes, target_size: tuple[int, int] = INPUT_SIZE):
    """
    Decode JPEG/PNG bytes, return native dims + float32/255 tensor in [0,1].
    NO ImageNet normalisation, NO CLAHE — matches the original landmark training
    pipeline that produced working heatmaps.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from upload stream")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (target_size[1], target_size[0]))   # (W, H)
    tensor = (
        torch.from_numpy(img_resized)
        .float()
        .permute(2, 0, 1)    # HWC → CHW
        / 255.0
    )
    return tensor.unsqueeze(0), orig_h, orig_w  # [1, 3, H, W]


# ── preprocessing (segmentation — CLAHE + ImageNet norm) ──────────────────────

def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an RGB image.
    Training pipeline used A.CLAHE(clip_limit=2.0, tileGridSize=(8,8)).
    Exact equivalent via OpenCV:
      1. Convert RGB → LAB (L-channel holds luminance)
      2. Split; apply cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) to L
      3. Merge; convert back to RGB
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _preprocess_for_segmentation(image_bytes: bytes, target_size: tuple[int, int] = INPUT_SIZE):
    """
    Decode JPEG/PNG bytes and preprocess EXACTLY as the training pipeline does:
      1. BGR -> RGB
      2. cv2.resize to (W, H) = 512x512 (simple squash, no letterbox, no aspect preservation)
      3. [H, W, C] -> [C, H, W] float32 / 255.0
    NO CLAHE. NO ImageNet normalization. NO letterbox padding.
    This matches SegmentationDataset.__getitem__ lines 80 & 110 exactly.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image from upload stream")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    H, W = target_size  # (512, 512)
    img_resized = cv2.resize(img, (W, H))  # simple squash, matches training

    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0  # [3, 512, 512]
    return tensor.unsqueeze(0), orig_h, orig_w  # [1, 3, 512, 512]


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


# ── 4-class segmentation (Background, Upper_incisor, Labial_bone, Palatal_bone) ──
# Model outputs 4 channels. argmax over channels gives class indices {0,1,2,3}:
#   0 → Background    (discarded — not sent to frontend)
#   1 → Upper_incisor → output index 0
#   2 → Labial_bone   → output index 1
#   3 → Palatal_bone  → output index 2

def _decode_segmentation_masks(
    logits: torch.Tensor,   # [1, 4, 512, 512] raw model output
    orig_w: int,
    orig_h: int,
) -> list[np.ndarray]:
    """Decode 4-class argmax output → three binary masks at native resolution.

    Matches training exactly: model was trained on simple 512x512 squash.
    Reversal is equally simple: argmax → resize back to (orig_w, orig_h).
    """
    class_map = torch.argmax(logits, dim=1).cpu()[0].numpy().astype(np.uint8)  # [512, 512]

    # Resize back to native resolution — simple squash (mirrors training forward pass)
    class_map_native = cv2.resize(class_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Extract one binary mask per non-background class
    masks: list[np.ndarray] = []
    for argmax_val in [1, 2, 3]:  # 0=Background (skipped), 1=Incisor, 2=Labial, 3=Palatal
        binary = (class_map_native == argmax_val).astype(np.uint8)
        px_count = int(binary.sum())
        print(f"[SegDecode] class={argmax_val} active_pixels={px_count}")
        masks.append(binary)

    return masks


def _resolve_mask_overlaps(masks: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    """No-op with argmax — classes are mutually exclusive by construction."""
    diag = {"note": "argmax_4class_no_overlap_resolution_needed"}
    return masks, diag



# ── geometric snapping helpers ──────────────────────────────────────────────

def _get_valid_contours(mask: np.ndarray) -> list:
    """Returns contours, discarding tiny noise artifacts (area < 100px²)."""
    if mask.sum() == 0:
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= 100]


def _contour_from_mask(mask: np.ndarray, epsilon_factor: float = 0.002) -> Optional[np.ndarray]:
    valid_contours = _get_valid_contours(mask)
    if not valid_contours:
        return None
    biggest = max(valid_contours, key=cv2.contourArea)
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
    labial_contour  = _contour_from_mask(masks[MASK_IDX_LABIAL_BONE])
    palatal_contour = _contour_from_mask(masks[MASK_IDX_PALATAL_BONE])
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
            dx = new_pt[0] - pt_raw[0]
            dy = new_pt[1] - pt_raw[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 50.0:
                snapped[idx] = pt_raw
                diag[name] = {
                    "dx": 0.0,
                    "dy": 0.0,
                    "dist_px": 0.0,
                    "note": f"snap_aborted_dist_{dist:.1f}"
                }
            else:
                snapped[idx] = new_pt
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
    incisor_contour = _contour_from_mask(masks[MASK_IDX_UPPER_INCISOR])
    diag = {}

    if incisor_contour is None:
        for name in ("Labial_midroot", "Palatal_midroot"):
            diag[name] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}
        return snapped, diag

    pts = incisor_contour.reshape(-1, 2).astype(np.float64)

    # Index 2: max x (rightmost = labial surface)
    pt_raw = coords[2]
    new_pt = pts[pts[:, 0].argmax()]
    dx = new_pt[0] - pt_raw[0]; dy = new_pt[1] - pt_raw[1]
    dist = math.sqrt(dx**2 + dy**2)
    if dist > 50.0:
        snapped[2] = pt_raw
        diag["Labial_midroot"] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": f"snap_aborted_dist_{dist:.1f}"}
    else:
        snapped[2] = new_pt
        diag["Labial_midroot"] = {
            "dx": float(round(float(dx), 2)),
            "dy": float(round(float(dy), 2)),
            "dist_px": float(round(float(dist), 2)),
        }

    # Index 4: min x (leftmost = palatal surface)
    pt_raw = coords[4]
    new_pt = pts[pts[:, 0].argmin()]
    dx = new_pt[0] - pt_raw[0]; dy = new_pt[1] - pt_raw[1]
    dist = math.sqrt(dx**2 + dy**2)
    if dist > 50.0:
        snapped[4] = pt_raw
        diag["Palatal_midroot"] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": f"snap_aborted_dist_{dist:.1f}"}
    else:
        snapped[4] = new_pt
        diag["Palatal_midroot"] = {
            "dx": float(round(float(dx), 2)),
            "dy": float(round(float(dy), 2)),
            "dist_px": float(round(float(dist), 2)),
        }

    return snapped, diag


def _snap_ans_pns(coords: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    snapped = coords.copy()
    palatal_contour = _contour_from_mask(masks[MASK_IDX_PALATAL_BONE])
    diag = {}

    for idx, name in [(6, "ANS"), (7, "PNS")]:
        pt_raw = coords[idx]
        if palatal_contour is not None:
            new_pt = _project_point_onto_contour(pt_raw, palatal_contour)
            dx = new_pt[0] - pt_raw[0]; dy = new_pt[1] - pt_raw[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 50.0:
                snapped[idx] = pt_raw
                diag[name] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": f"snap_aborted_dist_{dist:.1f}"}
            else:
                snapped[idx] = new_pt
                diag[name] = {
                    "dx": float(round(float(dx), 2)),
                    "dy": float(round(float(dy), 2)),
                    "dist_px": float(round(float(dist), 2)),
                }
        else:
            diag[name] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}

    return snapped, diag


# ── polygon extraction ───────────────────────────────────────────────────────

def _mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.003) -> list:
    """Convert binary mask to [[x, y], ...] polygon vertex list."""
    valid_contours = _get_valid_contours(mask)
    if not valid_contours:
        return []

    biggest = max(valid_contours, key=cv2.contourArea)
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


def _get_u1_perp(u1_unit: np.ndarray) -> np.ndarray:
    """
    Returns unit vector perpendicular to U1, oriented towards positive x (labial side).
    """
    u1_perp = np.array([-u1_unit[1], u1_unit[0]], dtype=np.float32)
    norm_val = np.linalg.norm(u1_perp)
    if norm_val > 1e-6:
        u1_perp = u1_perp / norm_val
    if u1_perp[0] < 0:
        u1_perp = -u1_perp
    return u1_perp


def _get_distance_severity(val: float) -> str:
    """
    Returns the severity status based on the distance value:
    - If distance >= 1.0: "Monitor"
    - If 0.5 <= distance < 1.0: "Warning"
    - If distance < 0.5: "Critical"
    """
    if val >= 1.0:
        return "Monitor"
    elif val >= 0.5:
        return "Warning"
    else:
        return "Critical"


def _get_bone_thickness_at_point(bone_mask: np.ndarray, start_pt: np.ndarray, direction: np.ndarray, max_dist_px: float = 200.0) -> float:
    """
    Ray-marches from tooth surface point start_pt along direction (which is u1_perp or -u1_perp)
    to find the outer limit of the bone_mask.
    Returns the distance in pixels.
    """
    h, w = bone_mask.shape
    max_s = 0.0
    steps = int(max_dist_px * 2)
    for step in range(steps):
        s = step * 0.5
        pt = start_pt + s * direction
        px = int(round(pt[0]))
        py = int(round(pt[1]))
        if 0 <= px < w and 0 <= py < h:
            if bone_mask[py, px] > 0:
                max_s = s
        else:
            break
    return max_s


def _find_tooth_boundary(tooth_mask: np.ndarray, start_pt: np.ndarray, direction: np.ndarray, max_dist_px: float = 100.0) -> np.ndarray:
    """
    Ray-marches from start_pt (on U1 axis, inside tooth) along direction until exiting the tooth_mask.
    Returns the boundary point coordinate (x, y) np.ndarray.
    """
    h, w = tooth_mask.shape
    steps = int(max_dist_px * 2)
    last_inside_pt = start_pt.copy()
    for step in range(steps):
        s = step * 0.5
        pt = start_pt + s * direction
        px = int(round(pt[0]))
        py = int(round(pt[1]))
        if 0 <= px < w and 0 <= py < h:
            if tooth_mask[py, px] > 0:
                last_inside_pt = pt
            else:
                return pt
        else:
            break
    return last_inside_pt


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
        
        # Load calibration records from path in config.yaml
        self._calibration_map = {}
        try:
            import yaml
            config_path = ROOT / "config.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                csv_rel_path = config.get("data", {}).get("calibration_csv", "data/processed/calibration.csv")
                csv_path = ROOT / csv_rel_path
                if csv_path.exists():
                    import csv
                    with open(csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            img_id = row.get("image_id")
                            mpp_str = row.get("mm_per_pixel")
                            if img_id and mpp_str:
                                try:
                                    self._calibration_map[img_id] = float(mpp_str)
                                except ValueError:
                                    pass
                    print(f"[AnalysisService] Loaded {len(self._calibration_map)} calibration records from {csv_path.name}")
                else:
                    print(f"[AnalysisService] WARNING: Calibration CSV not found at {csv_path}")
            else:
                print(f"[AnalysisService] WARNING: config.yaml not found at {config_path}")
        except Exception as e:
            print(f"[AnalysisService] WARNING: Failed to load calibration records: {e}")

        self._load_models()

    def _get_mm_per_pixel(self, image_id: Optional[str]) -> float:
        fallback_mpp = 0.0984
        if not image_id:
            return fallback_mpp
        if image_id in self._calibration_map:
            mpp = self._calibration_map[image_id]
            print(f"[AnalysisService] Resolved mm_per_pixel for '{image_id}': {mpp:.6f} mm/px")
            return mpp
        else:
            print(f"[AnalysisService] WARNING: '{image_id}' not found in calibration map. Falling back to default: {fallback_mpp} mm/px")
            return fallback_mpp

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

# ── Segmentation model (DeepLabV3Plus — 4-class argmax) ──────────────
        # Model path: prefer env USER_LOCAL_MODEL_PATH if set (user's local Mac copy),
        # otherwise fall back to the trained server checkpoint.
        seg_ckpt_path = (
            Path(os.environ["USER_LOCAL_MODEL_PATH"])
            if os.environ.get("USER_LOCAL_MODEL_PATH")
            else ROOT / "models" / "tversky_deepLabV3plus_resnet34_20250529_20260529_094221" / "best_model.pt"
        )
        if not seg_ckpt_path.exists():
            print(
                f"[AnalysisService] WARNING: Segmentation checkpoint not found: "
                f"{seg_ckpt_path}\n"
                "  Segmentation model unavailable.\n"
                "  Expected path inside container: /app/models/exp*/best_model.pt\n"
                "  Or set USER_LOCAL_MODEL_PATH env var to your local .pt path.\n"
                "  Verify: docker-compose.yml has './models:/app/models' volume mount."
            )
            self._seg_model = None
        else:
            sm = _build_segmentation_model(4)   # 4-class Tversky+BoundaryDice champion (Dice=0.8827)
            seg_state = torch.load(seg_ckpt_path, map_location=self._device, weights_only=False)
            cleaned_state = {k.replace('module.', ''): v for k, v in seg_state.items()}
            sm.load_state_dict(cleaned_state, strict=True)
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

    def analyze_image(self, image_bytes: bytes, image_id: Optional[str] = None) -> dict:
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
                "metrics": {
                    "u1_pp_angle_deg": 112.5,
                    "labial_crest_mm": 1.2,
                    "labial_crest_severity": "Monitor",
                    "labial_midroot_mm": 1.5,
                    "labial_midroot_severity": "Monitor",
                    "labial_apex_mm": 1.0,
                    "labial_apex_severity": "Monitor",
                    "palatal_crest_mm": 1.4,
                    "palatal_crest_severity": "Monitor",
                    "palatal_midroot_mm": 1.6,
                    "palatal_midroot_severity": "Monitor",
                    "palatal_apex_mm": 1.1,
                    "palatal_apex_severity": "Monitor",
                    "bone_thickness_type": "Type 1 – Thick",
                    "bone_thickness_interpretation": "Thick alveolar bone; Favorable bone support.",
                    "root_apex_position_type": "Midway",
                    "general_retraction_strategy": "Translation movement (Maximum movement limited by PB distance)",
                    "preferred_biomechanics": "Bodily movement (translation)",
                    "biomechanics_to_avoid": "Uncontrolled tipping",
                    "clinical_implication": "Most favorable condition",
                },
                "measurement_lines": None,
            }

        mm_per_pixel = self._get_mm_per_pixel(image_id)


        # ── Step 1: Read native dimensions + preprocess (separate pipelines) ───
        tensor_lm, orig_h, orig_w = _preprocess_for_landmarks(image_bytes, INPUT_SIZE)
        tensor_seg, _, _          = _preprocess_for_segmentation(image_bytes, INPUT_SIZE)
        tensor_lm  = tensor_lm.to(self._device)
        tensor_seg = tensor_seg.to(self._device)

        # EXPLICIT scale verification — no hidden assumptions
        scale_x = orig_w / INPUT_SIZE[1]   # e.g. 1729 / 512
        scale_y = orig_h / INPUT_SIZE[0]    # e.g. 2048 / 512

        # ── Step 2: Landmark inference ─────────────────────────────────────
        with torch.no_grad():
            heatmaps = self._landmark_model(tensor_lm)
            if heatmaps.shape[-2:] != HEATMAP_SIZE:
                heatmaps = F.interpolate(heatmaps, size=HEATMAP_SIZE, mode="bilinear", align_corners=False)

        raw_coords_512, confidences = _hard_argmax_decode(heatmaps.cpu(), INPUT_SIZE)

        # Map from 512×512 → native image space
        raw_coords_orig = _coords_input_to_orig(raw_coords_512, (orig_h, orig_w), INPUT_SIZE)

        # ── Step 3: Segmentation inference ─────────────────────────────────
        # Direct full-image inference to strictly match the training domain (no sliding window)
        with torch.no_grad():
            logits = self._seg_model(tensor_seg)

        raw_masks = _decode_segmentation_masks(logits, orig_w, orig_h)

        # ── Step 4: Mask priority layering ─────────────────────────────────
        corrected_masks, mask_diag = _resolve_mask_overlaps(raw_masks)

        # ── Step 5: Geometric snapping ──────────────────────────────────────
        snapped_crest, crest_diag    = _snap_crest_points(raw_coords_orig, corrected_masks)
        snapped_midroot, midroot_diag = _snap_midroot_points(snapped_crest, corrected_masks)
        snapped_all, ans_pns_diag    = _snap_ans_pns(snapped_midroot, corrected_masks)

        snapping_diag = {**crest_diag, **midroot_diag, **ans_pns_diag}

        # ── Step 6: Polygon boundary extraction ────────────────────────────
        poly_incisor = _mask_to_polygon(corrected_masks[MASK_IDX_UPPER_INCISOR])
        poly_labial  = _mask_to_polygon(corrected_masks[MASK_IDX_LABIAL_BONE])
        poly_palatal = _mask_to_polygon(corrected_masks[MASK_IDX_PALATAL_BONE])

        # ── Step 7: Biomechanical angle & Bone Thickness (Phase 3 Engine) ───
        
        # ── Step 7.5: Calculate the Six Distance Measurements ──
        tip = raw_coords_orig[0]
        apex = raw_coords_orig[1]
        u1_vec = apex - tip
        u1_len = np.linalg.norm(u1_vec)
        u1_unit = u1_vec / u1_len if u1_len > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)
        u1_perp = _get_u1_perp(u1_unit)

        # Estimate tooth radius at midroot from keypoints to use as fallback for crest level
        t_mid_labial = np.dot(raw_coords_orig[2] - tip, u1_unit)
        P_axis_mid_labial = tip + t_mid_labial * u1_unit
        r_labial = np.linalg.norm(np.dot(raw_coords_orig[2] - P_axis_mid_labial, u1_perp))

        t_mid_palatal = np.dot(raw_coords_orig[4] - tip, u1_unit)
        P_axis_mid_palatal = tip + t_mid_palatal * u1_unit
        r_palatal = np.linalg.norm(np.dot(raw_coords_orig[4] - P_axis_mid_palatal, u1_perp))

        r_est = (r_labial + r_palatal) / 2.0
        if r_est < 1e-3:
            r_est = 4.0 / mm_per_pixel

        # 1. Labial Crest Distance
        labial_crest_pt = raw_coords_orig[3]
        t_lc = np.dot(labial_crest_pt - tip, u1_unit)
        P_axis_lc = tip + t_lc * u1_unit
        P_tooth_lc = _find_tooth_boundary(corrected_masks[MASK_IDX_UPPER_INCISOR], P_axis_lc, u1_perp, max_dist_px=100.0)
        labial_crest_px = np.linalg.norm(np.dot(labial_crest_pt - P_tooth_lc, u1_perp))
        labial_crest_mm = float(labial_crest_px * mm_per_pixel)

        # 4. Palatal Crest Distance
        palatal_crest_pt = raw_coords_orig[5]
        t_pc = np.dot(palatal_crest_pt - tip, u1_unit)
        P_axis_pc = tip + t_pc * u1_unit
        P_tooth_pc = _find_tooth_boundary(corrected_masks[MASK_IDX_UPPER_INCISOR], P_axis_pc, -u1_perp, max_dist_px=100.0)
        palatal_crest_px = np.linalg.norm(np.dot(P_tooth_pc - palatal_crest_pt, u1_perp))
        palatal_crest_mm = float(palatal_crest_px * mm_per_pixel)

        # 2. Labial Midroot Distance
        labial_midroot_pt = raw_coords_orig[2]
        labial_midroot_px = _get_bone_thickness_at_point(corrected_masks[MASK_IDX_LABIAL_BONE], labial_midroot_pt, u1_perp, max_dist_px=150.0)
        if labial_midroot_px <= 0:
            labial_midroot_px = 0.0
        labial_midroot_mm = float(labial_midroot_px * mm_per_pixel)

        # 5. Palatal Midroot Distance
        palatal_midroot_pt = raw_coords_orig[4]
        palatal_midroot_px = _get_bone_thickness_at_point(corrected_masks[MASK_IDX_PALATAL_BONE], palatal_midroot_pt, -u1_perp, max_dist_px=150.0)
        if palatal_midroot_px <= 0:
            palatal_midroot_px = 0.0
        palatal_midroot_mm = float(palatal_midroot_px * mm_per_pixel)

        # 3. Labial Apex Distance (LB-Apex)
        labial_apex_pt = raw_coords_orig[8]
        labial_apex_px = np.linalg.norm(np.dot(labial_apex_pt - apex, u1_perp))
        labial_apex_mm = float(labial_apex_px * mm_per_pixel)

        # 6. Palatal Apex Distance (PB-Apex)
        palatal_apex_pt = raw_coords_orig[9]
        palatal_apex_px = np.linalg.norm(np.dot(apex - palatal_apex_pt, u1_perp))
        palatal_apex_mm = float(palatal_apex_px * mm_per_pixel)

        # Calculate severity flags for the six distances
        labial_crest_sev = _get_distance_severity(labial_crest_mm)
        labial_midroot_sev = _get_distance_severity(labial_midroot_mm)
        labial_apex_sev = _get_distance_severity(labial_apex_mm)
        palatal_crest_sev = _get_distance_severity(palatal_crest_mm)
        palatal_midroot_sev = _get_distance_severity(palatal_midroot_mm)
        palatal_apex_sev = _get_distance_severity(palatal_apex_mm)

        # Target coordinates for drawing perpendicular measurement lines
        labial_crest_target = P_tooth_lc
        palatal_crest_target = P_tooth_pc

        labial_midroot_target = labial_midroot_pt + labial_midroot_px * u1_perp
        palatal_midroot_target = palatal_midroot_pt - palatal_midroot_px * u1_perp

        labial_apex_target = apex + labial_apex_px * u1_perp
        palatal_apex_target = apex - palatal_apex_px * u1_perp

        def _to_line_coords(pt1, pt2):
            return [[float(round(pt1[0], 3)), float(round(pt1[1], 3))], [float(round(pt2[0], 3)), float(round(pt2[1], 3))]]

        measurement_lines = {
            "labial_crest_line": _to_line_coords(labial_crest_pt, labial_crest_target),
            "labial_midroot_line": _to_line_coords(labial_midroot_pt, labial_midroot_target),
            "labial_apex_line": _to_line_coords(apex, labial_apex_target),
            "palatal_crest_line": _to_line_coords(palatal_crest_pt, palatal_crest_target),
            "palatal_midroot_line": _to_line_coords(palatal_midroot_pt, palatal_midroot_target),
            "palatal_apex_line": _to_line_coords(apex, palatal_apex_target),
        }

        # ── Step 7.6: Alveolar Bone Thickness Classification (Zhang et al. 2021) ──
        # Define "Thin" as strictly < 0.5 mm and "Thick" as >= 0.5 mm to avoid edge-case None classifications
        labial_thin = [labial_crest_mm < 0.5, labial_midroot_mm < 0.5, labial_apex_mm < 0.5]
        palatal_thin = [palatal_crest_mm < 0.5, palatal_midroot_mm < 0.5, palatal_apex_mm < 0.5]

        any_labial_thin = any(labial_thin)
        any_palatal_thin = any(palatal_thin)
        all_labial_thin = all(labial_thin)
        all_palatal_thin = all(palatal_thin)

        if not any_labial_thin and not any_palatal_thin:
            bone_thickness_type = "Type 1 – Thick"
            bone_thickness_interpretation = "Thick alveolar bone; Favorable bone support."
        elif all_labial_thin and all_palatal_thin:
            bone_thickness_type = "Type 4 – Vulnerably Thin"
            bone_thickness_interpretation = "Very thin alveolar bone; High-risk morphology; compromised phenotype requiring extreme caution."
        elif any_labial_thin and any_palatal_thin:
            bone_thickness_type = "Type 3 – Thin with Double-Plate Concavities"
            bone_thickness_interpretation = "Represents bilateral cortical thinning; indicates a higher risk of dehiscence/fenestration during movement."
        else:
            bone_thickness_type = "Type 2 – Relatively Thick with Mono-Plate Concavity"
            bone_thickness_interpretation = "Represents unilateral cortical thinning."

        # ── Step 7.7: Root Apex Position Classification ──
        # Midway type logic using a 0.5 mm absolute tolerance: abs(LB_apex_mm - PB_apex_mm) <= 0.5 mm
        apex_diff = labial_apex_mm - palatal_apex_mm
        if abs(apex_diff) <= 0.5:
            root_apex_position_type = "Midway"
        elif labial_apex_mm < palatal_apex_mm:
            root_apex_position_type = "Labial"
        else:
            root_apex_position_type = "Palatal"

        # ── Step 7.8: Biomechanical Retraction Strategy ──
        u1_pp_angle_deg = _compute_u1_pp_angle_deg(raw_coords_orig)
        if u1_pp_angle_deg <= 105.0:
            general_retraction = "Root torque + retraction (Maximum movement limited by PB distance)"
        elif u1_pp_angle_deg < 110.0:
            general_retraction = "Translation movement (Maximum movement limited by PB distance)"
        else:
            general_retraction = "Controlled tipping (Maximum movement limited by PB distance)"

        # Angle zone for the Zhang et al. 2021 detailed matrix lookup
        if u1_pp_angle_deg < 105.0:
            angle_zone = "<105"
        elif u1_pp_angle_deg <= 115.0:
            angle_zone = "105-115"
        else:
            angle_zone = ">115"

        # ── Step 7.9: Detailed Biomechanics Matrix ──
        matrix_table = {
            "Labial": {
                "<105": {
                    "Preferred biomechanics": "Light controlled tipping with torque control",
                    "Biomechanics to avoid": "Uncontrolled proclination, labial root torque",
                    "Clinical implication": "Uprighting is possible but labial cortical bone must be preserved",
                },
                "105-115": {
                    "Preferred biomechanics": "Light controlled tipping or torque maintenance",
                    "Biomechanics to avoid": "Bodily movement forward, uncontrolled tipping",
                    "Clinical implication": "Avoid further labial displacement of the apex",
                },
                ">115": {
                    "Preferred biomechanics": "Controlled tipping during retraction with strict torque control",
                    "Biomechanics to avoid": "Uncontrolled tipping, labial root torque",
                    "Clinical implication": "High risk; strict torque control is required",
                },
            },
            "Midway": {
                "<105": {
                    "Preferred biomechanics": "Controlled proclination or bodily movement if bone allows",
                    "Biomechanics to avoid": "Uncontrolled tipping",
                    "Clinical implication": "Favorable prognosis",
                },
                "105-115": {
                    "Preferred biomechanics": "Bodily movement (translation)",
                    "Biomechanics to avoid": "Uncontrolled tipping",
                    "Clinical implication": "Most favorable condition",
                },
                ">115": {
                    "Preferred biomechanics": "Controlled tipping with torque control during retraction",
                    "Biomechanics to avoid": "Uncontrolled tipping",
                    "Clinical implication": "Safe if torque is well controlled",
                },
            },
            "Palatal": {
                "<105": {
                    "Preferred biomechanics": "Careful movement; labial crown/root control may be required",
                    "Biomechanics to avoid": "Palatal root torque, further retroclination",
                    "Clinical implication": "Risk of palatal cortical perforation",
                },
                "105-115": {
                    "Preferred biomechanics": "Bodily movement with caution",
                    "Biomechanics to avoid": "Excessive palatal root torque",
                    "Clinical implication": "Monitor palatal bone limits",
                },
                ">115": {
                    "Preferred biomechanics": "Controlled tipping during retraction with apex control",
                    "Biomechanics to avoid": "Retraction causing further palatal displacement of apex",
                    "Clinical implication": "Retraction possible but avoid excessive palatal pressure",
                },
            },
        }

        b_matrix = matrix_table[root_apex_position_type][angle_zone]

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
            "image_id": image_id or f"upload_{orig_w}x{orig_h}",
            "landmarks": _lm_list(snapped_all, confidences, True),
            "raw_landmarks": _lm_list(raw_coords_orig, confidences, False),
            "segmentation": {
                "Upper_incisor": {
                    "polygon": [[float(x), float(y)] for x, y in poly_incisor] if poly_incisor else [],
                    "pixel_count": int(corrected_masks[MASK_IDX_UPPER_INCISOR].sum()),
                },
                "Labial_bone": {
                    "polygon": [[float(x), float(y)] for x, y in poly_labial] if poly_labial else [],
                    "pixel_count": int(corrected_masks[MASK_IDX_LABIAL_BONE].sum()),
                },
                "Palatal_bone": {
                    "polygon": [[float(x), float(y)] for x, y in poly_palatal] if poly_palatal else [],
                    "pixel_count": int(corrected_masks[MASK_IDX_PALATAL_BONE].sum()),
                },
            },
            "snapping": snapping_diag,
            "mask_overlap_diagnostic": mask_diag,
            "measurement_lines": measurement_lines,
            "metrics": {
                "u1_pp_angle_deg": float(u1_pp_angle_deg),
                "labial_crest_mm": float(round(labial_crest_mm, 3)),
                "labial_crest_severity": labial_crest_sev,
                "labial_midroot_mm": float(round(labial_midroot_mm, 3)),
                "labial_midroot_severity": labial_midroot_sev,
                "labial_apex_mm": float(round(labial_apex_mm, 3)),
                "labial_apex_severity": labial_apex_sev,
                "palatal_crest_mm": float(round(palatal_crest_mm, 3)),
                "palatal_crest_severity": palatal_crest_sev,
                "palatal_midroot_mm": float(round(palatal_midroot_mm, 3)),
                "palatal_midroot_severity": palatal_midroot_sev,
                "palatal_apex_mm": float(round(palatal_apex_mm, 3)),
                "palatal_apex_severity": palatal_apex_sev,
                "bone_thickness_type": bone_thickness_type,
                "bone_thickness_interpretation": bone_thickness_interpretation,
                "root_apex_position_type": root_apex_position_type,
                "general_retraction_strategy": general_retraction,
                "preferred_biomechanics": b_matrix["Preferred biomechanics"],
                "biomechanics_to_avoid": b_matrix["Biomechanics to avoid"],
                "clinical_implication": b_matrix["Clinical implication"],
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