#!/usr/bin/env python3
"""
src/phase2/inference.py
=======================
Cohort-wide inference with GEOMETRIC CONSTRAINTS for web-ready JSON export.

Fixes applied:
  1. LANDMARK SNAPPING: Crest (3=Labial_crest, 5=Palatal_crest) and Midroot
     (2=Labial_midroot, 4=Palatal_midroot) points are re-projected onto the
     nearest anatomical boundary of their respective bone/tooth masks.
     Uses Shapely contour extraction + OpenCV approxPolyDP boundary matching.
  2. MASK PRIORITY LAYERING: Palatal_bone (class 2) yields territory to
     Upper_incisor (class 0) → zero overlap, zero gap. Order:
       (a) Palatal_bone raw mask
       (b) Upper_incisor raw mask
       (c) Palatal_bone = Palatal_bone AND (NOT Upper_incisor)
  3. WEB-READY JSON: data/processed/biomechanical_features.json with snapped
     landmark coordinates (x,y,confidence) + corrected polygon boundaries.

Dynamic report: live-stream PNS/ANS coordinate shifts + overlap pixel counts.
"""

from __future__ import annotations

import json, sys, warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# ── device / sizes ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (512, 512)
HEATMAP_SIZE = (256, 256)
NUM_KEYPOINTS = 10

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]

# Segmentation class indices
CLASS_UPPER_INCISOR = 0
CLASS_LABIAL_BONE   = 1
CLASS_PALATAL_BONE  = 2

# Mask overlay colours (B, G, R) for visualisation
MASK_COLORS = {
    "Upper_incisor": (255, 80, 80),   # Red
    "Labial_bone":   (80, 255, 80),   # Green
    "Palatal_bone":  (80, 80, 255),   # Blue
}

# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK MODEL (HRNet-W32 — matches src/phase2/model.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_landmark_model(num_keypoints: int = NUM_KEYPOINTS) -> torch.nn.Module:
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

    return CephalometricModel(num_keypoints)


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION MODEL (DeepLabV3Plus — matches scripts/visualize_test_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_segmentation_model(num_classes: int = 3) -> torch.nn.Module:
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


def find_best_segmentation_checkpoint(models_dir: Path) -> Path:
    """Return best_model.pt from DeepLabV3Plus exp dir with highest Dice."""
    candidates = []
    if not models_dir.is_dir():
        return models_dir / "best_model.pt"
    for sub in sorted(models_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("exp") or "DeepLabV3Plus" not in sub.name:
            continue
        bm = sub / "best_model.pt"
        if bm.exists():
            dice_str = sub.name.split("_dice")[1].split("_")[0] if "_dice" in sub.name else "0"
            try:
                dice = float(dice_str)
            except ValueError:
                dice = 0.0
            candidates.append((dice, bm))
    if not candidates:
        return models_dir / "best_model.pt"
    candidates.sort(reverse=True)
    return candidates[0][1]


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: Path, input_size: tuple[int, int] = INPUT_SIZE) -> torch.Tensor:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size[1], input_size[0]))   # (W, H)
    tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)    # [1, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# LANDMARK DECODING — hard per-channel argmax (NOT soft-argmax)
# ─────────────────────────────────────────────────────────────────────────────

def hard_argmax_decode(
    heatmaps: torch.Tensor,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel hard-argmax: each keypoint finds its own local peak independently.
    Returns coords [N, 2] and confidences [N] in input_size pixel space.
    """
    B, N, H, W = heatmaps.shape
    sig = torch.sigmoid(heatmaps).cpu().numpy()   # [B, N, H, W]

    coords_list, conf_list = [], []
    for c in range(N):
        hm = sig[0, c]                    # [H, W]
        flat_idx = hm.argmax()
        y_hm, x_hm = divmod(flat_idx, W)
        conf = float(hm[y_hm, x_hm])
        x_in = x_hm * (input_size[1] / W)
        y_in = y_hm * (input_size[0] / H)
        coords_list.append([x_in, y_in])
        conf_list.append(conf)

    return np.array(coords_list, dtype=np.float32), np.array(conf_list, dtype=np.float32)


def coords_input_to_orig(
    coords_input: np.ndarray,
    orig_size: tuple[int, int],
    input_size: tuple[int, int],
) -> np.ndarray:
    """Map coords from INPUT_SIZE space → native original image space."""
    orig_h, orig_w = orig_size
    inp_h, inp_w   = input_size
    out = coords_input.copy()
    out[:, 0] = coords_input[:, 0] * (orig_w / inp_w)
    out[:, 1] = coords_input[:, 1] * (orig_h / inp_h)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MASK POST-PROCESSING — priority layering (Upper_incisor > Palatal_bone)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_mask_overlaps(
    masks: list[np.ndarray],
) -> tuple[list[np.ndarray], dict]:
    """
    Enforce priority layering so Upper_incisor (class 0) carves out its
    territory from Palatal_bone (class 2), leaving zero overlap pixels.

    Order of operations:
      1. Palatal_bone (class 2) — raw mask, unchanged.
      2. Upper_incisor  (class 0) — raw mask, unchanged.
      3. Palatal_bone  = Palatal_bone AND (NOT Upper_incisor)

    Returns corrected masks list + diagnostic dict with overlap pixel counts.
    """
    palatal  = masks[CLASS_PALATAL_BONE].copy().astype(bool)
    incisor  = masks[CLASS_UPPER_INCISOR].copy().astype(bool)
    labial   = masks[CLASS_LABIAL_BONE].copy().astype(bool)

    overlap_before = int(np.logical_and(palatal, incisor).sum())

    # Priority rule: Upper_incisor wins over Palatal_bone
    palatal_corrected = np.logical_and(palatal, ~incisor).astype(np.uint8)

    # Recompute overlap after correction
    overlap_after = int(np.logical_and(palatal_corrected, incisor).sum())

    corrected = [incisor.astype(np.uint8), labial.astype(np.uint8), palatal_corrected]

    diag = {
        "overlap_before": overlap_before,
        "overlap_after":  overlap_after,
        "pixels_corrected": overlap_before - overlap_after,
    }
    return corrected, diag


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC LANDMARK SNAPPING — Shapely + OpenCV contour projection
# ─────────────────────────────────────────────────────────────────────────────

def _contour_from_mask(mask: np.ndarray, epsilon_factor: float = 0.002) -> Optional[np.ndarray]:
    """Extract outer boundary contour from a binary mask via OpenCV.findContours."""
    if mask.sum() == 0:
        return None
    # RETR_EXTERNAL gives the outermost contour only
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Largest contour by area = outer boundary of this structure
    biggest = max(contours, key=cv2.contourArea)
    # Smooth + approximate to reduce noise while preserving shape
    perimeter = cv2.arcLength(biggest, closed=True)
    if perimeter < 1.0:
        return biggest
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(biggest, epsilon, closed=True)
    return approx


def _project_point_onto_contour(pt: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Project a 2D point onto a closed contour and return the nearest boundary pixel.
    Returns the contour point closest to `pt` (in Euclidean distance).
    """
    if contour is None or len(contour) == 0:
        return pt.copy()
    # contour shape: [N, 1, 2] → [N, 2]
    pts = contour.reshape(-1, 2).astype(np.float64)
    # Compute distances to all contour vertices
    dists = np.linalg.norm(pts - pt.astype(np.float64), axis=1)
    nearest_idx = int(dists.argmin())
    return pts[nearest_idx].astype(np.float32)


def snap_crest_points(
    coords: np.ndarray,
    masks: list[np.ndarray],
    keypoint_names: list[str] = KEYPOINT_NAMES,
) -> tuple[np.ndarray, dict]:
    """
    Snap crest points (index 3=Labial_crest, 5=Palatal_crest) onto the
    nearest anatomical peak of their respective bone boundary contours.

    Crest points are defined as the most coronal/extreme points on the bone
    ridge. We find the point on the contour with minimum y-coordinate
    (highest on screen = smallest y value in image space) that is also
    close to the predicted location — this represents the true bone crest.

    Returns snapped coords + diagnostic dict with per-point shift distances.
    """
    snapped = coords.copy()

    # Contours
    labial_contour  = _contour_from_mask(masks[CLASS_LABIAL_BONE])
    palatal_contour = _contour_from_mask(masks[CLASS_PALATAL_BONE])

    diag = {}

    # Index 3: Labial_crest → snap to Labial_bone contour peak
    pt_raw = coords[3]   # (x, y) in original image space
    if labial_contour is not None:
        # Peak = contour point with minimum y (most coronal)
        crest_pts = labial_contour.reshape(-1, 2).astype(np.float64)
        # Among points within 2x sigma of raw prediction y, pick the most coronal
        y_center = pt_raw[1]
        y_tolerance = 60.0   # px — generous tolerance for crest search window
        candidates = crest_pts[np.abs(crest_pts[:, 1] - y_center) < y_tolerance]
        if len(candidates) > 0:
            # Pick point with minimum y (coronal-most in image coords)
            peak_idx = candidates[:, 1].argmin()
            new_pt = candidates[peak_idx]
        else:
            # Fallback: nearest contour point
            new_pt = _project_point_onto_contour(pt_raw, labial_contour)
        snapped[3] = new_pt
        dx = new_pt[0] - pt_raw[0]
        dy = new_pt[1] - pt_raw[1]
        dist = np.sqrt(dx**2 + dy**2)
        diag["Labial_crest"] = {"dx": round(dx, 2), "dy": round(dy, 2), "dist_px": round(dist, 2)}
    else:
        diag["Labial_crest"] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}

    # Index 5: Palatal_crest → snap to Palatal_bone contour peak
    pt_raw = coords[5]
    if palatal_contour is not None:
        crest_pts = palatal_contour.reshape(-1, 2).astype(np.float64)
        y_center = pt_raw[1]
        y_tolerance = 60.0
        candidates = crest_pts[np.abs(crest_pts[:, 1] - y_center) < y_tolerance]
        if len(candidates) > 0:
            peak_idx = candidates[:, 1].argmin()
            new_pt = candidates[peak_idx]
        else:
            new_pt = _project_point_onto_contour(pt_raw, palatal_contour)
        snapped[5] = new_pt
        dx = new_pt[0] - pt_raw[0]
        dy = new_pt[1] - pt_raw[1]
        dist = np.sqrt(dx**2 + dy**2)
        diag["Palatal_crest"] = {"dx": round(dx, 2), "dy": round(dy, 2), "dist_px": round(dist, 2)}
    else:
        diag["Palatal_crest"] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}

    return snapped, diag


def snap_midroot_points(
    coords: np.ndarray,
    masks: list[np.ndarray],
    keypoint_names: list[str] = KEYPOINT_NAMES,
) -> tuple[np.ndarray, dict]:
    """
    Snap midroot points (index 2=Labial_midroot, 4=Palatal_midroot) onto the
    left/right tooth surface boundaries of the Upper_incisor mask.

    - Point 2 (Labial_midroot)  → rightmost (max x) point on Upper_incisor contour
    - Point 4 (Palatal_midroot) → leftmost  (min x) point on Upper_incisor contour

    These represent the labial and palatal root surface intersections.
    """
    snapped = coords.copy()
    incisor_contour = _contour_from_mask(masks[CLASS_UPPER_INCISOR])
    diag = {}

    if incisor_contour is None:
        diag["Labial_midroot"]   = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}
        diag["Palatal_midroot"]  = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}
        return snapped, diag

    pts = incisor_contour.reshape(-1, 2).astype(np.float64)

    # Index 2: Labial_midroot → max x (rightmost = labial surface in standard orientation)
    pt_raw = coords[2]
    new_pt_lab = pts[pts[:, 0].argmax()]
    snapped[2] = new_pt_lab
    dx = new_pt_lab[0] - pt_raw[0]; dy = new_pt_lab[1] - pt_raw[1]
    diag["Labial_midroot"] = {"dx": round(dx, 2), "dy": round(dy, 2), "dist_px": round(np.sqrt(dx**2+dy**2), 2)}

    # Index 4: Palatal_midroot → min x (leftmost = palatal surface)
    pt_raw = coords[4]
    new_pt_pal = pts[pts[:, 0].argmin()]
    snapped[4] = new_pt_pal
    dx = new_pt_pal[0] - pt_raw[0]; dy = new_pt_pal[1] - pt_raw[1]
    diag["Palatal_midroot"] = {"dx": round(dx, 2), "dy": round(dy, 2), "dist_px": round(np.sqrt(dx**2+dy**2), 2)}

    return snapped, diag


def snap_ans_pns(
    coords: np.ndarray,
    masks: list[np.ndarray],
    keypoint_names: list[str] = KEYPOINT_NAMES,
) -> tuple[np.ndarray, dict]:
    """
    ANS (index 6) and PNS (index 7) are anatomical reference landmarks.
    Snap each to the nearest point on the Palatal_bone contour boundary
    (the maxillary suture line), preserving anatomical fidelity.
    """
    snapped = coords.copy()
    palatal_contour = _contour_from_mask(masks[CLASS_PALATAL_BONE])
    diag = {}

    for idx, name in [(6, "ANS"), (7, "PNS")]:
        pt_raw = coords[idx]
        if palatal_contour is not None:
            new_pt = _project_point_onto_contour(pt_raw, palatal_contour)
            snapped[idx] = new_pt
            dx = new_pt[0] - pt_raw[0]; dy = new_pt[1] - pt_raw[1]
            diag[name] = {"dx": round(dx, 2), "dy": round(dy, 2), "dist_px": round(np.sqrt(dx**2+dy**2), 2)}
        else:
            diag[name] = {"dx": 0.0, "dy": 0.0, "dist_px": 0.0, "note": "no_contour"}

    return snapped, diag


# ─────────────────────────────────────────────────────────────────────────────
# POLYGON EXTRACTION FROM CORRECTED MASKS → list of points per class
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.003) -> list:
    """Convert binary mask to a list of [x, y] polygon vertices."""
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_cohort_inference(
    image_ids: list[str],
    landmark_model: torch.nn.Module,
    seg_model: torch.nn.Module,
    image_dir: Path,
    orig_sizes: dict,
    output_path: Path,
    verbose: bool = True,
) -> dict:
    """
    Run geometric-constrained inference across a cohort of images.

    For each image:
      1. Run landmark inference → raw coords
      2. Run segmentation inference → raw masks
      3. Resolve mask overlaps (priority layering)
      4. Apply geometric snapping to crest/midroot/ANS-PNS points
      5. Extract corrected polygons for each class
      6. Export snapped landmark + polygon data to JSON

    Returns the full output dict for post-processing / web integration.
    """
    n = len(image_ids)
    all_results = []
    cumulative_diag = {
        "total_images": n,
        "ans_shifts": [],
        "pns_shifts": [],
        "crest_shifts": [],
        "overlap_before": [],
        "overlap_after": [],
    }

    for i, image_id in enumerate(image_ids):
        if verbose:
            print(f"  [{i+1}/{n}] Processing {image_id} ...")

        # — locate image file —
        img_path = None
        for ext in [".jpg", ".JPG", ".png", ".PNG"]:
            p = image_dir / f"{image_id}{ext}"
            if p.exists():
                img_path = p; break
        if img_path is None:
            print(f"    WARNING: image not found for {image_id}, skipping")
            continue

        orig_h, orig_w = orig_sizes.get(image_id, (2048, 1729))

        # — preprocess —
        img_tensor = preprocess_image(img_path).to(DEVICE)

        # — landmark inference —
        landmark_model.eval()
        with torch.no_grad():
            pred = landmark_model(img_tensor)
            if pred.shape[-2:] != HEATMAP_SIZE:
                pred = F.interpolate(pred, size=HEATMAP_SIZE, mode="bilinear", align_corners=False)
        raw_coords_input, confidences = hard_argmax_decode(pred.cpu(), INPUT_SIZE)
        raw_coords_orig = coords_input_to_orig(raw_coords_input, (orig_h, orig_w), INPUT_SIZE)

        # — segmentation inference —
        seg_model.eval()
        with torch.no_grad():
            logits = seg_model(img_tensor)
            sig = torch.sigmoid(logits).cpu()[0].numpy()   # [3, 512, 512]

        # Raw binary masks at 512x512, then resize to native
        raw_masks_512 = [(sig[c] > 0.5).astype(np.uint8) for c in range(3)]
        raw_masks = [
            cv2.resize(raw_masks_512[c], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            for c in range(3)
        ]

        # — resolve mask overlaps (priority layering) —
        corrected_masks, mask_diag = resolve_mask_overlaps(raw_masks)
        cumulative_diag["overlap_before"].append(mask_diag["overlap_before"])
        cumulative_diag["overlap_after"].append(mask_diag["overlap_after"])

        if verbose:
            print(f"    Overlap pixels: before={mask_diag['overlap_before']}, after={mask_diag['overlap_after']}")

        # — snap landmarks (crest → bone, midroot → tooth, ANS/PNS → palatal) —
        snapped_crest, crest_diag = snap_crest_points(raw_coords_orig, corrected_masks)
        snapped_midroot, midroot_diag = snap_midroot_points(snapped_crest, corrected_masks)
        snapped_all, ans_pns_diag = snap_ans_pns(snapped_midroot, corrected_masks)

        # Collect diagnostic shifts
        for name, info in [("ANS", ans_pns_diag.get("ANS", {})), ("PNS", ans_pns_diag.get("PNS", {}))]:
            cumulative_diag[f"{name.lower()}_shifts"].append(info)
        cumulative_diag["crest_shifts"].append({**crest_diag, **midroot_diag})

        if verbose:
            for name, info in ans_pns_diag.items():
                print(f"    {name} shift: dx={info.get('dx',0):.1f}, dy={info.get('dy',0):.1f}, dist={info.get('dist_px',0):.1f}px")

        # — extract corrected polygons —
        poly_incisor = mask_to_polygon(corrected_masks[CLASS_UPPER_INCISOR])
        poly_labial  = mask_to_polygon(corrected_masks[CLASS_LABIAL_BONE])
        poly_palatal = mask_to_polygon(corrected_masks[CLASS_PALATAL_BONE])

        # — assemble result record —
        result = {
            "image_id": image_id,
            "filename": img_path.name,
            "width": orig_w,
            "height": orig_h,
            "landmarks": [
                {
                    "name": KEYPOINT_NAMES[k],
                    "x": float(round(snapped_all[k, 0], 3)),
                    "y": float(round(snapped_all[k, 1], 3)),
                    "confidence": float(round(confidences[k], 4)),
                    "snapped": k in (2, 3, 4, 5, 6, 7),
                }
                for k in range(NUM_KEYPOINTS)
            ],
            "raw_landmarks": [
                {
                    "name": KEYPOINT_NAMES[k],
                    "x": float(round(raw_coords_orig[k, 0], 3)),
                    "y": float(round(raw_coords_orig[k, 1], 3)),
                    "confidence": float(round(confidences[k], 4)),
                }
                for k in range(NUM_KEYPOINTS)
            ],
            "snapping": {
                **crest_diag,
                **midroot_diag,
                **ans_pns_diag,
            },
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
            "mask_overlap_diagnostic": mask_diag,
        }
        all_results.append(result)

    # — write JSON —
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj):
        """Recursively convert numpy types + non-serializable objects to native Python."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = _sanitize(all_results)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return {"results": all_results, "diagnostic": cumulative_diag}


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_dynamic_report(diagnostic: dict, test_image_ids: list[str]):
    """Print live-stream breakdown of PNS/ANS shifts and overlap resolution."""
    print("\n" + "=" * 70)
    print(" DYNAMIC REPORT — GEOMETRIC SNAPPING & MASK CORRECTION")
    print("=" * 70)

    print(f"\n{'Image ID':<22} {'ANS dx':>8} {'ANS dy':>8} {'PNS dx':>8} {'PNS dy':>8} {'Overlap bef':>12} {'Overlap aft':>11}")
    print("-" * 85)

    shifts = diagnostic.get("ans_shifts", [])
    pns_shifts = diagnostic.get("pns_shifts", [])
    overlap_bef = diagnostic.get("overlap_before", [])
    overlap_aft = diagnostic.get("overlap_after", [])

    for i, img_id in enumerate(test_image_ids):
        ans = shifts[i] if i < len(shifts) else {}
        pns = pns_shifts[i] if i < len(pns_shifts) else {}
        ob = overlap_bef[i] if i < len(overlap_bef) else 0
        oa = overlap_aft[i] if i < len(overlap_aft) else 0
        print(f"  {img_id:<20} {ans.get('dx',0):>8.2f} {ans.get('dy',0):>8.2f} "
              f"{pns.get('dx',0):>8.2f} {pns.get('dy',0):>8.2f} "
              f"{ob:>12} {oa:>11}")

    print("\n" + "-" * 85)
    total_bef = sum(overlap_bef)
    total_aft = sum(overlap_aft)
    print(f"  {'TOTALS':<20} {'':>8} {'':>8} {'':>8} {'':>8} "
          f"{total_bef:>12} {total_aft:>11}")

    print(f"\n  ✅ Overlapping pixel count DROPPED TO ZERO: {total_aft == 0}")
    print(f"     Total pixels corrected: {total_bef - total_aft}")

    # Crest point shift summary
    crest_shifts = diagnostic.get("crest_shifts", [])
    print(f"\n{'Crest Snapping Summary':}")
    print(f"  {'Image ID':<22} {'L.Crest dx':>10} {'L.Crest dy':>10} {'P.Crest dx':>10} {'P.Crest dy':>10}")
    print("-" * 65)
    for i, img_id in enumerate(test_image_ids):
        cs = crest_shifts[i] if i < len(crest_shifts) else {}
        lc = cs.get("Labial_crest", {})
        pc = cs.get("Palatal_crest", {})
        print(f"  {img_id:<20} {lc.get('dx',0):>10.2f} {lc.get('dy',0):>10.2f} "
              f"{pc.get('dx',0):>10.2f} {pc.get('dy',0):>10.2f}")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(" GEOMETRIC-CONSTRAINED COHORT INFERENCE")
    print(" Geometric Snapping + Mask Priority Layering → Web-Ready JSON")
    print("=" * 70)

    # ── 1. Load models ────────────────────────────────────────────────────────
    print("\n[1] Loading models ...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {DEVICE}")

# Landmark
    landmark_ckpt_candidates = [
        ROOT / "outputs" / "checkpoints" / "fold1_best.pth",
        ROOT / "data" / "processed" / "checkpoints" / "fold1_best.pth",
    ]
    landmark_ckpt = None
    for p in landmark_ckpt_candidates:
        if p.exists():
            landmark_ckpt = p
            break
    if landmark_ckpt is None:
        print(f"ERROR: fold1_best.pth not found in {[str(p) for p in landmark_ckpt_candidates]}")
        sys.exit(1)
    landmark_model = build_landmark_model(NUM_KEYPOINTS)
    ckpt = torch.load(landmark_ckpt, map_location=DEVICE, weights_only=False)
    state = {k: v for k, v in ckpt.get("model_state_dict", ckpt).items() if "uncertainty" not in k}
    landmark_model.load_state_dict(state, strict=False)
    landmark_model = landmark_model.to(DEVICE)
    landmark_model.eval()
    print(f"  Landmark: {landmark_ckpt.name}")

    # Segmentation
    models_base = ROOT / "models"
    seg_ckpt = find_best_segmentation_checkpoint(models_base)
    seg_model = build_segmentation_model(3)
    seg_state = torch.load(seg_ckpt, map_location=DEVICE, weights_only=False)
    seg_model.load_state_dict(seg_state, strict=False)
    seg_model = seg_model.to(DEVICE)
    seg_model.eval()
    print(f"  Segmentation: {seg_ckpt.parent.name}")

    # ── 2. Load cohort record metadata ───────────────────────────────────────
    print("\n[2] Loading cohort metadata ...")
    with open(ROOT / "data" / "processed" / "landmarks_clean.json") as f:
        landmarks_data = json.load(f)

    records = landmarks_data["images"] if isinstance(landmarks_data, dict) else landmarks_data

    # Build image_id → orig_size lookup
    orig_sizes = {}
    for rec in records:
        orig_sizes[rec["image_id"]] = (rec["height"], rec["width"])

    # ── 3. Select test cohort (01–04 T1) ─────────────────────────────────────
    print("\n[3] Selecting test cohort ...")
    test_image_ids = [f"Patient{i:02d}_T1" for i in range(1, 5)]
    # Verify existence
    image_dir = ROOT / "data" / "raw" / "images"
    found_ids = []
    for img_id in test_image_ids:
        exists = any((image_dir / f"{img_id}{ext}").exists() for ext in [".jpg", ".JPG", ".png"])
        if exists:
            found_ids.append(img_id)
        else:
            print(f"  WARNING: {img_id} not found — skipping")
    print(f"  Test cohort: {found_ids}")

    # ── 4. Run inference ─────────────────────────────────────────────────────
    print("\n[4] Running geometric-constrained inference ...")
    output_path = ROOT / "data" / "processed" / "biomechanical_features.json"
    result = run_cohort_inference(
        image_ids=found_ids,
        landmark_model=landmark_model,
        seg_model=seg_model,
        image_dir=image_dir,
        orig_sizes=orig_sizes,
        output_path=output_path,
        verbose=True,
    )

    # ── 5. Dynamic report ─────────────────────────────────────────────────────
    print_dynamic_report(result["diagnostic"], found_ids)

    print(f"\n[5] Web-ready JSON exported → {output_path}")
    print(f"     Records: {len(result['results'])}, Classes: Upper_incisor / Labial_bone / Palatal_bone")
    print("=" * 70)


if __name__ == "__main__":
    main()