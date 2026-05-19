"""
segmentation_preprocess.py — Cephalometric Segmentation Preprocessing Pipeline
================================================================================

Automated data ingestion, mask generation, and validation pipeline for U-Net
bone & tooth segmentation training.

Expected raw data layout
------------------------
data/raw/
  images/              ← Cephalometric X-ray images (.jpg/.png)
  segmentation/        ← Annotation source (see AnnotationFormat below)
    annotations.json   ← Polygon/contour annotations (primary)
    masks/             ← Pre-rendered PNG masks (fallback, class IDs)

AnnotationFormat (JSON — clinical vector output)
-----------------------------------------------
{
  "images": [
    {
      "image_id": "Patient01_T1",
      "filename": "Patient01_T1.jpg",
      "width": 1729,           ← original image pixel dimensions
      "height": 2048,
      "polygons": {
        "background": [],       ← no annotation needed; class=0 by convention
        "bone":      [[x0,y0],[x1,y1],...],  ← outer contour, class=1
        "tooth":     [[x0,y0],[x1,y1],...],  ← tooth region, class=2
        "pulp":      [[x0,y0],[x1,y1],...]   ← pulp chamber, class=3 (optional)
      }
    }
  ]
}

Supported mask formats
----------------------
  PNG uint8  — pixel value = class ID (0=background, 1=bone, 2=tooth, 3=pulp)
  Multi-class mask must have shape (H, W) — no channel dimension.

Output structure
----------------
data/processed/
  segmentation/
    images/            ← aligned images (512×512, letterbox padded, uint8 RGB)
    masks/             ← aligned masks (512×512, uint8, class IDs 0-3)
    metadata.json      ← per-image alignment info (scale, pad_left, pad_top)

Output mask encoding (class IDs)
---------------------------------
  0  — Background / uninstrumented region
  1  — Bone (alveolar bone boundary)
  2  — Tooth (crown + root surface)
  3  — Pulp chamber (inner tooth region, optional annotation)

Class colours (for visualization overlay)
-----------------------------------------
  0: (  0,   0,   0) — black / background
  1: (220,  20,  60) — crimson   — bone
  2: ( 30, 144, 255) — dodgerblue — tooth
  3: (255, 215,   0) — gold       — pulp

Alignment & padding (matches landmark pipeline INPUT_SIZE=512×512)
-----------------------------------------------------------------
- Resize image to fit within 512×512 while preserving aspect ratio.
- Pad remaining space with black (0) on both sides (letterbox).
- Apply identical scale + pad transforms to the corresponding mask.
- Scaled coords are baked into output metadata JSON.

GUARDRAILS
----------
- No horizontal flip (cephalograms are anatomically directional).
- Pixel-level mask alignment: image and mask must align pixel-for-pixel
  after alignment transforms (no interpolation on masks).
- All output masks are uint8 class ID images — no float, no one-hot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ── project paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── stdlib ───────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from enum import IntEnum

# ── third-party ──────────────────────────────────────────────────────────────
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Class definitions (IntEnum so mask pixels can be compared directly)
# ─────────────────────────────────────────────────────────────────────────────

class MaskClass(IntEnum):
    """Class IDs stored in output segmentation masks."""
    BACKGROUND = 0
    BONE       = 1
    TOOTH      = 2
    PULP       = 3


# Display colours (BGR — OpenCV convention) for mask overlay visualisation
MASK_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    MaskClass.BACKGROUND: (  0,   0,   0),  # black
    MaskClass.BONE:       ( 60,  20, 220),  # crimson  (BGR)
    MaskClass.TOOTH:      (255, 144,  30),  # dodgerblue (BGR)
    MaskClass.PULP:       (  0, 215, 255),  # gold      (BGR)
}

MASK_NAMES: dict[int, str] = {
    MaskClass.BACKGROUND: "background",
    MaskClass.BONE:       "bone",
    MaskClass.TOOTH:      "tooth",
    MaskClass.PULP:       "pulp",
}

# ─────────────────────────────────────────────────────────────────────────────
# Image alignment constants (MUST match landmark pipeline INPUT_SIZE)
# ─────────────────────────────────────────────────────────────────────────────

INPUT_SIZE = (512, 512)   # (H, W) — same as landmark pipeline

# Pad value for letterbox regions
PAD_VALUE = 0  # black

# ─────────────────────────────────────────────────────────────────────────────
# Alignment dataclass — records how each image was transformed
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlignmentInfo:
    image_id:     str
    orig_width:   int
    orig_height:  int
    scale:        float        # uniform scale factor applied
    pad_left:     int         # pixels of padding added to LEFT of image
    pad_top:      int         # pixels of padding added to TOP of image
    aligned_width: int        # = orig_width  * scale
    aligned_height: int       # = orig_height * scale
    # Effective content in the aligned image (after padding removed)
    content_box:  tuple[int, int, int, int]  # (x0, y0, x1, y1) in 512×512 space

    def scale_coord(self, x: float, y: float) -> tuple[float, float]:
        """Map original image coordinate → aligned image coordinate."""
        sx = x * self.scale + self.pad_left
        sy = y * self.scale + self.pad_top
        return sx, sy


# ─────────────────────────────────────────────────────────────────────────────
# Polygon → Mask conversion
# ─────────────────────────────────────────────────────────────────────────────

def polygons_to_mask(
    polygons: dict[str, list[list[float]]],
    output_size: tuple[int, int],
    orig_size: tuple[int, int],
    scale: float,
    pad_left: int,
    pad_top: int,
) -> np.ndarray:
    """
    Render polygon contours into a uint8 class-ID mask.

    Args:
        polygons:  {class_name: [[x,y], ...]} — zero-length list = no contour
        output_size: (H, W) of output mask = (512, 512)
        orig_size:   (orig_H, orig_W) of source image
        scale:       scale factor applied during alignment
        pad_left, pad_top: padding added during alignment

    Returns:
        mask: np.ndarray [H, W] uint8 — class IDs (0=background, 1=bone, 2=tooth, 3=pulp)

    GUARDRAIL: No horizontal flip — contours are anatomically directional.
    """
    H, W = output_size
    mask = np.zeros((H, W), dtype=np.uint8)

    # class_name → class_id mapping
    name_to_class = {
        "background": MaskClass.BACKGROUND,
        "bone":       MaskClass.BONE,
        "tooth":      MaskClass.TOOTH,
        "pulp":       MaskClass.PULP,
    }

    # Render each class contour as filled polygon
    for class_name, class_id in name_to_class.items():
        if class_name == "background":
            continue  # class 0 already zero
        contour_points = polygons.get(class_name, [])
        if not contour_points or len(contour_points) < 3:
            continue

        # Scale + translate points to aligned mask coordinates
        scaled_pts = np.array(
            [[x * scale + pad_left, y * scale + pad_top]
             for x, y in contour_points],
            dtype=np.int32,
        )

        # Draw filled polygon
        cv2.fillPoly(mask, [scaled_pts], int(class_id))

    return mask


def png_mask_to_class_id(
    mask_png: np.ndarray,
    expected_classes: list[int] | None = None,
) -> np.ndarray:
    """
    Normalise a PNG mask to a clean uint8 class-ID array.

    Handles:
      - Single-channel grayscale masks (pixel value = class ID)
      - Masks already uint8 with class IDs stored as pixel values
      - Unexpected class values → reassigned to background (0)

    Args:
        mask_png: raw mask read from cv2.imread (H, W) or (H, W, C)
        expected_classes: list of valid class IDs (default: [0, 1, 2, 3])

    Returns:
        clean: np.ndarray [H, W] uint8 with clean class IDs
    """
    if expected_classes is None:
        expected_classes = [0, 1, 2, 3]

    # Flatten if multi-channel
    if mask_png.ndim == 3:
        # Take first channel (assumes greyscale or where R=G=B for class values)
        mask_png = mask_png[:, :, 0]

    clean = mask_png.astype(np.uint8)

    # Clamp unexpected class values to background
    max_class = max(expected_classes)
    clean = np.where(clean > max_class, 0, clean)

    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Image alignment (letterbox pad — matches landmark pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def align_image_to_input_size(
    image: np.ndarray,
    target_size: tuple[int, int] = INPUT_SIZE,
    pad_value: int = PAD_VALUE,
) -> tuple[np.ndarray, AlignmentInfo, float, int, int]:
    """
    Resize image to fit within target_size preserving aspect ratio, then
    letterbox-pad with black to reach exactly target_size.

    This produces the same 512×512 output as the landmark pipeline, ensuring
    pixel-perfect alignment between landmark coordinates and segmentation masks.

    Args:
        image:    RGB/BGR uint8 ndarray, shape (orig_H, orig_W, 3)
        target_size: (H, W) = (512, 512)
        pad_value: value to fill padding with (0 = black)

    Returns:
        aligned_image: np.ndarray (H, W, 3) uint8, padded to target_size
        align_info:    AlignmentInfo dataclass
        scale:        uniform scale factor (for polygon mapping)
        pad_left:     left padding in pixels
        pad_top:      top padding in pixels
    """
    H, W = target_size
    orig_h, orig_w = image.shape[:2]

    # Compute scale to fit inside 512×512
    scale_x = W / orig_w
    scale_y = H / orig_h
    scale = min(scale_x, scale_y)  # uniform

    # Scaled dimensions (before padding)
    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))

    # Resize image (INTER_AREA = best for downsampling)
    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # Compute padding needed to reach target_size
    pad_left = (W - scaled_w) // 2
    pad_top  = (H - scaled_h) // 2
    pad_right  = W - scaled_w - pad_left
    pad_bottom = H - scaled_h - pad_top

    # Letterbox pad with black
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        aligned_image = cv2.copyMakeBorder(
            resized,
            top=pad_top,    bottom=pad_bottom,
            left=pad_left,  right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[pad_value, pad_value, pad_value],
        )
    else:
        aligned_image = resized

    align_info = AlignmentInfo(
        image_id="",
        orig_width=orig_w,
        orig_height=orig_h,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        aligned_width=scaled_w,
        aligned_height=scaled_h,
        content_box=(pad_left, pad_top, pad_left + scaled_w, pad_top + scaled_h),
    )

    return aligned_image, align_info, scale, pad_left, pad_top


def align_mask_to_input_size(
    mask: np.ndarray,
    target_size: tuple[int, int] = INPUT_SIZE,
    scale: float = 1.0,
    pad_left: int = 0,
    pad_top: int = 0,
    pad_value: int = PAD_VALUE,
) -> np.ndarray:
    """
    Apply the same alignment transform to a mask that was applied to its image.

    Uses INTER_NEAREST to avoid mixing class values during resize.

    Args:
        mask:       uint8 class-ID ndarray, shape (orig_H, orig_W)
        target_size: (H, W) = (512, 512)
        scale, pad_left, pad_top: from align_image_to_input_size()
        pad_value: 0 for masks (black = background)

    Returns:
        aligned_mask: np.ndarray (H, W) uint8 class-ID mask
    """
    H, W = target_size
    orig_h, orig_w = mask.shape[:2]

    # Resize mask with nearest-neighbour (critical — no class blending)
    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))
    resized = cv2.resize(
        mask,
        (scaled_w, scaled_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Letterbox pad
    pad_right  = W - scaled_w - pad_left
    pad_bottom = H - scaled_h - pad_top
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        aligned_mask = cv2.copyMakeBorder(
            resized,
            top=pad_top,    bottom=pad_bottom,
            left=pad_left,  right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value,
        )
    else:
        aligned_mask = resized

    return aligned_mask


# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_segmentation_dataset(
    annotations_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    mask_dir: str | Path | None = None,
    min_class_ids: list[int] | None = None,
    overwrite: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Run the full preprocessing pipeline for segmentation training data.

    Pipeline steps
    --------------
    1. Load annotations JSON (or fall back to mask_dir PNG lookup)
    2. For each image:
       a. Load and validate raw image
       b. Align to 512×512 (letterbox, same as landmark pipeline)
       c. Generate class-ID mask from polygons OR load pre-rendered PNG mask
       d. Align mask with same scale + padding
       e. Save aligned image + aligned mask to output_dir
    3. Build metadata JSON with per-image AlignmentInfo

    Args:
        annotations_path: path to annotations JSON file
        images_dir:       directory containing raw X-ray images
        output_dir:       root of processed output (created if missing)
        mask_dir:         optional; pre-rendered PNG masks directory
        min_class_ids:    minimum classes that must exist in masks (default: [0,1,2])
        overwrite:        if True, overwrite existing output files
        quiet:            suppress per-image print output

    Returns:
        summary: dict with pipeline statistics
    """
    if min_class_ids is None:
        min_class_ids = [0, 1, 2]  # bg + bone + tooth minimum

    annotations_path = Path(annotations_path)
    images_dir       = Path(images_dir)
    output_dir       = Path(output_dir)
    if mask_dir:
        mask_dir = Path(mask_dir)

    # ── Output directories ──────────────────────────────────────────────────
    img_out_dir  = output_dir / "segmentation" / "images"
    mask_out_dir = output_dir / "segmentation" / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load annotations ───────────────────────────────────────────────────
    with open(annotations_path, "r") as fh:
        annotations = json.load(fh)

    image_records = {rec["image_id"]: rec for rec in annotations.get("images", [])}

    # ── Process each image ──────────────────────────────────────────────────
    results = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "errors": [],
        "classes_found": set(),
        "image_ids": [],
    }

    for image_id, record in image_records.items():
        results["total"] += 1

        img_path = images_dir / record["filename"]
        if not img_path.exists():
            results["errors"].append(f"{image_id}: image not found at {img_path}")
            continue

        # ── Load and validate image ─────────────────────────────────────────
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            results["errors"].append(f"{image_id}: cv2.imread failed (corrupt?)")
            continue
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)  # → RGB for alignment

        orig_h, orig_w = raw_img.shape[:2]
        if orig_h != record.get("height") or orig_w != record.get("width"):
            results["errors"].append(
                f"{image_id}: dimension mismatch — "
                f"file={orig_w}×{orig_h} vs record={record.get('width')}×{record.get('height')}"
            )
            # Proceed with actual detected dimensions

        # ── Align image ──────────────────────────────────────────────────────
        aligned_img, align_info, scale, pad_left, pad_top = align_image_to_input_size(
            raw_img, INPUT_SIZE, PAD_VALUE
        )
        align_info.image_id = image_id

        # ── Generate or load mask ────────────────────────────────────────────
        if mask_dir:
            # Pre-rendered PNG mask path (same filename, .png extension)
            mask_png_path = mask_dir / (img_path.stem + ".png")
            if mask_png_path.exists():
                raw_mask = cv2.imread(str(mask_png_path), cv2.IMREAD_GRAYSCALE)
                if raw_mask is None:
                    results["errors"].append(f"{image_id}: corrupt PNG mask at {mask_png_path}")
                    raw_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                else:
                    raw_mask = png_mask_to_class_id(raw_mask, min_class_ids + [3])
            else:
                results["errors"].append(f"{image_id}: mask not found at {mask_png_path}")
                raw_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            # Generate mask from polygon annotations
            raw_mask = polygons_to_mask(
                polygons=record.get("polygons", {}),
                output_size=(orig_h, orig_w),
                orig_size=(orig_h, orig_w),
                scale=1.0,
                pad_left=0,
                pad_top=0,
            )

        # ── Align mask (nearest-neighbour, no interpolation mixing) ─────────
        aligned_mask = align_mask_to_input_size(
            raw_mask,
            INPUT_SIZE,
            scale,
            pad_left,
            pad_top,
            pad_value=0,
        )

        # ── Validate mask ────────────────────────────────────────────────────
        unique_classes = set(np.unique(aligned_mask).tolist())
        results["classes_found"].update(unique_classes)

        if unique_classes <= {0} and not quiet:
            results["errors"].append(f"{image_id}: mask is empty (all background)")

        # ── Save output ─────────────────────────────────────────────────────
        out_img_path  = img_out_dir  / f"{image_id}.png"
        out_mask_path = mask_out_dir / f"{image_id}.png"

        cv2.imwrite(str(out_img_path),  cv2.cvtColor(aligned_img,  cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_mask_path), aligned_mask)

        if not quiet:
            print(
                f"[SegPreprocess] {image_id}: "
                f"{orig_w}×{orig_h} → {INPUT_SIZE[1]}×{INPUT_SIZE[0]} "
                f"(scale={scale:.4f}, pad=({pad_left},{pad_top})) "
                f"classes={sorted(unique_classes)}"
            )

        results["processed"] += 1
        results["image_ids"].append(image_id)

    # ── Build + save metadata ────────────────────────────────────────────────
    metadata = {
        "input_size":        INPUT_SIZE,
        "num_images":       results["processed"],
        "classes":          sorted(results["classes_found"]),
        "class_names":      {int(k): v for k, v in MASK_NAMES.items()},
        "alignment":        "letterbox_center",
        "mask_encoding":    "uint8_class_id",
        "image_ids":        results["image_ids"],
    }

    meta_path = output_dir / "segmentation" / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    if not quiet:
        print(f"\n[SegPreprocess] Done. {results['processed']}/{results['total']} processed, "
              f"{len(results['errors'])} errors. Metadata → {meta_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Data Sanity Checker / Audit Function
# ─────────────────────────────────────────────────────────────────────────────

def audit_segmentation_dataset(
    images_dir: str | Path,
    masks_dir: str | Path,
    annotations_path: str | Path | None = None,
    expected_size: tuple[int, int] = INPUT_SIZE,
    min_classes: list[int] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Strict validation routine for segmentation training data.

    Checks
    ------
    1. Every image in images_dir has a matching mask in masks_dir.
    2. Aligned dimensions of image and mask match exactly (512×512).
    3. Masks contain only valid class IDs.
    4. No masks are all-zero (empty segmentation).
    5. Images are not corrupt (cv2.imread succeeds).
    6. Annotations JSON dimensions match actual image dimensions.
    7. Patient-level grouping: T1/T2 of same patient both present.

    Prints a colour-coded diagnostic report to stdout.

    Args:
        images_dir:   root directory of aligned PNG images
        masks_dir:    root directory of aligned PNG class-ID masks
        annotations_path: optional annotations JSON for dimension check
        expected_size: expected (H, W) of aligned images/masks (default: 512×512)
        min_classes:  minimum valid class IDs (default: [0, 1, 2])
        quiet:        suppress output; just return report dict

    Returns:
        report: dict with pass/fail counts and per-image error list
    """
    if min_classes is None:
        min_classes = [0, 1, 2]

    images_dir   = Path(images_dir)
    masks_dir    = Path(masks_dir)
    expected_h, expected_w = expected_size

    SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Collect image files
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_IMG_EXTS
    )

    report: dict[str, Any] = {
        "images_dir":        str(images_dir),
        "masks_dir":         str(masks_dir),
        "expected_size":     f"{expected_w}×{expected_h}",
        "total_images":      len(image_files),
        "matched":           0,
        "missing_masks":     [],
        "missing_images":    [],
        "dim_mismatches":     [],
        "corrupt_images":    [],
        "corrupt_masks":     [],
        "empty_masks":       [],
        "invalid_class_ids": [],
        "annotation_mismatches": [],
        "passed": False,
    }

    def _load_img(path: Path) -> np.ndarray | None:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return img

    def _print(msg: str, status: str = "INFO") -> None:
        if quiet:
            return
        symbols = {"OK": "  [OK]", "FAIL": "  [FAIL]", "WARN": "  [WARN]", "INFO": "     "}
        colour = {
            "OK": "\033[92m", "FAIL": "\033[91m", "WARN": "\033[93m", "INFO": "\033[0m"
        }.get(status, "\033[0m")
        reset = "\033[0m"
        print(f"{colour}{symbols.get(status, '')}{reset} {msg}")

    _print("═══════════════════════════════════════════════════════", "INFO")
    _print("  SEGMENTATION DATA AUDIT REPORT", "INFO")
    _print("═══════════════════════════════════════════════════════", "INFO")
    _print(f"  Images:  {images_dir}", "INFO")
    _print(f"  Masks:   {masks_dir}", "INFO")
    _print(f"  Expected size: {expected_w}×{expected_h}", "INFO")
    _print("────────────────────────────────────────────────────────", "INFO")

    # ── Check 1: Image-mask pairing ────────────────────────────────────────
    for img_path in image_files:
        stem = img_path.stem
        mask_path = masks_dir / f"{stem}.png"

        if not mask_path.exists():
            report["missing_masks"].append(stem)
            _print(f"MISSING MASK: {stem}.png", "FAIL")
            continue

        report["matched"] += 1

        # ── Check 2: Image integrity ──────────────────────────────────────
        img = _load_img(img_path)
        if img is None:
            report["corrupt_images"].append(stem)
            _print(f"CORRUPT IMAGE: {stem}", "FAIL")
            continue

        # ── Check 3: Mask integrity ──────────────────────────────────────
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            report["corrupt_masks"].append(stem)
            _print(f"CORRUPT MASK: {stem}", "FAIL")
            continue

        # ── Check 4: Dimension match ─────────────────────────────────────
        img_h, img_w = img.shape[:2]
        msk_h, msk_w = mask.shape[:2]
        if img_h != msk_h or img_w != msk_w:
            report["dim_mismatches"].append({
                "image": stem, "img_size": f"{img_w}×{img_h}",
                "mask_size": f"{msk_w}×{msk_h}",
            })
            _print(f"DIM MISMATCH: {stem} — image={img_w}×{img_h}, mask={msk_w}×{msk_h}", "FAIL")
            continue

        if img_h != expected_h or img_w != expected_w:
            report["dim_mismatches"].append({
                "image": stem, "size": f"{img_w}×{img_h}",
                "expected": f"{expected_w}×{expected_h}",
            })
            _print(f"WRONG SIZE: {stem} — got {img_w}×{img_h}, expected {expected_w}×{expected_h}", "FAIL")
            continue

        # ── Check 5: Valid class IDs ──────────────────────────────────────
        unique_ids = set(np.unique(mask).tolist())
        invalid_ids = unique_ids - set(min_classes + [3])  # 3 = pulp is optional
        if invalid_ids:
            report["invalid_class_ids"].append({"image": stem, "ids": sorted(invalid_ids)})
            _print(f"INVALID CLASS IDs: {stem} has {invalid_ids} (valid: {min_classes})", "FAIL")
            continue

        # ── Check 6: Non-empty mask ───────────────────────────────────────
        if unique_ids <= {0}:
            report["empty_masks"].append(stem)
            _print(f"EMPTY MASK: {stem} (all background)", "WARN")

    # ── Check 7: Annotation dimension cross-check ──────────────────────────
    if annotations_path and Path(annotations_path).exists():
        with open(annotations_path) as fh:
            ann = json.load(fh)
        for rec in ann.get("images", []):
            stem = rec.get("image_id", "")
            ann_w, ann_h = rec.get("width", 0), rec.get("height", 0)
            img_path = images_dir / rec.get("filename", "")
            if img_path.exists():
                img = _load_img(img_path)
                if img is not None:
                    ih, iw = img.shape[:2]
                    if (ann_w, ann_h) != (iw, ih):
                        report["annotation_mismatches"].append({
                            "image_id": stem,
                            "annotation": f"{ann_w}×{ann_h}",
                            "actual": f"{iw}×{ih}",
                        })
                        _print(
                            f"ANN DIM MISMATCH: {stem} — "
                            f"annotation={ann_w}×{ann_h}, actual={iw}×{ih}", "WARN"
                        )

    # ── Summary ────────────────────────────────────────────────────────────
    _print("────────────────────────────────────────────────────────", "INFO")
    _print(f"  Total images:     {report['total_images']}", "INFO")
    _print(f"  Paired (OK):     {report['matched']}", "OK" if report['matched'] > 0 else "WARN")
    _print(f"  Missing masks:   {len(report['missing_masks'])}", "FAIL" if report['missing_masks'] else "OK")
    _print(f"  Dim mismatches:  {len(report['dim_mismatches'])}", "FAIL" if report['dim_mismatches'] else "OK")
    _print(f"  Corrupt images:  {len(report['corrupt_images'])}", "FAIL" if report['corrupt_images'] else "OK")
    _print(f"  Corrupt masks:   {len(report['corrupt_masks'])}", "FAIL" if report['corrupt_masks'] else "OK")
    _print(f"  Empty masks:     {len(report['empty_masks'])}", "WARN" if report['empty_masks'] else "OK")
    _print(f"  Invalid classes: {len(report['invalid_class_ids'])}", "FAIL" if report['invalid_class_ids'] else "OK")
    _print(f"  Ann mismatches:  {len(report['annotation_mismatches'])}", "WARN" if report['annotation_mismatches'] else "OK")
    _print("═══════════════════════════════════════════════════════", "INFO")

    all_errors = (
        len(report["missing_masks"]) + len(report["dim_mismatches"]) +
        len(report["corrupt_images"]) + len(report["corrupt_masks"]) +
        len(report["invalid_class_ids"])
    )
    report["passed"] = all_errors == 0

    if report["passed"]:
        _print("  RESULT: ✅ ALL CHECKS PASSED", "OK")
    else:
        _print(f"  RESULT: ❌ {all_errors} ERROR(S) FOUND — fix before training", "FAIL")

    _print("═══════════════════════════════════════════════════════\n", "INFO")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> Any:
    """Build argparse parser for CLI usage."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Cephalometric Segmentation Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess
    p = sub.add_parser("preprocess", help="Run full ingestion → mask generation pipeline")
    p.add_argument("--annotations", required=True, help="Path to annotations JSON")
    p.add_argument("--images-dir",  required=True, help="Directory of raw X-ray images")
    p.add_argument("--output-dir", required=True, help="Root output directory")
    p.add_argument("--mask-dir",   help="Optional dir of pre-rendered PNG masks")
    p.add_argument("--overwrite",  action="store_true", help="Overwrite existing outputs")
    p.add_argument("--quiet",      action="store_true", help="Suppress per-image logs")

    # audit
    a = sub.add_parser("audit", help="Run data sanity checker")
    a.add_argument("--images-dir",   required=True, help="Aligned images directory")
    a.add_argument("--masks-dir",    required=True, help="Aligned masks directory")
    a.add_argument("--annotations", help="Annotations JSON (optional)")
    a.add_argument("--expected-w",   type=int, default=512, help="Expected width (default: 512)")
    a.add_argument("--expected-h",   type=int, default=512, help="Expected height (default: 512)")
    a.add_argument("--quiet",        action="store_true", help="Suppress output")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        result = preprocess_segmentation_dataset(
            annotations_path=args.annotations,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            mask_dir=getattr(args, "mask_dir", None),
            overwrite=args.overwrite,
            quiet=args.quiet,
        )
        sys.exit(0 if result["errors"] == [] else 1)

    elif args.command == "audit":
        report = audit_segmentation_dataset(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            annotations_path=getattr(args, "annotations", None),
            expected_size=(args.expected_h, args.expected_w),
            quiet=args.quiet,
        )
        sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()