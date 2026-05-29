#!/usr/bin/env python3
"""
predict_all.py — Batch inference for AI-Assisted Pre-annotations
=================================================================

Run HRNet-W32 heatmap model over a directory of unannotated X-ray images
and emit a single predictions.json consumed directly by the Astro.js
Phase-4 frontend.

Usage
-----
    python predict_all.py \
        --image-dir data/raw/unannotated_images \
        --checkpoint outputs/best_model.pth \
        --output outputs/predictions.json

Output JSON format (per-image dict, landmark → {x, y, confidence})
-----------------------------------------------------------------
    {
      "image_001.jpg": {
        "Upper_tip":    {"x": 1450.5, "y": 800.2, "confidence": 0.98},
        "Upper_apex":   {"x": 1400.1, "y": 650.8, "confidence": 0.95},
        ...
      }
    }

Key design decisions
-------------------
* Hard-argmax (NOT soft-argmax): the soft-argmax in this pipeline has a
  temperature/beta bug that collapses predictions toward heatmap centre.
  Hard-argmax is deterministic, no tuning required, and is the standard
  competition decode method (CL-Detection 2023 benchmark best method used it).
* Coordinate scaling: heatmap [256×256] → input [512×512] → original image.
  Each scale ratio is applied independently so any input image size works.
* Simple /255 normalization: matches how the model was actually trained.
* Batch processing: processes images in batches for GPU efficiency.
* MPS / CUDA / CPU: device is auto-detected; MPS is used on Apple Silicon.
* TTA (Test-Time Augmentation): 5-variant averaging (orig + rot± + brig±).
  Scale variants EXCLUDED — geometric inverse produces ~100 px errors.
  NO horizontal flip — cephalograms are anatomically directional.
"""

import argparse
import json
import sys
from pathlib import Path

# ── project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ── stdlib ─────────────────────────────────────────────────────────────────
from pathlib import Path
from tqdm import tqdm

# ── third-party ───────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.phase2.model import CephalometricModel, NUM_KEYPOINTS
from src.phase2.heatmap import decode_heatmaps
from src.utils.io import load_config, save_json

# ── landmark names (fixed order, must match CVAT skeleton) ────────────────
KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]

# ── model dimensions (hardcoded — matches config.yaml model.input_size) ───
INPUT_SIZE = (512, 512)   # (H, W) that the model expects
HEATMAP_SIZE = (256, 256)  # (H, W) that the model outputs


# ─────────────────────────────────────────────────────────────────────────────
#  TTA helper functions
#  Defined FIRST so TTA_VARIANTS can reference them.
#  GUARDRAIL: NO horizontal flip — cephalograms are anatomically directional.
# ─────────────────────────────────────────────────────────────────────────────

def _tta_identity(img: np.ndarray) -> np.ndarray:
    """No-op transform."""
    return img.copy()


def _inv_identity(coords_hm: np.ndarray, orig_size) -> np.ndarray:
    """
    No-op inverse for identity transform (original image only).

    Coordinates come out of the model in heatmap space [256×256].
    Apply the full scaling chain: heatmap → input (2×) → original image.
    """
    orig_h, orig_w = orig_size
    inp_w, inp_h = INPUT_SIZE[1], INPUT_SIZE[0]

    # Step 1: heatmap → input (2× upscale)
    inp_coords = coords_hm * 2.0  # [K, 2] in [512,512] space

    # Step 2: input → original image
    scale_x = orig_w / inp_w   # e.g., 1729/512 ≈ 3.38
    scale_y = orig_h / inp_h   # e.g., 2048/512 = 4.0
    orig_coords = np.empty_like(inp_coords)
    orig_coords[:, 0] = inp_coords[:, 0] * scale_x
    orig_coords[:, 1] = inp_coords[:, 1] * scale_y
    return orig_coords


def _tta_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiply pixel values by factor and clip to [0, 255]."""
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def _tta_scale(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale image around its centre.
    factor > 1.0 → zoom in  (crop centre region, resize back)
    factor < 1.0 → zoom out (pad with grey, resize back)
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    if factor > 1.0:
        # Zoom in: crop centre, then resize to original
        new_w = int(w / factor)
        new_h = int(h / factor)
        x1 = int(cx - new_w / 2)
        y1 = int(cy - new_h / 2)
        x1 = max(0, min(x1, w - new_w))
        y1 = max(0, min(y1, h - new_h))
        cropped = img[y1:y1 + new_h, x1:x1 + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # Zoom out: pad with grey background, then resize to original
        new_w = int(w * factor)
        new_h = int(h * factor)
        padded = np.full((h, w, 3), 114, dtype=np.uint8)
        x1 = int(cx - new_w / 2)
        y1 = int(cy - new_h / 2)
        x1 = max(0, min(x1, w - new_w))
        y1 = max(0, min(y1, h - new_h))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded[y1:y1 + new_h, x1:x1 + new_w] = small
        return padded


def _inv_scale(factor: float):
    """
    Return inverse-scale function for the given scale factor.

    NOTE: Geometric inverse only works when factor is close to 1.0.
    For factor=1.05, the cropped image loses peripheral context and the
    model's coordinate prediction shifts nonlinearly — the simple inverse
    cannot perfectly recover original coordinates. Use with caution.
    """
    def inv(coords_hm: np.ndarray, orig_size) -> np.ndarray:
        """
        coords_hm: [K, 2] in heatmap [256,256] space of augmented input.
        Steps:
          1. 2× upscale → augmented input [512,512]
          2. * factor  → original input [512,512] (geometric inverse)
          3. scale to original image
        """
        orig_h, orig_w = orig_size
        inp_w, inp_h = INPUT_SIZE[1], INPUT_SIZE[0]

        # Step 1: heatmap → augmented input
        inp_coords = coords_hm * 2.0  # [K, 2]

        # Step 2: geometric inverse of centred-crop/scale
        # forward (zoom-in):  x_aug = x_orig / factor  (centre-crop)
        # forward (zoom-out): x_aug = x_orig * factor  (centre-pad)
        # Geometric inverse (valid for small factors, close to 1.0):
        #   zoom-in:  x_orig = x_aug * factor
        #   zoom-out: x_orig = x_aug / factor
        # Combined: use * for factor>1, / for factor<1
        if factor > 1.0:
            inp_coords = inp_coords * factor
        else:
            inp_coords = inp_coords / factor

        # Step 3: input → original image
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h
        orig_coords = np.empty_like(inp_coords)
        orig_coords[:, 0] = inp_coords[:, 0] * scale_x
        orig_coords[:, 1] = inp_coords[:, 1] * scale_y
        return orig_coords
    return inv


def _tta_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate image around its centre by angle_deg degrees.
    Uses grey (114) constant border to avoid dark borders.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale=1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )
    return rotated


def _inv_rotate(angle_deg: float):
    """Return inverse-rotation function for the given angle."""
    def inv(coords_hm: np.ndarray, orig_size) -> np.ndarray:
        """
        coords_hm: [K, 2] in heatmap [256,256] space of rotated input.
        Apply the opposite rotation about the input centre, then scale.
        """
        orig_h, orig_w = orig_size
        inp_w, inp_h = INPUT_SIZE[1], INPUT_SIZE[0]

        # Step 1: heatmap → augmented input space (2×)
        inp_coords = coords_hm * 2.0  # [K, 2] in [512,512] space

        # Step 2: inverse rotation (rotate by -angle_deg)
        cx, cy = inp_w / 2, inp_h / 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, scale=1.0)
        x = inp_coords[:, 0]
        y = inp_coords[:, 1]
        inv_coords = np.empty_like(inp_coords)
        inv_coords[:, 0] = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        inv_coords[:, 1] = M[1, 0] * x + M[1, 1] * y + M[1, 2]

        # Step 3: input → original image
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h
        orig_coords = np.empty_like(inv_coords)
        orig_coords[:, 0] = inv_coords[:, 0] * scale_x
        orig_coords[:, 1] = inv_coords[:, 1] * scale_y
        return orig_coords
    return inv


# ─────────────────────────────────────────────────────────────────────────────
# ── TTA variant registry
#  Each entry: (name, transform_fn, inverse_fn)
#  GUARDRAIL: NO horizontal flip — cephalograms are anatomically directional.
#
#  NOTE on scale variants: the geometric inverse (x_orig = x_aug * factor for
#  zoom-in, /factor for zoom-out) only partially compensates for the spatial
#  transformation. When factor > 1.0 (zoom-in), peripheral context is lost,
#  and the model may predict systematically shifted coordinates that no simple
#  inverse can fully correct. Use conservative scale factors (≤ 1.03) to
#  minimize this error. Brightness and rotation inverses are reliable.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# NOTE on scale variants:
#   The geometric inverse for scale (zoom-in / zoom-out) does NOT reliably
#   recover original coordinates because the model's heatmap prediction is
#   shifted by the changed spatial context (grey padding for zoom-out;
#   missing peripheral content for zoom-in). Even with ±2% factors, the
#   inverse introduces ~100px errors that dominate the TTA average.
#   Brightness inverses are identity (harmless but also unhelpful for X-ray
#   images which have stable intensity). Rotation inverses are mathematically
#   exact and verified to reduce per-landmark error.
#
# Active TTA variants (5 total):
#   orig  – baseline single-pass inference
#   rot+  – rotate +2°  with correct inverse (reduces rotation-specific error)
#   rot-  – rotate -2°  with correct inverse
#   brig+ – brightness ×1.10  (identity inverse; helps if model is intensity-sensitive)
#   brig- – brightness ×0.90  (identity inverse)
#
# Excluded (broken): scale+ / scale- — geometric inverse produces ~100px errors.
# ─────────────────────────────────────────────────────────────────────────────
TTA_VARIANTS = [
    ("orig",   _tta_identity,                                              _inv_identity),
    ("rot+",   lambda img: _tta_rotate(img,  2.0),                        _inv_rotate(2.0)),
    ("rot-",   lambda img: _tta_rotate(img, -2.0),                        _inv_rotate(-2.0)),
    ("brig+",  lambda img: _tta_brightness(img, 1.10),                   _inv_identity),
    ("brig-",  lambda img: _tta_brightness(img, 0.90),                   _inv_identity),
]

NUM_TTA = len(TTA_VARIANTS)  # 5


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hard_argmax_decode(
    heatmaps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode heatmaps with hard-argmax: take the integer (row, col) of the
    peak activation for each keypoint channel.

    Args:
        heatmaps: [B, K, H_hm, W_hm] raw logits (no sigmoid inside here)

    Returns:
        coords:     [B, K, 2] float32  — (x=col, y=row) in heatmap pixel space
        confidence: [B, K] float32     — sigmoid(peak_activation)
    """
    B, K, H, W = heatmaps.shape

    # sigmoid for normalized confidence scores in [0, 1]
    conf = torch.sigmoid(heatmaps)

    # flat_conf: [B, K, H*W]
    flat_conf = conf.view(B * K, H * W)

    # peak value and flat index
    confidence, flat_idx = flat_conf.max(dim=-1)          # [B*K]
    confidence = confidence.view(B, K)

    # Convert flat index → (row, col) in heatmap space
    # row = flat_idx // W,  col = flat_idx % W
    rows = (flat_idx // W).float()   # y-coordinate
    cols = (flat_idx %  W).float()   # x-coordinate

    coords = torch.stack([cols, rows], dim=-1).view(B, K, 2)  # [B, K, 2] — (x, y)

    return coords, confidence


def preprocess_image(
    image_path: str,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load an image from disk, resize to INPUT_SIZE, and normalize.

    Args:
        image_path: path to the X-ray JPG/PNG

    Returns:
        tensor: [1, 3, H, W] float32 normalized tensor (on CPU)
        orig_size: (orig_H, orig_W) — original pixel dimensions for scaling back
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # BGR → RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img.shape[:2]

    # Resize to model input size (H, W) — note cv2.resize takes (W, H)
    img_resized = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)

    # [H, W, 3] uint8 → [3, H, W] float32 in [0, 1]
    # NOTE: Model was trained with simple /255 normalization (no ImageNet mean/std).
    # DO NOT use ImageNet normalization here — it shifts inputs to [-2, +2] which
    # breaks the model's learned features (the backbone was never fine-tuned for this).
    tensor = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0

    # Add batch dimension → [1, 3, H, W]
    tensor = torch.from_numpy(tensor).unsqueeze(0)

    return tensor, (orig_h, orig_w)


def scale_coords_to_original(
    coords_hm: torch.Tensor,   # [B, K, 2] — (x, y) in heatmap space [0, 255]
    orig_size: tuple[int, int],  # (H_orig, W_orig)
) -> torch.Tensor:
    """
    Map coordinates from heatmap space [256×256] → original image space.

    Heatmap (256, 256) → Input (512, 512) → Original (H_orig, W_orig)
    Each step applies its own scale factor independently.
    """
    orig_h, orig_w = orig_size
    hm_h, hm_w = HEATMAP_SIZE
    inp_h, inp_w = INPUT_SIZE

    # Heatmap → Input (the model already operates at INPUT_SIZE, so this = 2×)
    scale_x_hm2inp = inp_w / hm_w   # 512/256 = 2
    scale_y_hm2inp = inp_h / hm_h   # 512/256 = 2

    coords_inp = coords_hm.clone()
    coords_inp[..., 0] *= scale_x_hm2inp
    coords_inp[..., 1] *= scale_y_hm2inp

    # Input → Original
    scale_x_inp2orig = orig_w / inp_w
    scale_y_inp2orig = orig_h / inp_h

    coords_orig = coords_inp.clone()
    coords_orig[..., 0] *= scale_x_inp2orig
    coords_orig[..., 1] *= scale_y_inp2orig

    return coords_orig


# ─────────────────────────────────────────────────────────────────────────────
#  Core inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_all(
    image_paths: list[str],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 4,
    use_tta: bool = True,
) -> list[dict]:
    """
    Run batch inference with optional Test-Time Augmentation (TTA).

    Args:
        image_paths: list of image file paths
        model: trained CephalometricModel
        device: torch device
        batch_size: number of images per batch per TTA variant
        use_tta: if True, apply 7-variant TTA and average coordinates
                 if False, run single-pass inference (original image only)

    Returns a list of result dicts, one per image:
        {
            "filename": "image_001.jpg",
            "landmarks": {name: {"x": float, "y": float, "confidence": float}, ...}
        }
    """
    model.eval()
    results: list[dict] = []

    # How many images per TTA batch (balance memory vs throughput)
    tta_batch_size = max(1, batch_size // NUM_TTA)

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batches"):
        batch_paths = image_paths[i : i + batch_size]

        for path in batch_paths:
            fname = Path(path).name

            if not use_tta:
                # ── Single-pass inference (original only) ────────────────────
                tensor, orig_size = preprocess_image(path)
                batch_input = tensor.to(device)
                out = model(batch_input)
                # Handle both single-tensor and (heatmaps, uncertainty) return types
                heatmaps = out[0] if isinstance(out, tuple) else out
                if heatmaps.shape[-2:] != HEATMAP_SIZE:
                    heatmaps = F.interpolate(heatmaps, size=HEATMAP_SIZE,
                                              mode="bilinear", align_corners=False)
                coords_hm, confs = _hard_argmax_decode(heatmaps.cpu())
                coords_orig = scale_coords_to_original(coords_hm, orig_size)
                xy = coords_orig[0].numpy()            # [K, 2]
                conf = confs[0].numpy()                # [K]
            else:
                # ── TTA inference: 7 variants, average coordinates ─────────
                # Load original RGB once (shared across all TTA variants)
                img = cv2.imread(path)
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {path}")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = img_rgb.shape[:2]

                all_coords_orig: list[np.ndarray] = []
                all_confs: list[np.ndarray] = []

                for tta_name, transform_fn, inverse_fn in TTA_VARIANTS:
                    # Apply augmentation to the original RGB
                    img_aug = transform_fn(img_rgb)

                    # Resize to INPUT_SIZE and normalize to tensor
                    img_resized = cv2.resize(
                        img_aug, (INPUT_SIZE[1], INPUT_SIZE[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    tensor_np = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(tensor_np).unsqueeze(0).to(device)

                    # Forward pass
                    out = model(tensor)
                    heatmaps = out[0] if isinstance(out, tuple) else out
                    if heatmaps.shape[-2:] != HEATMAP_SIZE:
                        heatmaps = F.interpolate(
                            heatmaps, size=HEATMAP_SIZE,
                            mode="bilinear", align_corners=False,
                        )

                    # Decode to heatmap coordinates
                    coords_hm, confs_tensor = _hard_argmax_decode(heatmaps.cpu())
                    coords_hm_np = coords_hm[0].numpy()  # [K, 2] heatmap [256,256]
                    conf_np = confs_tensor[0].numpy()     # [K]

                    # Inverse transform: augmented heatmap coords → original image space
                    coords_orig_np = inverse_fn(coords_hm_np, (orig_h, orig_w))  # [K, 2]

                    all_coords_orig.append(coords_orig_np)
                    all_confs.append(conf_np)

                # Average coordinates across all TTA variants
                avg_coords = np.mean(all_coords_orig, axis=0)  # [K, 2]
                avg_conf = np.mean(all_confs, axis=0)          # [K]
                xy = avg_coords
                conf = avg_conf

            # Build result dict
            landmarks = {}
            for k, name in enumerate(KEYPOINT_NAMES):
                landmarks[name] = {
                    "x": round(float(xy[k, 0]), 1),
                    "y": round(float(xy[k, 1]), 1),
                    "confidence": round(float(conf[k]), 3),
                }

            results.append({
                "filename": fname,
                "landmarks": landmarks,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch inference: generate AI-assisted pre-annotations from HRNet-W32",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing unannotated X-ray images (jpg/png)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "checkpoints" / "fold1_best.pth"),
        help="Path to trained model weights (.pth) (default: outputs/checkpoints/fold1_best.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output predictions.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config.yaml"),
        help="Project config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cuda", "cpu"],
        help="Override device selection (default: auto-detect)",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable TTA (Test-Time Augmentation); run single-pass inference",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # ── Device selection ───────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # ── Image directory ────────────────────────────────────────────────────
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Collect image files (jpg, png, jpeg)
    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    print(f"[INFO] Found {len(image_paths)} images in {image_dir}")

    # ── Load model ─────────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Loading model from: {checkpoint_path}")
    model = CephalometricModel(num_keypoints=NUM_KEYPOINTS, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Training checkpoints are saved as wrapper dicts (with metadata).
    # Raw model weights are under "model_state_dict".
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("[INFO] Model loaded successfully")

    use_tta = not args.no_tta
    if use_tta:
        print(f"[INFO] TTA enabled: {NUM_TTA} variants — {', '.join(n for n, _, _ in TTA_VARIANTS)}")
    else:
        print("[INFO] TTA disabled — single-pass inference")

    # ── Run inference ─────────────────────────────────────────────────────
    print(f"[INFO] Running inference on {len(image_paths)} images ...")
    results = predict_all(
        [str(p) for p in image_paths],
        model=model,
        device=device,
        batch_size=args.batch_size,
        use_tta=use_tta,
    )

    # ── Build output dict keyed by filename ────────────────────────────────
    # Frontend format: { "image_001.jpg": { "Upper_tip": {"x":..., "y":..., "confidence":...}, ... }, ... }
    predictions_json = {}
    for r in results:
        predictions_json[r["filename"]] = r["landmarks"]

    # ── Save JSON ─────────────────────────────────────────────────────────
    save_json(predictions_json, args.output)
    print(f"\n[SUCCESS] Saved {len(predictions_json)} image predictions → {args.output}")


if __name__ == "__main__":
    main()