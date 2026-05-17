#!/usr/bin/env python3
"""
predict_all.py — Batch inference for AI-Assisted Pre-annotations
=================================================================

Run HRNet-W32 heatmap model over a directory of unannotated X-ray images
and emit a single predictions.json consumed directly by the Astro.js
Phase-4 frontend.

Usage
-----
    python predict_all.py \\
        --image-dir data/raw/unannotated_images \\
        --checkpoint outputs/best_model.pth \\
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
* ImageNet normalization: matches how HRNet-W32 was pretrained (timm coco).
* Batch processing: processes images in batches for GPU efficiency.
* MPS / CUDA / CPU: device is auto-detected; MPS is used on Apple Silicon.
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

# ── ImageNet normalization (matches HRNet-W32 coco pretraining) ────────────
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # kept for reference only (unused after fix)


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
        coords:    [B, K, 2] float32  — (x=col, y=row) in heatmap pixel space
        confidence: [B, K] float32    — sigmoid(peak_activation)
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
    batch_size: int = 8,
) -> list[dict]:
    """
    Run batch inference over a list of image paths.

    Returns a list of result dicts, one per image:
        {
            "filename": "image_001.jpg",
            "landmarks": {name: {"x": float, "y": float, "confidence": float}, ...}
        }
    """
    model.eval()
    results: list[dict] = []

    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batches"):
        batch_paths = image_paths[i : i + batch_size]

        # Preprocess all images in the batch
        batch_tensors: list[torch.Tensor] = []
        orig_sizes: list[tuple[int, int]] = []
        filenames: list[str] = []

        for path in batch_paths:
            tensor, orig_size = preprocess_image(path)
            batch_tensors.append(tensor)
            orig_sizes.append(orig_size)
            filenames.append(Path(path).name)

        # Stack into [B, 3, 512, 512]
        batch_input = torch.cat(batch_tensors, dim=0).to(device)

        # Forward pass → [B, K, 256, 256]
        heatmaps = model(batch_input)

        # Ensure correct heatmap size (belt-and-suspenders)
        if heatmaps.shape[-2:] != HEATMAP_SIZE:
            heatmaps = F.interpolate(
                heatmaps, size=HEATMAP_SIZE, mode="bilinear", align_corners=False
            )

        # Hard-argmax decode on CPU for downstream processing
        heatmaps_cpu = heatmaps.cpu()
        coords_hm, conf_hm = _hard_argmax_decode(heatmaps_cpu)  # [B, K, 2], [B, K]

        # Scale to original image sizes and build result dicts
        for b, (fname, orig_size) in enumerate(zip(filenames, orig_sizes)):
            coords_orig = scale_coords_to_original(coords_hm[b : b + 1], orig_size)  # [1, K, 2]
            confs = conf_hm[b].numpy()      # [K]
            xy    = coords_orig[0].numpy()   # [K, 2]

            landmarks = {}
            for k, name in enumerate(KEYPOINT_NAMES):
                landmarks[name] = {
                    "x": round(float(xy[k, 0]), 1),
                    "y": round(float(xy[k, 1]), 1),
                    "confidence": round(float(confs[k]), 3),
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
    checkpoint_path = ROOT / "outputs" / "checkpoints" / "fold1_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Loading model from: {checkpoint_path}")
    model = CephalometricModel(num_keypoints=NUM_KEYPOINTS, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    # ── Run inference ─────────────────────────────────────────────────────
    print(f"[INFO] Running inference on {len(image_paths)} images ...")
    results = predict_all(
        [str(p) for p in image_paths],
        model=model,
        device=device,
        batch_size=args.batch_size,
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
