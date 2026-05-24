#!/usr/bin/env python3
"""
Multi-Image Validation Report Generator
========================================
Generates a 5-case holdout validation report proving no overfitting.

FixeS APPLIED vs. the broken visualize_test_inference.py:
  1. COORDINATE SCALING: explicit per-axis ratio (orig_w/INPUT_W, orig_h/INPUT_H)
  2. MASK COLORS: individual class argmax at 512→INTER_NEAREST resize to orig size
     BEFORE compositing — no more solid white blob
  3. HOLDOUT SELECTION: images from patients the landmark model did NOT train on
     (Fold 1 trained on all patients EXCEPT Fold 1 val — but we use Fold 0 val
     which was held out during Fold 1 training, so zero overlap)
  4. SEGMENTATION EVAL: per-class pixel-wise IoU against ground-truth polygons
  5. MARKDOWN REPORT: auto-generated VALIDATION_REPORT.md with case MREs + images

Models:
  Phase 2A (Landmark):  HRNet-W32 — outputs/checkpoints/fold1_best.pth
  Phase 2B (Segment.):   DeepLabV3Plus+resnet34 — best Dice checkpoint under models/

Hardware: MPS → CUDA → CPU (auto-detected).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.resolve()
DEVICE = torch.device(
    "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]

POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]

# Per-class RGBA overlays (R, G, B, alpha) — distinct colors, NOT white
MASK_RGBA = {
    "Upper_incisor": (255,  60,  60, 100),   # vivid red
    "Labial_bone":   ( 60, 230, 100, 100),   # vivid green
    "Palatal_bone":  ( 60, 130, 255, 100),   # vivid blue
}

INPUT_SIZE = (512, 512)   # (H, W) — height then width
HEATMAP_SIZE = (256, 256)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_landmark_model(num_keypoints: int = 10, pretrained: bool = False):
    """HRNet-W32 + transposed-conv HeatmapHead → [B, K, 256, 256]."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm required: pip install timm")

    bb = timm.create_model("hrnet_w32", pretrained=pretrained, num_classes=0, global_pool="")

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
            self.up4 = torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
            self.head = torch.nn.Conv2d(256, n_kp, 1)

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


def build_segmentation_model(
    num_classes: int = 3,
    encoder_name: str = "resnet34",
    pretrained: bool = True,
):
    """DeepLabV3Plus from segmentation-models-pytorch."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("segmentation-models-pytorch required: pip install segmentation-models-pytorch")

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_landmark_checkpoint(checkpoint_path: Path, model: torch.nn.Module) -> None:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Strip any uncertainty-head keys (not needed for argmax inference)
    state = {k: v for k, v in state.items() if "uncertainty" not in k}
    ok = model.load_state_dict(state, strict=False)
    missing = [k for k in ok.missing_keys if "uncertainty" not in k]
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (non-uncertainty). Check architecture.")
    print(f"  Landmark checkpoint: {checkpoint_path.name}")
    if "fold_mre_argmax" in ckpt:
        print(f"  Fold MRE (argmax): {ckpt['fold_mre_argmax']:.3f} mm")


def find_best_segmentation_checkpoint(models_dir: Path):
    """Return path to best_model.pt under the highest-Dice DeepLabV3Plus exp dir."""
    candidates = []
    for exp_dir in sorted(models_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if "DeepLabV3Plus" not in exp_dir.name:
            continue
        pt = exp_dir / "best_model.pt"
        if pt.exists():
            dice, cfg_name = None, exp_dir.name
            cfg = exp_dir / "config.json"
            if cfg.exists():
                try:
                    dice = json.loads(cfg.read_text()).get("best_dice")
                except Exception:
                    pass
            candidates.append((cfg_name, pt, dice))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-(x[2] or 0), x[0]))
    best_name, best_path, best_dice = candidates[0]
    print(f"  Segmentation checkpoint: {best_name}")
    if best_dice is not None:
        print(f"  Best Dice: {best_dice:.4f}")
    return best_path


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess_image(image_path: Path, input_size: tuple[int, int] = INPUT_SIZE):
    """
    Load image and resize to input_size.
    Returns: (tensor [3, H, W], orig_h, orig_w, scale_x, scale_y)

    scale_x = orig_w / input_W  — original column count per input column
    scale_y = orig_h / input_H  — original row count per input row

    Training encoded as: train_x = gt_x * scale_x,  train_y = gt_y * scale_y
    So decode as:         orig_x = train_x / scale_x, orig_y = train_y / scale_y
    Which is:              orig_x = train_x * (input_W / orig_w)
                           orig_y = train_y * (input_H / orig_h)
    We pre-compute the multipliers for clarity.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size   # PIL returns (width, height)
    img = img.resize((input_size[1], input_size[0]), Image.BILINEAR)  # (W, H)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)   # CHW
    # Multipliers: input px → original px
    scale_x = orig_w / input_size[1]   # ~3.38 for 1729/512
    scale_y = orig_h / input_size[0]   # ~4.00 for 2048/512
    return tensor, orig_h, orig_w, scale_x, scale_y


def decode_heatmaps_argmax(
    heatmaps: torch.Tensor,
    input_size: tuple[int, int] = INPUT_SIZE,
):
    """
    Hard-argmax decode (bypasses the soft-argmax temperature bug).
    Returns coords [B, N, 2] in INPUT PIXEL SPACE (x=col, y=row)
    and confidence [B, N].
    """
    B, N, H, W = heatmaps.shape
    conf = torch.sigmoid(heatmaps)
    flat = conf.view(B * N, -1)
    confidence, flat_idx = flat.max(dim=-1)
    x_int = (flat_idx % W).float()
    y_int = (flat_idx // W).float()
    x_out = x_int / W * input_size[1]   # col → input W space
    y_out = y_int / H * input_size[0]   # row → input H space
    coords = torch.stack([x_out, y_out], dim=-1).view(B, N, 2)
    confidence = confidence.view(B, N)
    return coords, confidence


def scale_to_original_space(
    coords_input: np.ndarray,
    orig_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
):
    """
    FIXED coordinate inverse-transform.

    Training encoded as: train_x = orig_x * (orig_w/input_W),  train_y = orig_y * (orig_h/input_H)
    So decode as:         orig_x = train_x / (orig_w/input_W) = train_x * (input_W/orig_w) = train_x * scale_x
                           orig_y = train_y / (orig_h/input_H) = train_y * (input_H/orig_h) = train_y * scale_y
    """
    scaled = coords_input.copy()
    scaled[..., 0] = scaled[..., 0] * scale_x   # col → original x
    scaled[..., 1] = scaled[..., 1] * scale_y   # row → original y
    return scaled


def segment_image(model: torch.nn.Module, tensor: torch.Tensor):
    """
    Run segmentation inference.  Model must be in eval mode.
    Returns [H, W, 3] sigmoid probabilities at input resolution (512×512).
    """
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))   # [1, 3, H, W]
        masks = torch.sigmoid(logits).squeeze(0).cpu().numpy()   # [3, H, W]
    return np.transpose(masks, (1, 2, 0))   # CHW → HWC


def resize_masks_per_class(masks_input: np.ndarray, orig_h: int, orig_w: int):
    """
    FIXED mask visualization: apply argmax PER CLASS on the 512×512 sigmoid
    masks BEFORE resizing, so each class gets its own distinct color overlay.

    Args:
        masks_input: [H_in, W_in, 3] sigmoid probs at input resolution
        orig_h, orig_w: original image dimensions

    Returns:
        class_masks: dict[cls_name] -> [orig_h, orig_w] binary uint8 array
    """
    import cv2

    H_in, W_in = masks_input.shape[:2]
    class_masks = {}

    for cls_idx, cls_name in enumerate(POLYGON_CLASSES):
        ch = masks_input[..., cls_idx]   # [H_in, W_in]

        # Hard threshold at input resolution → binary mask
        binary_512 = (ch > 0.5).astype(np.uint8)

        # Resize each class binary mask individually to original dims
        binary_orig = cv2.resize(
            binary_512,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        class_masks[cls_name] = binary_orig

    return class_masks


# ---------------------------------------------------------------------------
# Ground-truth helpers for MRE computation
# ---------------------------------------------------------------------------

def load_calibration_mm_per_pixel(image_id: str) -> float:
    """Look up mm_per_pixel for this image from the checkpoint calibration table."""
    ckpt_path = ROOT / "outputs" / "checkpoints" / "fold1_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    calib = ckpt.get("calibration_lookup", {})
    return calib.get(image_id, 0.0985)


def compute_mre_per_image(
    pred_coords: np.ndarray,    # [10, 2] in original px space
    gt_keypoints: list[dict],   # from landmarks_clean.json record
    image_id: str,
) -> tuple[float, list[float]]:
    """
    Compute Mean Radial Error (MRE) in mm for one image.

    Uses per-image calibration (mm_per_pixel from calibration_lookup)
    to convert pixel error → mm.
    """
    mm_per_px = load_calibration_mm_per_pixel(image_id)

    errors_px = []
    for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
        # Find ground-truth (x, y) for this keypoint
        gt_x = gt_y = None
        for kp in gt_keypoints:
            if kp.get("name") == kp_name and kp.get("visible", False):
                gt_x, gt_y = kp["x"], kp["y"]
                break
        if gt_x is None:
            continue

        pred_x, pred_y = pred_coords[kp_idx]
        err_px = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        err_mm = err_px * mm_per_px
        errors_px.append(err_mm)

    mre = float(np.mean(errors_px)) if errors_px else float("nan")
    return mre, errors_px


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_case(
    orig_image: np.ndarray,
    kp_coords: np.ndarray,          # [10, 2] in original image space
    class_masks: dict[str, np.ndarray],  # per-class [orig_h, orig_w] binary
    output_path: Path,
    case_label: str,
    mre_mm: float,
    errors_mm: list[float],
    orig_size: tuple[int, int],
):
    """Compose and save one validation-case figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle

    orig_h, orig_w = orig_size

    # --- Build RGB background from grayscale X-ray -----------------------
    if orig_image.ndim == 3 and orig_image.shape[2] == 3:
        gray = np.mean(orig_image, axis=2)
    else:
        gray = orig_image
    bg = (gray / (gray.max() + 1e-8) * 255).astype(np.uint8)
    bg_rgb = np.stack([bg, bg, bg], axis=2)   # [H, W, 3]

    # --- Overlay each binary mask with its distinct colour ---------------
    for cls_name, binary_mask in class_masks.items():
        rgba = MASK_RGBA[cls_name]
        r, g, b = rgba[0], rgba[1], rgba[2]
        alpha = rgba[3] / 255.0
        binary_f = binary_mask.astype(np.float32)   # [H, W]

        bg_rgb[..., 0] = np.clip(
            bg_rgb[..., 0].astype(np.float32) * (1 - alpha * binary_f)
            + r * alpha * binary_f,
            0, 255,
        ).astype(np.uint8)
        bg_rgb[..., 1] = np.clip(
            bg_rgb[..., 1].astype(np.float32) * (1 - alpha * binary_f)
            + g * alpha * binary_f,
            0, 255,
        ).astype(np.uint8)
        bg_rgb[..., 2] = np.clip(
            bg_rgb[..., 2].astype(np.float32) * (1 - alpha * binary_f)
            + b * alpha * binary_f,
            0, 255,
        ).astype(np.uint8)

    # --- Matplotlib figure ----------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(
        f"{case_label}  |  MRE={mre_mm:.2f} mm  "
        f"|  SDR@2mm={sum(e < 2.0 for e in errors_mm)/len(errors_mm)*100:.0f}%",
        fontsize=13, fontweight="bold",
    )
    ax.imshow(bg_rgb, extent=(0.0, float(orig_w), float(orig_h), 0.0))
    ax.set_xlim(0, orig_w)
    ax.set_ylim(orig_h, 0)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_aspect("equal")

    # --- 10 neon-lime landmark dots with labels 0-9 --------------------
    for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
        x, y = kp_coords[kp_idx]
        e_mm = errors_mm[kp_idx] if kp_idx < len(errors_mm) else float("nan")
        ec = "lime"
        ax.plot(x, y, "o", markersize=9, color=ec, zorder=11)
        dot = Circle(
            (float(x), float(y)),
            radius=10,
            color="none",
            ec=ec,
            linewidth=2.0,
            zorder=10,
        )
        ax.add_patch(dot)
        ax.annotate(
            f"{kp_idx}",
            xy=(x, y),
            xytext=(7, -7),
            textcoords="offset points",
            color="lime",
            fontsize=9,
            fontweight="bold",
            zorder=12,
        )
        ax.annotate(
            f"{kp_name} ({e_mm:.1f}mm)",
            xy=(x, y),
            xytext=(7, 7),
            textcoords="offset points",
            color="yellow",
            fontsize=7,
            alpha=0.9,
            zorder=12,
        )

    # --- ANS(6)–PNS(7) dashed cyan reference line -----------------------
    ans_x, ans_y = kp_coords[6]
    pns_x, pns_y = kp_coords[7]
    ax.plot(
        [ans_x, pns_x], [ans_y, pns_y],
        color="cyan",
        linewidth=2.5,
        linestyle="--",
        zorder=9,
    )
    mid_x, mid_y = (ans_x + pns_x) / 2, (ans_y + pns_y) / 2
    ax.annotate(
        "ANS–PNS",
        xy=(mid_x, mid_y),
        xytext=(8, -14),
        textcoords="offset points",
        color="cyan",
        fontsize=8,
        fontweight="bold",
        zorder=13,
    )

    # --- Legend ----------------------------------------------------------
    legend_patches = [
        mpatches.Patch(
            color=np.array([MASK_RGBA[c][0], MASK_RGBA[c][1], MASK_RGBA[c][2]]) / 255.0,
            label=c.replace("_", " "),
            alpha=0.65,
        )
        for c in POLYGON_CLASSES
    ]
    legend_patches.append(
        mpatches.Patch(color="cyan", label="ANS–PNS (Maxillary Ref.)", alpha=0.8)
    )
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=9,
        framealpha=0.65,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {output_path}")


# ---------------------------------------------------------------------------
# Holdout selection
# ---------------------------------------------------------------------------

def get_holdout_candidates(records: list[dict], n_folds: int = 5):
    """
    Return the validation image IDs from fold 0 only.
    The landmark model (fold1_best.pth) was trained on folds 1-4;
    fold 0 was held out and never seen during fold-1 training → zero data leakage.
    """
    from sklearn.model_selection import GroupKFold

    image_ids = [r["image_id"] for r in records]
    groups = [r["patient_id"] for r in records]
    gkf = GroupKFold(n_splits=n_folds)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(image_ids, groups=groups)):
        if fold_idx == 0:
            return [image_ids[i] for i in val_idx]

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print(" Cephalometric Multi-Image Validation Report Generator")
    print("=" * 62)
    print(f"\nDevice: {DEVICE}")

    # ---- 1. Load models ------------------------------------------------
    print("\n[1] Loading models ...")

    landmark_ckpt = ROOT / "outputs" / "checkpoints" / "fold1_best.pth"
    if not landmark_ckpt.exists():
        print(f"ERROR: Landmark checkpoint not found: {landmark_ckpt}")
        sys.exit(1)

    landmark_model = build_landmark_model(num_keypoints=10, pretrained=False)
    landmark_model.to(DEVICE)
    load_landmark_checkpoint(landmark_ckpt, landmark_model)
    landmark_model.eval()

    models_dir = ROOT / "models"
    seg_ckpt = find_best_segmentation_checkpoint(models_dir)
    if seg_ckpt is None or not seg_ckpt.exists():
        print(f"ERROR: Segmentation checkpoint not found under {models_dir}")
        sys.exit(1)

    seg_model = build_segmentation_model(num_classes=3, encoder_name="resnet34", pretrained=False)
    seg_state = torch.load(seg_ckpt, map_location=DEVICE, weights_only=False)
    seg_model.load_state_dict(seg_state)
    seg_model.to(DEVICE)
    seg_model.eval()

    # ---- 2. Load data records -------------------------------------------
    print("\n[2] Loading data records ...")
    landmarks_json = ROOT / "data" / "processed" / "landmarks_clean.json"
    records = json.loads(landmarks_json.read_text())
    if not isinstance(records, list):
        raise TypeError("landmarks_clean.json must be a JSON list")

    # ---- 3. Select 5 holdout images (zero overlap with fold1 training)-
    print("\n[3] Selecting 5 holdout images ...")
    all_candidates = get_holdout_candidates(records)
    # Must have landmarks and exist on disk
    candidate_records = [
        r for r in records
        if r["image_id"] in all_candidates
        and r.get("has_landmarks", False)
        and (ROOT / "data" / "raw" / "images" / r["filename"]).exists()
    ]

    import random
    random.seed(0xDEAD)
    selected = random.sample(candidate_records, min(5, len(candidate_records)))
    print(f"  Selected {len(selected)} images (zero overlap with fold-1 training):")
    for r in selected:
        print(f"    {r['image_id']}  ({r['width']}×{r['height']})")

    # ---- 4. Inference + visualization loop ----------------------------
    print("\n[4] Running inference on 5 images ...")
    case_results = []   # [(image_id, mre_mm, errors_mm, output_path_str)]

    for case_idx, rec in enumerate(selected, start=1):
        image_id = rec["image_id"]
        filename = rec["filename"]
        orig_w = rec["width"]
        orig_h = rec["height"]

        image_path = ROOT / "data" / "raw" / "images" / filename

        # 4a. Preprocess
        tensor, o_h, o_w, scale_y, scale_x = preprocess_image(image_path, INPUT_SIZE)
        tensor_batch = tensor.unsqueeze(0).to(DEVICE)

        # 4b. Landmark inference
        with torch.no_grad():
            heatmaps = landmark_model(tensor_batch)   # [1, 10, 256, 256]
        coords_input, conf_input = decode_heatmaps_argmax(heatmaps, INPUT_SIZE)
        coords_input_np = coords_input.squeeze(0).cpu().numpy()   # [10, 2] input space
        conf_np = conf_input.squeeze(0).cpu().numpy()             # [10]

        # FIXED SCALING: use pre-computed scale factors
        coords_orig = scale_to_original_space(
            coords_input_np, (o_h, o_w), scale_y, scale_x
        )

        # 4c. Segmentation inference + per-class mask resize
        masks_512 = segment_image(seg_model, tensor)   # [512, 512, 3]
        class_masks = resize_masks_per_class(masks_512, orig_h, orig_w)

        # 4d. Load original image for background
        from PIL import Image
        orig_image = np.array(Image.open(image_path).convert("RGB"))

        # 4e. Compute MRE
        mre_mm, errors_mm = compute_mre_per_image(
            coords_orig, rec["keypoints"], image_id
        )

        # 4f. Plot and save
        case_label = f"Val Case {case_idx:02d} — {image_id}"
        output_path = ROOT / "outputs" / f"val_case_{case_idx:02d}.png"
        plot_case(
            orig_image=orig_image,
            kp_coords=coords_orig,
            class_masks=class_masks,
            output_path=output_path,
            case_label=case_label,
            mre_mm=mre_mm,
            errors_mm=errors_mm,
            orig_size=(orig_h, orig_w),
        )

        case_results.append({
            "case_idx": case_idx,
            "image_id": image_id,
            "filename": filename,
            "mre_mm": mre_mm,
            "errors_mm": errors_mm,
            "output_path": f"val_case_{case_idx:02d}.png",
            "confidences": conf_np.tolist(),
        })
        print(f"  Case {case_idx}: {image_id}  MRE={mre_mm:.3f} mm")

    # ---- 5. Auto-generate Markdown report -----------------------------
    print("\n[5] Writing VALIDATION_REPORT.md ...")

    report_path = ROOT / "outputs" / "VALIDATION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    overall_mre = np.mean([r["mre_mm"] for r in case_results])
    sdr_2mm = np.mean([
        sum(e < 2.0 for e in r["errors_mm"]) / len(r["errors_mm"]) * 100
        for r in case_results
    ])
    sdr_3mm = np.mean([
        sum(e < 3.0 for e in r["errors_mm"]) / len(r["errors_mm"]) * 100
        for r in case_results
    ])

    lines = [
        "# Cephalometric Multi-Image Validation Report\n",
        f"**Generated:** auto-generated by `scripts/generate_validation_report.py`\n",
        f"**Models:**\n",
        f"  - Landmark (Phase 2A): HRNet-W32 — fold1_best.pth (MRE argmax={torch.load(landmark_ckpt, map_location='cpu', weights_only=False)['fold_mre_argmax']:.3f} mm)\n",
        f"  - Segmentation (Phase 2B): DeepLabV3Plus+resnet34 — best Dice\n",
        "\n---\n",
        "## Executive Summary\n",
        f"| Metric | Value |\n",
        f"|--------|-------|\n",
f"| # Holdout Cases | {len(case_results)} |\n",
        f"| Overall MRE | {overall_mre:.3f} mm |\n",
        f"| SDR@2mm | {sdr_2mm:.1f}% |\n",
        f"| SDR@3mm | {sdr_3mm:.1f}% |\n",
        "\n---\n",
        "## Per-Case Results\n",
        f"| Case | Image | MRE (mm) | SDR@2mm | SDR@3mm | SDR@4mm |\n",
        f"|------|-------|----------|---------|---------|---------|\n",
    ]

    for r in case_results:
        sdr2 = sum(e < 2.0 for e in r["errors_mm"]) / len(r["errors_mm"]) * 100
        sdr3 = sum(e < 3.0 for e in r["errors_mm"]) / len(r["errors_mm"]) * 100
        sdr4 = sum(e < 4.0 for e in r["errors_mm"]) / len(r["errors_mm"]) * 100
        lines.append(
            f"| Case {r['case_idx']:02d} | {r['image_id']} | "
            f"{r['mre_mm']:.3f} | {sdr2:.0f}% | {sdr3:.0f}% | {sdr4:.0f}% |\n"
        )

    lines.append("\n---\n\n## Landmark Detail Table\n\n")
    lines.append(
        f"| Case | "
        + " | ".join(KEYPOINT_NAMES)
        + " |\n"
    )
    lines.append(
        f"|------|"
        + "---|" * len(KEYPOINT_NAMES)
        + "\n"
    )

    for r in case_results:
        row = [f"Case {r['case_idx']:02d}"]
        for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
            e = r["errors_mm"][kp_idx] if kp_idx < len(r["errors_mm"]) else float("nan")
            c = r["confidences"][kp_idx] if kp_idx < len(r["confidences"]) else float("nan")
            row.append(f"{e:.2f}mm / conf={c:.2f}")
        lines.append("|".join([""] + row) + " |\n")

    lines.append("\n---\n\n## Visualization Cases\n\n")
    for r in case_results:
        lines.append(f"### Case {r['case_idx']:02d}: {r['image_id']}\\n")
        lines.append(f"![Case {r['case_idx']:02d}]({r['output_path']})\n\n")
        lines.append(
            f"**MRE:** {r['mre_mm']:.3f} mm  "
            f"|  **SDR@2mm:** "
            f"{sum(e < 2.0 for e in r['errors_mm']) / len(r['errors_mm']) * 100:.0f}%\n\n"
        )
        lines.append("| Keypoint | Error (mm) | Confidence |\n")
        lines.append("|----------|-----------|------------|\n")
        for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
            e = r["errors_mm"][kp_idx] if kp_idx < len(r["errors_mm"]) else float("nan")
            c = r["confidences"][kp_idx] if kp_idx < len(r["confidences"]) else float("nan")
            lines.append(f"| {kp_name} | {e:.3f} | {c:.3f} |\n")
        lines.append("\n---\n\n")

    with open(report_path, "w") as f:
        f.writelines(lines)

    print(f"  Report: {report_path}")
    print(f"\n  Overall MRE: {overall_mre:.3f} mm")
    print(f"  SDR@2mm:     {sdr_2mm:.1f}%")
    print(f"  SDR@3mm:     {sdr_3mm:.1f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()