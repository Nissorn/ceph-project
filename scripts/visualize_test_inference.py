#!/usr/bin/env python3
"""
Visualize landmark + segmentation inference on a holdout test image.

Generates outputs/test_inference_result.png:
  - Raw X-ray as background
  - 3 bone masks as semi-transparent colored overlays
  - 10 landmark keypoints as neon lime dots with labels 0-9
  - ANS–PNS reference line (maxillary superimposition plane)

Models loaded:
  Phase 2A (Landmark):  HRNet-W32  — outputs/checkpoints/fold1_best.pth
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
    "Upper_tip",
    "Upper_apex",
    "Labial_midroot",
    "Labial_crest",
    "Palatal_midroot",
    "Palatal_crest",
    "ANS",
    "PNS",
    "LB",
    "PB",
]

POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]
MASK_COLORS = {
    "Upper_incisor": (255, 80, 80, 102),   # red
    "Labial_bone":   (80, 220, 120, 102),   # green
    "Palatal_bone":  (80, 120, 255, 102),   # blue
}

INPUT_SIZE = (512, 512)
HEATMAP_SIZE = (256, 256)

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_landmark_model(num_keypoints: int = 10, pretrained: bool = False) -> torch.nn.Module:
    """HRNet-W32 + transposed-conv HeatmapHead → [B, K, 256, 256] heatmaps.

    NOTE: the checkpoint contains CBAM + EUPE modules added during the session's
    Phase 3 experiments. We load with strict=False so only the essential
    backbone/head weights need to match; any extra modules in the checkpoint
    are silently ignored.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm required: pip install timm")

    backbone = timm.create_model(
        "hrnet_w32",
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
    )

    class HeatmapHead(torch.nn.Module):
        """Standard transposed-conv head (no CBAM/EUPE in the forward path)."""

        def __init__(self, in_channels: int = 2048, num_kp: int = 10):
            super().__init__()
            self.reduce = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
            )
            self.up1 = torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
            self.up2 = torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
            self.up3 = torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
            self.up4 = torch.nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
            self.head = torch.nn.Conv2d(256, num_kp, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.reduce(x)
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
            return self.head(x)

    class CephalometricModel(torch.nn.Module):
        def __init__(self, num_kp: int = 10):
            super().__init__()
            self.backbone = backbone
            self.head = HeatmapHead(2048, num_kp)
            self.num_keypoints = num_kp

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone(x)
            return self.head(features)

    return CephalometricModel(num_keypoints)


def build_segmentation_model(
    num_classes: int = 3,
    encoder_name: str = "resnet34",
    pretrained: bool = True,
) -> torch.nn.Module:
    """DeepLabV3Plus from segmentation-models-pytorch."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation-models-pytorch required: pip install segmentation-models-pytorch"
        )

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    return model


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def load_landmark_checkpoint(checkpoint_path: Path, model: torch.nn.Module) -> None:
    """Load wrapper-dict checkpoint (foldN_best.pth from train.py)."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Strip EUPE uncertainty head keys — not needed for argmax inference
    state = {k: v for k, v in state.items() if "uncertainty" not in k}
    model.load_state_dict(state, strict=False)
    print(f"  Landmark checkpoint loaded: {checkpoint_path}")
    if "fold_mre_argmax" in ckpt:
        print(f"  Fold MRE (argmax): {ckpt['fold_mre_argmax']:.3f} mm")


def find_best_segmentation_checkpoint(models_dir: Path) -> Path | None:
    """Return path to best_model.pt under the highest-dice DeepLabV3Plus exp dir."""
    candidates = []
    for exp_dir in sorted(models_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if "DeepLabV3Plus" not in exp_dir.name:
            continue
        pt = exp_dir / "best_model.pt"
        if pt.exists():
            dice = None
            cfg = exp_dir / "config.json"
            if cfg.exists():
                try:
                    dice = json.loads(cfg.read_text()).get("best_dice")
                except Exception:
                    pass
            candidates.append((exp_dir.name, pt, dice))
    if not candidates:
        return None
    # Sort by dice desc, name asc
    candidates.sort(key=lambda x: (-(x[2] or 0), x[0]))
    best_name, best_path, best_dice = candidates[0]
    print(f"  Segmentation checkpoint: {best_name}")
    if best_dice is not None:
        print(f"  Best Dice: {best_dice:.4f}")
    return best_path


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def preprocess_image(image_path: Path, input_size: tuple[int, int] = INPUT_SIZE) -> torch.Tensor:
    """Load image, resize to input_size, normalize with /255.0, convert to CHW float tensor."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize(input_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # HWC → CHW
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
    return tensor, orig_h, orig_w


def decode_heatmaps_argmax(
    heatmaps: torch.Tensor,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hard-argmax decode (NOT soft-argmax — the latter has a temperature bug).
    Returns coords [B, N, 2] in input pixel space (x, y) and confidence [B, N].
    """
    B, N, H, W = heatmaps.shape
    # Sigmoid for normalized confidence scores
    conf = torch.sigmoid(heatmaps)
    flat = conf.view(B * N, -1)
    confidence, flat_idx = flat.max(dim=-1)
    # Integer coordinates in heatmap space
    x_int = (flat_idx % W).float()
    y_int = (flat_idx // W).float()
    # Scale heatmap space → input space
    x_out = x_int / W * input_size[1]
    y_out = y_int / H * input_size[0]
    coords = torch.stack([x_out, y_out], dim=-1).view(B, N, 2)
    confidence = confidence.view(B, N)
    return coords, confidence


def scale_to_original(
    coords_input: np.ndarray,
    orig_size: tuple[int, int],
    input_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Map coordinates from input space (512×512) to original image space."""
    ox, oy = orig_size
    ix, iy = input_size
    scaled = coords_input.copy()
    scaled[..., 0] = scaled[..., 0] / ix * ox
    scaled[..., 1] = scaled[..., 1] / iy * oy
    return scaled


def segment_image(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Run segmentation inference.
    Returns mask array [H, W, 3] with sigmoid probabilities per class.
    """
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(DEVICE))  # [1, 3, H, W]
        masks = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [3, H, W]
    # Rearrange to HWC for visualization
    return np.transpose(masks, (1, 2, 0))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def build_overlay(
    orig_image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Composite one bone-mask channel onto the grayscale X-ray.
    mask: [H, W, 3] sigmoid probabilities — takes channel 0 (Upper_incisor)
    Returns RGBA uint8 image suitable for plt.imshow.
    """
    # Convert orig to grayscale and broadcast to 3 channels
    if orig_image.ndim == 3:
        gray = np.mean(orig_image, axis=2).astype(np.uint8)
    else:
        gray = orig_image.astype(np.uint8)
    bg = np.stack([gray, gray, gray], axis=2)  # [H, W, 3] RGB

    # RGBA canvas
    rgba = np.zeros(list(bg.shape[:2]) + [4], dtype=np.uint8)
    rgba[..., :3] = bg

    # Overlay each class channel with its colour
    for cls_idx, cls_name in enumerate(POLYGON_CLASSES):
        ch = mask[..., cls_idx]
        color = MASK_COLORS[cls_name]
        coloured = np.zeros_like(bg)
        coloured[..., 0] = color[0]
        coloured[..., 1] = color[1]
        coloured[..., 2] = color[2]
        # Blend using sigmoid probability as alpha weight
        weight = (ch > 0.5).astype(np.float32) * alpha
        for c in range(3):
            rgba[..., c] = (
                rgba[..., c].astype(np.float32) * (1 - weight)
                + coloured[..., c].astype(np.float32) * weight
            ).astype(np.uint8)
        rgba[..., 3] = (
            rgba[..., 3].astype(np.float32)
            + weight * 200 * (rgba[..., 3] == 0)
        ).astype(np.uint8)

    return rgba


def plot_result(
    orig_image: np.ndarray,
    kp_coords: np.ndarray,   # [10, 2] in original image space
    masks: np.ndarray,       # [H, W, 3] sigmoid probs
    output_path: Path,
    orig_size: tuple[int, int],
) -> None:
    """Compose and save the final matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle, FancyArrowPatch

    H, W = orig_size[1], orig_size[0]

    # Build composite image with all mask overlays
    composite = orig_image.copy().astype(np.float32)
    if composite.ndim == 3 and composite.shape[2] == 3:
        gray = np.mean(composite, axis=2)
    else:
        gray = composite

    # Start with grayscale X-ray as uint8
    bg_gray = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 0 else gray.astype(np.uint8)
    bg_rgb = np.stack([bg_gray, bg_gray, bg_gray], axis=2)

    # Blend each mask channel
    for cls_idx, cls_name in enumerate(POLYGON_CLASSES):
        ch = masks[..., cls_idx]
        color = MASK_COLORS[cls_name]
        alpha = 0.40
        for c in range(3):
            bg_rgb[..., c] = np.clip(
                bg_rgb[..., c].astype(np.float32) * (1 - alpha * (ch > 0.5))
                + color[c] * alpha * (ch > 0.5) * 255,
                0, 255
            ).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title("Cephalometric Test Inference\nLandmarks + Alveolar Bone Segmentation", fontsize=13)
    ax.imshow(bg_rgb, extent=(0.0, float(W), float(H), 0.0))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_aspect("equal")

    # Draw mask colour legend patches
    legend_patches = []
    for cls_name, color in MASK_COLORS.items():
        patch = mpatches.Patch(
            color=np.array(color[:3]) / 255.0,
            label=cls_name.replace("_", " "),
            alpha=0.6,
        )
        legend_patches.append(patch)

    # Plot 10 landmark keypoints
    for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
        x, y = kp_coords[kp_idx]
        # Neon lime dot
        dot = Circle(
            (float(x), float(y)),
            radius=12,
            color="none",
            ec="lime",
            linewidth=2.0,
            zorder=10,
        )
        ax.add_patch(dot)
        ax.plot(x, y, "o", markersize=8, color="lime", zorder=11)
        # Label
        ax.annotate(
            f"{kp_idx}",
            xy=(x, y),
            xytext=(6, -6),
            textcoords="offset points",
            color="lime",
            fontsize=8,
            fontweight="bold",
            zorder=12,
        )
        ax.annotate(
            kp_name,
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            color="yellow",
            fontsize=7,
            alpha=0.85,
            zorder=12,
        )

    # ANS(6)–PNS(7) reference line (maxillary superimposition plane)
    ans_x, ans_y = kp_coords[6]
    pns_x, pns_y = kp_coords[7]
    ax.plot(
        [ans_x, pns_x], [ans_y, pns_y],
        color="cyan",
        linewidth=2.0,
        linestyle="--",
        zorder=9,
        label="ANS–PNS (Maxillary Ref.)",
    )
    ref_patch = mpatches.Patch(color="cyan", label="ANS–PNS (Maxillary Ref.)", alpha=0.7)
    legend_patches.append(ref_patch)

    # Legend
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8,
        framealpha=0.6,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualization saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print(" Cephalometric Test Inference Visualization")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")

    # ---- 1. Load models -------------------------------------------------
    print("\n[1] Loading models...")

    # Phase 2A — Landmark (HRNet-W32)
    landmark_ckpt = ROOT / "outputs" / "checkpoints" / "fold1_best.pth"
    if not landmark_ckpt.exists():
        print(f"ERROR: Landmark checkpoint not found: {landmark_ckpt}")
        sys.exit(1)
    landmark_model = build_landmark_model(num_keypoints=10, pretrained=False)
    landmark_model.to(DEVICE)
    load_landmark_checkpoint(landmark_ckpt, landmark_model)
    landmark_model.eval()

    # Phase 2B — Segmentation (DeepLabV3Plus+resnet34)
    models_dir = ROOT / "models"
    seg_ckpt = find_best_segmentation_checkpoint(models_dir)
    if seg_ckpt is None or not seg_ckpt.exists():
        print(f"ERROR: Segmentation checkpoint not found under {models_dir}")
        sys.exit(1)
    seg_model = build_segmentation_model(num_classes=3, encoder_name="resnet34", pretrained=False)
    seg_state = torch.load(seg_ckpt, map_location=DEVICE, weights_only=False)
    seg_model.load_state_dict(seg_state)
    print(f"  Segmentation checkpoint loaded: {seg_ckpt}")
    seg_model.to(DEVICE)
    seg_model.eval()

    # ---- 2. Select holdout test image ------------------------------------
    print("\n[2] Selecting holdout test image...")

    landmarks_json = ROOT / "data" / "processed" / "landmarks_clean.json"
    calib_csv = ROOT / "data" / "processed" / "calibration.csv"

    records = json.loads(landmarks_json.read_text())
    if not isinstance(records, list):
        raise TypeError("landmarks_clean.json must be a JSON list")

    # Pick first image with landmarks for demonstration
    test_record = None
    for r in records:
        if r.get("has_landmarks"):
            test_record = r
            break

    if test_record is None:
        print("ERROR: No records with landmarks found.")
        sys.exit(1)

    image_id = test_record["image_id"]
    filename = test_record["filename"]
    orig_h = test_record["height"]
    orig_w = test_record["width"]

    image_path = ROOT / "data" / "raw" / "images" / filename
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    print(f"  image_id : {image_id}")
    print(f"  filename : {filename}")
    print(f"  orig size : {orig_w} × {orig_h}")

    # ---- 3. Run inference -----------------------------------------------
    print("\n[3] Running inference...")

    tensor, _, _ = preprocess_image(image_path, INPUT_SIZE)
    tensor_batch = tensor.unsqueeze(0).to(DEVICE)

    # 3a. Landmark detection
    with torch.no_grad():
        heatmaps = landmark_model(tensor_batch)  # [1, 10, 256, 256]
    coords_input, conf_input = decode_heatmaps_argmax(heatmaps, INPUT_SIZE)
    coords_input_np = coords_input.squeeze(0).cpu().numpy()   # [10, 2] input space
    conf_np = conf_input.squeeze(0).cpu().numpy()               # [10]

    # Scale to original image space
    coords_orig = scale_to_original(coords_input_np, (orig_w, orig_h), INPUT_SIZE)

    # 3b. Segmentation
    masks = segment_image(seg_model, tensor)  # [H, W, 3]

    # Resize masks to original image size for visualization
    masks_resized = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
    for c in range(3):
        import cv2
        masks_resized[..., c] = cv2.resize(masks[..., c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Load original image for background
    from PIL import Image
    orig_image = np.array(Image.open(image_path).convert("RGB"))

    # ---- 4. Print landmark results --------------------------------------
    print("\n[4] Landmark coordinates (original image space):")
    for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
        x, y = coords_orig[kp_idx]
        c = conf_np[kp_idx]
        print(f"  [{kp_idx}] {kp_name:<20} ({x:7.1f}, {y:7.1f})  conf={c:.3f}")

    # ---- 5. Plot and save ----------------------------------------------
    print("\n[5] Generating visualization...")
    output_path = ROOT / "outputs" / "test_inference_result.png"
    plot_result(
        orig_image=orig_image,
        kp_coords=coords_orig,
        masks=masks_resized,
        output_path=output_path,
        orig_size=(orig_w, orig_h),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()