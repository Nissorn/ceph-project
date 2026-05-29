#!/usr/bin/env python3
"""
scripts/batch_holdout_visual.py
================================
Batch holdout diagnostic visualizer.
Processes N holdout images with both landmark + segmentation models,
produces individual PNG overlays, and emits a markdown status table.

Outputs: outputs/holdout_diagnostic_01.png ... holdout_diagnostic_NN.png
"""
from __future__ import annotations

import sys, json, warnings, textwrap, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
INPUT_SIZE = (512, 512)
HEATMAP_SIZE = (256, 256)
NUM_KEYPOINTS = 10

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]
MASK_CONFIG = [
    ("Upper_incisor", (255, 80,  80),  0.4),
    ("Labial_bone",   (80,  255, 80),  0.4),
    ("Palatal_bone",  (80,  80,  255), 0.4),
]


# ── calibration ───────────────────────────────────────────────────────────────
def load_calibration(cal_csv: Path):
    import pandas as pd
    if not cal_csv.exists():
        return {}
    df = pd.read_csv(cal_csv)
    return dict(zip(df["image_id"], df["mm_per_pixel"]))


# ── model builders ─────────────────────────────────────────────────────────────
def build_landmark_model(num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = False):
    import timm

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


def build_segmentation_model(num_classes: int = 3, encoder_name: str = "resnet34", pretrained: bool = False):
    import segmentation_models_pytorch as smp
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
    )
    return model


def find_best_segmentation_checkpoint(models_dir: Path) -> Path:
    candidates = []
    if not models_dir.is_dir():
        return models_dir / "best_model.pt"
    for sub in sorted(models_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith("exp") or "DeepLabV3Plus" not in sub.name:
            continue
        bm = sub / "best_model.pt"
        if bm.exists():
            dice_str = sub.name.split("_dice")[1] if "_dice" in sub.name else "0"
            try:
                dice = float(dice_str)
            except ValueError:
                dice = 0.0
            candidates.append((dice, bm))
    if not candidates:
        return models_dir / "best_model.pt"
    candidates.sort(reverse=True)
    _, best_path = candidates[0]
    return best_path


# ── inference helpers ──────────────────────────────────────────────────────────
def preprocess_image(image_path: Path, input_size=INPUT_SIZE):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size[1], input_size[0]))
    tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0), img  # [1,3,H,W], orig RGB


def decode_coordinates_hard_argmax(heatmaps: torch.Tensor, input_size=INPUT_SIZE):
    B, N, H, W = heatmaps.shape
    heatmaps_sigmoid = torch.sigmoid(heatmaps).cpu()
    coords_list, conf_list = [], []
    for c in range(N):
        hm = heatmaps_sigmoid[0, c, :, :]
        flat_idx = hm.argmax().item()
        y_hm, x_hm = divmod(flat_idx, W)
        conf = hm[y_hm, x_hm].item()
        x_in = x_hm * (input_size[1] / W)
        y_in = y_hm * (input_size[0] / H)
        coords_list.append([x_in, y_in])
        conf_list.append(conf)
    return np.array(coords_list, dtype=np.float32), np.array(conf_list, dtype=np.float32)


def decode_coordinates_to_original(coords_input, orig_size, input_size=INPUT_SIZE):
    orig_h, orig_w = orig_size
    scale_x = orig_w / input_size[1]
    scale_y = orig_h / input_size[0]
    coords_orig = coords_input.copy()
    coords_orig[:, 0] = coords_input[:, 0] * scale_x
    coords_orig[:, 1] = coords_input[:, 1] * scale_y
    return coords_orig


def run_landmark_inference(model, image_tensor, heatmap_size=HEATMAP_SIZE, input_size=INPUT_SIZE):
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor.to(DEVICE))
        if pred.shape[-2:] != heatmap_size:
            pred = F.interpolate(pred, size=heatmap_size, mode="bilinear", align_corners=False)
    return decode_coordinates_hard_argmax(pred.cpu(), input_size)


def run_segmentation_inference(model, image_tensor, orig_size):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(DEVICE))
        sigmoid = torch.sigmoid(logits).cpu()[0]
    masks = []
    for ch in range(3):
        mask_512 = (sigmoid[ch].numpy() > 0.5).astype(np.uint8)
        mask_orig = cv2.resize(mask_512, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_orig)
    return masks


# ── overlay rendering ──────────────────────────────────────────────────────────
def blend_masks_cv2(background, masks, colors, alpha=0.4):
    result = background.copy().astype(np.float32)
    for mask, color in zip(masks, colors):
        b, g, r = color
        colored = np.zeros_like(result)
        colored[:, :, 0] = mask * b
        colored[:, :, 1] = mask * g
        colored[:, :, 2] = mask * r
        result = cv2.addWeighted(colored.astype(np.float32), alpha, result, 1.0 - alpha, 0.0)
    return result.clip(0, 255).astype(np.uint8)


def draw_ans_pns_line(img, coords_orig, color=(0, 255, 255), thickness=3):
    if coords_orig.shape[0] < 8:
        return img
    ans = coords_orig[6]
    pns = coords_orig[7]
    pt1 = (int(round(ans[0])), int(round(ans[1])))
    pt2 = (int(round(pns[0])), int(round(pns[1])))
    img = cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    img = cv2.circle(img, pt1, 6, color, -1, lineType=cv2.LINE_AA)
    img = cv2.circle(img, pt2, 6, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, "ANS", (pt1[0] + 8, pt1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)
    cv2.putText(img, "PNS", (pt2[0] + 8, pt2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, lineType=cv2.LINE_AA)
    return img


def draw_keypoints(img, coords_orig, confidences, dot_radius=7):
    for i, ((x, y), conf) in enumerate(zip(coords_orig, confidences)):
        px, py = int(round(x)), int(round(y))
        cv2.circle(img, (px, py), dot_radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (px, py), dot_radius + 2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(img, f"{i}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    return img


def plot_inference(img_rgb, coords_orig, confidences, seg_masks, orig_size, output_path):
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]
    img = blend_masks_cv2(img_rgb.copy(), seg_masks, colors, alpha=0.4)
    img = draw_ans_pns_line(img, coords_orig, color=(0, 255, 255), thickness=3)
    img = draw_keypoints(img, coords_orig, confidences, dot_radius=7)

    # Legend
    legend_y, legend_x = 30, img.shape[1] - 220
    line_h = 28
    entries = [
        ("Upper_incisor", (255, 80, 80)),
        ("Labial_bone",   (80, 255, 80)),
        ("Palatal_bone",  (80, 80, 255)),
        ("ANS-PNS",       (0, 255, 255)),
        ("Keypoints 0-9", (0, 255, 0)),
    ]
    cv2.rectangle(img, (legend_x - 10, 10), (img.shape[1] - 10, 10 + len(entries) * line_h + 20), (0, 0, 0), -1, lineType=cv2.LINE_AA)
    for label, color in entries:
        cv2.rectangle(img, (legend_x, legend_y), (legend_x + 25, legend_y + 18), color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (legend_x + 35, legend_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        legend_y += line_h

    # Title bar
    title = f"Holdout: {output_path.stem}  |  Landmarks: 10  |  Masks: 3  |  Device: {DEVICE}"
    cv2.rectangle(img, (0, img.shape[0] - 40), (img.shape[1], img.shape[0]), (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, title, (20, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, lineType=cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def ans_pns_distance_mm(coords_orig, image_id, cal_dict):
    if coords_orig.shape[0] < 8:
        return None
    ans = coords_orig[6]
    pns = coords_orig[7]
    px_dist = math.sqrt((ans[0] - pns[0])**2 + (ans[1] - pns[1])**2)
    mm_per_px = cal_dict.get(image_id)
    if mm_per_px is None:
        return None
    return px_dist * mm_per_px


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" BATCH HOLDOUT DIAGNOSTIC — 5-10 Image Visual Report")
    print("=" * 70)

    # Hardcoded holdout list (NOT in training set)
    HOLDOUT_IMAGES = [
        "Patient100_T1",
        "Patient101_T1",
        "Patient102_T1",
        "Patient82_T1",
        "Patient328_T1",
        "Patient193_T1",
        "Patient175_T1",
        "Patient20_T2",
    ]
    IMAGE_DIR = ROOT / "data" / "raw" / "images"

    # ── Load models ─────────────────────────────────────────────────────────
    print("\n[1] Loading models ...")
    landmark_ckpt = ROOT / "data" / "processed" / "checkpoints" / "fold1_best.pth"
    if not landmark_ckpt.exists():
        landmark_ckpt = ROOT / "outputs" / "checkpoints" / "fold1_best.pth"
    assert landmark_ckpt.exists(), f"LANDMARK CKPT NOT FOUND: {landmark_ckpt}"

    landmark_model = build_landmark_model(num_keypoints=NUM_KEYPOINTS, pretrained=False)
    ckpt = torch.load(landmark_ckpt, map_location=DEVICE, weights_only=False)
    # Properly extract model_state_dict — ckpt is a wrapper dict with keys
    # ['model_state_dict', 'fold_mre_argmax', 'fold_idx', ...]
    state = ckpt.get("model_state_dict", ckpt)
    # Strip any uncertainty/FEUPE keys that may be merged in checkpoint
    state = {k: v for k, v in state.items() if "uncertainty" not in k}
    landmark_model.load_state_dict(state, strict=False)
    landmark_model = landmark_model.to(DEVICE)
    landmark_model.eval()
    fold_mre = ckpt.get("fold_mre_argmax", "?")
    print(f"  Landmark: {landmark_ckpt.name}  fold_mre={fold_mre} mm")

    models_base = ROOT / "models"
    seg_ckpt = find_best_segmentation_checkpoint(models_base)
    seg_model = build_segmentation_model(num_classes=3, encoder_name="resnet34", pretrained=False)
    seg_state = torch.load(seg_ckpt, map_location=DEVICE, weights_only=False)
    seg_model.load_state_dict(seg_state, strict=False)
    seg_model = seg_model.to(DEVICE)
    seg_model.eval()
    print(f"  Segmentation: {seg_ckpt.parent.name}")

    # Calibration
    cal_csv = ROOT / "data" / "processed" / "calibration.csv"
    cal_dict = load_calibration(cal_csv)
    print(f"  Calibration entries: {len(cal_dict)}")

    OUTPUT_DIR = ROOT / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process each holdout image ──────────────────────────────────────────
    print(f"\n[2] Processing {len(HOLDOUT_IMAGES)} holdout images ...")
    results = []

    for idx, image_id in enumerate(HOLDOUT_IMAGES, start=1):
        img_path = None
        for ext in [".jpg", ".png", ".JPG", ".PNG"]:
            p = IMAGE_DIR / f"{image_id}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            print(f"  SKIP {image_id}: file not found")
            continue

        img_raw = cv2.imread(str(img_path))
        orig_h, orig_w = img_raw.shape[:2]
        img_rgb_orig = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # Preprocess
        img_tensor, img_rgb = preprocess_image(img_path)

        # Landmark inference
        coords_input, confidences = run_landmark_inference(landmark_model, img_tensor)
        coords_orig = decode_coordinates_to_original(coords_input, (orig_h, orig_w), INPUT_SIZE)

        # Segmentation inference
        seg_masks = run_segmentation_inference(seg_model, img_tensor, (orig_h, orig_w))

        # ANS-PNS distance
        ans_pns_mm = ans_pns_distance_mm(coords_orig, image_id, cal_dict)

        # Save PNG
        out_png = OUTPUT_DIR / f"holdout_diagnostic_{idx:02d}.png"
        plot_inference(img_rgb_orig, coords_orig, confidences, seg_masks, (orig_h, orig_w), out_png)

        results.append({
            "idx": idx,
            "image_id": image_id,
            "coords_orig": coords_orig,
            "confidences": confidences,
            "orig_size": (orig_h, orig_w),
            "ans_pns_mm": ans_pns_mm,
            "out_png": out_png,
        })
        print(f"  [{idx:02d}] {image_id} -> {out_png.name}  "
              f"Upper_tip=({coords_orig[0,0]:.0f},{coords_orig[0,1]:.0f})  "
              f"ANS=({coords_orig[6,0]:.0f},{coords_orig[6,1]:.0f})  "
              f"PNS=({coords_orig[7,0]:.0f},{coords_orig[7,1]:.0f})  "
              f"ANS-PNS={ans_pns_mm:.1f}mm" if ans_pns_mm else f"  [{idx:02d}] {image_id} -> {out_png.name}")

    # ── Print markdown report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" HOLDOUT DIAGNOSTIC REPORT")
    print("=" * 70)

    header = (
        "| # | Patient ID | Upper_tip (0) | ANS (6) | PNS (7) | "
        "ANS-PNS mm | Confidence | PNG |"
    )
    sep = header.replace("|", "|").replace("—", "-")
    print(header)
    print("|" + "-|" * (header.count("|") - 1))

    for r in results:
        ut = r["coords_orig"][0]
        ans = r["coords_orig"][6]
        pns = r["coords_orig"][7]
        conf_str = f"{r['confidences'][0]:.3f}" if r["confidences"][0] else "N/A"
        ans_pns_str = f"{r['ans_pns_mm']:.2f} mm" if r["ans_pns_mm"] else "no_cal"
        print(
            f"| {r['idx']} | {r['image_id']} "
            f"| ({ut[0]:.0f}, {ut[1]:.0f}) "
            f"| ({ans[0]:.0f}, {ans[1]:.0f}) "
            f"| ({pns[0]:.0f}, {pns[1]:.0f}) "
            f"| {ans_pns_str} "
            f"| {conf_str} "
            f"| {r['out_png'].name} |"
        )

    print("\n" + "=" * 70)
    print(" OUTPUT FILES:")
    for r in results:
        print(f"  {r['out_png']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
