#!/usr/bin/env python3
"""
Visual Evaluation Pipeline for Phase 2B — Alveolar Bone Segmentation.
Automatically finds the best DeepLabV3+ model, runs inference on a validation image,
overlays predicted polygons + ground truth landmarks, and saves to reports/visual_results/.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]
NUM_CLASSES     = len(POLYGON_CLASSES) + 1
CLASS_TO_IDX     = {cls: i+1 for i, cls in enumerate(POLYGON_CLASSES)}
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

IMAGES_DIR   = PROJECT_ROOT / "data/raw/images"
SEG_JSON     = PROJECT_ROOT / "data/processed/segmentation_train.json"
LANDMARKS_JSON = PROJECT_ROOT / "data/processed/landmarks_clean.json"
MODEL_DIR    = PROJECT_ROOT / "models"
OUT_DIR      = PROJECT_ROOT / "reports/visual_results"

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]

# Colors per class (RGB, matplotlib format)
CLASS_COLORS = {
    "Upper_incisor": "#FF4444",  # red
    "Labial_bone":   "#44AAFF",  # blue
    "Palatal_bone":  "#44FF88",  # green
}

LM_COLOR   = "#FFFF00"  # yellow landmarks
LM_ALPHA    = 0.85

# ─── CLAHE helper ─────────────────────────────────────────────────────────
def apply_clahe(img: np.ndarray, clip_limit=2.0, tile_grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    chs = [clahe.apply(c) for c in cv2.split(img.astype(np.uint8))]
    return cv2.merge(chs)

# ─── MODEL FINDER ──────────────────────────────────────────────────────────
def find_best_model():
    """Find the 4-class model with highest val_dice from post-recovery grid search."""
    candidates = []
    for cfg_path in MODEL_DIR.glob("exp????_*/config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
            if cfg.get("num_classes") == 4:
                dice = cfg.get("val_dice", 0)
                candidates.append((dice, cfg, cfg_path.parent))
        except Exception:
            pass
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        raise FileNotFoundError("No 4-class model configs found in models/")
    best_dice, best_cfg, model_dir = candidates[0]
    print(f"[INFO] Best 4-class model: {model_dir.name}  dice={best_dice:.4f}  arch={best_cfg['arch_name']}  nc={best_cfg.get('num_classes','?')}")
    return model_dir, best_cfg

# ─── DATASET HELPERS ───────────────────────────────────────────────────────
def load_records():
    return json.loads(SEG_JSON.read_text())

def load_landmarks():
    return json.loads(LANDMARKS_JSON.read_text())

# ─── INFERENCE ────────────────────────────────────────────────────────────
def build_model(cfg):
    arch = cfg["arch_name"]
    enc  = cfg["encoder_name"]
    base = smp.DeepLabV3Plus(
        encoder_name=enc, encoder_weights="imagenet",
        in_channels=3, classes=NUM_CLASSES, activation=None,
    )
    if torch.cuda.device_count() > 1:
        base = nn.DataParallel(base)
    ckpt = cfg.get("checkpoint_path") or (MODEL_DIR / cfg["run_id"] / "best_model.pt")
    base.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    base = base.to(DEVICE)
    base.eval()
    return base

def run_inference(model, image_path, use_clahe=True, input_size=(512, 512)):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    if use_clahe:
        img = apply_clahe(img)

    img_resized = cv2.resize(img, (input_size[1], input_size[0]))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    pred_mask_large = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred_mask_large, img, orig_w, orig_h

# ─── POLYGON DECODING ──────────────────────────────────────────────────────
def decode_polygons(mask):
    """Convert argmax mask to list-of-polygons for each class using contours."""
    polygons = {}
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        bin_mask = (mask == cls_idx).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            eps = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            pts = approx.reshape(-1, 2).tolist()
            pts_list.extend(pts) if pts else None
        polygons[cls_name] = pts_list
    return polygons

# ─── PLOTTING ─────────────────────────────────────────────────────────────
def scale_pts(pts, orig_size, display_size):
    """Scale points from original image space to display space."""
    scale_x = display_size[0] / orig_size[0]
    scale_y = display_size[1] / orig_size[1]
    return [(x * scale_x, y * scale_y) for (x, y) in pts]

def get_landmarks_for_image(image_id, landmarks_records):
    """Return list of {name, x, y, visible} for a given image_id."""
    for rec in landmarks_records:
        if rec.get("image_id") == image_id:
            return rec.get("keypoints", [])
    return []

def plot_and_save(pred_polygons, image_id, image_rgb,
                  landmarks_records, out_path):
    """Plot polygons + landmarks on native-res image, then resize for display."""
    H_orig, W_orig = image_rgb.shape[:2]

    # Draw polygons on a copy at native resolution
    overlay = image_rgb.copy()
    legend_patches = []

    for cls_name, pts in pred_polygons.items():
        if not pts:
            continue
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        color_hex = CLASS_COLORS.get(cls_name, "#FFFFFF")
        color_rgb = matplotlib.colors.hex2color(color_hex)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        color_cv  = (int(255 * color_rgb[0]), int(255 * color_rgb[1]), int(255 * color_rgb[2]))

        cv2.fillPoly(overlay, [pts_arr], color_cv)
        cv2.polylines(overlay, [pts_arr], isClosed=True, color=color_bgr, thickness=2)

        legend_patches.append(
            plt.Line2D([0], [0], color=color_hex, lw=3, label=f"Pred: {cls_name}"))

    # Draw landmarks at native-pixel coords on native-res overlay (for polygon alignment)
    lm_list = get_landmarks_for_image(image_id, landmarks_records)
    for lm in lm_list:
        if not lm.get("visible", True):
            continue
        x, y = int(lm["x"]), int(lm["y"])
        cv2.circle(overlay, (x, y), radius=6, color=(0, 255, 255), thickness=-1)
        cv2.putText(overlay, lm["name"], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Now resize the composite to display size
    DISPLAY_W, DISPLAY_H = 1200, 900
    scale_x = DISPLAY_W / W_orig
    scale_y = DISPLAY_H / H_orig
    display_img = cv2.resize(overlay, (DISPLAY_W, DISPLAY_H))

    legend_patches.append(
        plt.Line2D([0], [0], marker="o", color=LM_COLOR, linestyle="None",
                   markersize=10, label="Ground-truth landmarks"))

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(display_img, extent=(0, W_orig, 0, H_orig), origin='upper')
    ax.set_title(f"Segmentation + Landmark Overlay — {image_id}", fontsize=14, pad=10)
    ax.set_xlim(0, W_orig)
    ax.set_ylim(0, H_orig)
    ax.set_xlabel(f"Width ({W_orig} px)")
    ax.set_ylabel(f"Height ({H_orig} px)")
    ax.axis("off")
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
              facecolor="white", edgecolor="gray")
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved visualization to {out_path}")

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Specific image filename (e.g. Patient03_T1.jpg). "
                                                "Defaults to first val image.")
    ap.add_argument("--model-dir", default=None, help="Explicit model directory.")
    ap.add_argument("--out-name", default=None, help="Output filename.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Find best model ──────────────────────────
    if args.model_dir:
        model_dir = Path(args.model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        print(f"[INFO] Using specified model: {model_dir.name}")
    else:
        model_dir, cfg = find_best_model()

    ckpt_file = model_dir / "best_model.pt"
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_file}")
        sys.exit(1)
    cfg["checkpoint_path"] = str(ckpt_file)

    model = build_model(cfg)
    use_clahe = cfg.get("use_clahe", True)
    arch = cfg.get("arch_name", "DeepLabV3Plus")
    print(f"[INFO] Model built: {arch} | CLAHE={use_clahe}")

    # ── Pick image ───────────────────────────────
    records = load_records()
    landmarks = load_landmarks()

    if args.image:
        image_path = IMAGES_DIR / args.image
        image_id = Path(args.image).stem
    else:
        # Use first patient with both landmarks and polygons
        for rec in records:
            img_id = rec.get("image_id", "")
            has_kp = any(lm.get("image_id") == img_id for lm in landmarks) if isinstance(landmarks[0], dict) else False
            if rec.get("has_landmarks") and rec.get("polygons"):
                image_path = IMAGES_DIR / rec["filename"]
                image_id = img_id
                break
        else:
            rec = records[0]
            image_path = IMAGES_DIR / rec["filename"]
            image_id = rec.get("image_id", Path(rec["filename"]).stem)

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)
    print(f"[INFO] Running on image: {image_path.name}  (id={image_id})")

    # ── Inference ────────────────────────────────
    pred_mask, image_rgb, W, H = run_inference(model, image_path, use_clahe=use_clahe)
    pred_polygons = decode_polygons(pred_mask)

    # ── Plot & save ─────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name or f"eval_{image_id}_{ts}.png"
    out_path = OUT_DIR / out_name
    plot_and_save(pred_polygons, image_id, image_rgb, landmarks, out_path)
    print(f"[DONE] {out_path}")

if __name__ == "__main__":
    main()
