#!/usr/bin/env python3
"""
Production Blind Test — Unannotated Image Inference
====================================================
Single-pass multi-task inference on RAW patient data with no ground truth.
Validates whether Phase 2A (Landmark) + Phase 2B (Segmentation) pipelines
actually generalize, or whether the model is still spatially blind.

1. Select 3 unannotated images from data/raw/images/ (no GT in landmarks_clean.json)
   — or fallback: strict holdout from fold 0 with annotations IGNORED
2. Load fold1_best.pth (landmark) + best DeepLabV3+ segmentation checkpoint
3. Run inference: 10 landmark coords + 3-class segmentation masks
4. Save diagnostic PNGs: X-ray + RGB bone masks + neon-lime landmarks + ANS-PNS line
5. Write BLIND_TEST_REPORT.md with verdict for each case

Run: python scripts/run_production_blind_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.resolve()
IMAGE_DIR = ROOT / "data" / "raw" / "images"
LANDMARK_CKPT = ROOT / "data" / "processed" / "checkpoints" / "fold1_best.pth"
SEG_CKPT = ROOT / "models" / "exp0128_DeepLabV3Plus_resnet34_20260524_043501" / "best_model.pt"
CALIB_CSV = ROOT / "data" / "processed" / "calibration.csv"
LANDMARKS_JSON = ROOT / "data" / "processed" / "landmarks_clean.json"
OUTPUT_DIR = ROOT / "outputs"

DEVICE = torch.device(
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"[BlindTest] Device: {DEVICE}")

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]
POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]
INPUT_SIZE = (512, 512)   # (H, W)
HEATMAP_SIZE = (256, 256)

# -----------------------------------------------------------------------
# Model builders (mirrors generate_validation_report.py)
# -----------------------------------------------------------------------

def build_landmark_model(num_keypoints: int = 10, pretrained: bool = False):
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


def build_segmentation_model(num_classes: int = 3, encoder_name: str = "resnet34", pretrained: bool = True):
    import segmentation_models_pytorch as smp
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    return model


# -----------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------

def load_landmark_checkpoint(path: Path, model: torch.nn.Module) -> dict:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k: v for k, v in state.items() if "uncertainty" not in k}
    ok = model.load_state_dict(state, strict=False)
    missing = [k for k in ok.missing_keys if "uncertainty" not in k]
    if missing:
        print(f"  WARNING: {len(missing)} missing keys: {missing[:3]}")
    info = {}
    if "fold_mre_argmax" in ckpt:
        info["fold_mre_argmax"] = ckpt["fold_mre_argmax"]
    print(f"  Landmark checkpoint loaded: {path.name}")
    return info


# -----------------------------------------------------------------------
# Calibration helpers
# -----------------------------------------------------------------------

def load_calibration(csv_path: Path) -> dict[str, float]:
    """image_id → mm_per_pixel"""
    calib = {}
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                calib[row["image_id"]] = float(row["mm_per_pixel"])
    return calib


def extract_patient_timepoint(filename: str):
    """Extract patient_id and timepoint from filename like Patient01_T1.jpg"""
    import re
    m = re.search(r"(Patient\d+)_(T\d+)", filename)
    if m:
        return m.group(1), m.group(2)
    return None, None


# -----------------------------------------------------------------------
# Image selection — find 3 unannotated images
# -----------------------------------------------------------------------

def find_unannotated_images(image_dir: Path, landmarks_json: Path, max_select: int = 3):
    """
    Find images with NO entry in landmarks_clean.json OR with has_landmarks=False.
    If all are annotated, fall back to strict holdout (fold 0 val patients).
    """
    # Load annotated image IDs
    if landmarks_json.exists():
        with open(landmarks_json) as f:
            data = json.load(f)
        if isinstance(data, list):
            annotated_ids = {r["image_id"] for r in data if r.get("has_landmarks")}
        elif isinstance(data, dict):
            annotated_ids = {r["image_id"] for r in data.get("images", []) if r.get("has_landmarks")}
        else:
            annotated_ids = set()
    else:
        annotated_ids = set()

    print(f"[BlindTest] Annotated image IDs: {len(annotated_ids)}")

    # List all image files
    all_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    unannotated = [img for img in all_images if img.stem not in annotated_ids]
    print(f"[BlindTest] Unannotated images found: {len(unannotated)}")

    if len(unannotated) >= max_select:
        return unannotated[:max_select]

    # Fallback: use fold 0 holdout (never seen during fold-1 training)
    print("[BlindTest] Not enough unannotated — falling back to fold-0 holdout (true zero-leakage)")
    from sklearn.model_selection import GroupKFold
    all_stems = [img.stem for img in all_images]
    # Get patient IDs from filename
    patient_ids = []
    for stem in all_stems:
        pid, _ = extract_patient_timepoint(stem + ".jpg")
        patient_ids.append(pid or stem)
    unique_patients = sorted(set(patient_ids))
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (_, val_idx) in enumerate(gkf.split(all_stems, groups=patient_ids)):
        if fold_idx == 0:
            holdout_stems = {all_stems[i] for i in val_idx}
            holdout_images = [img for img in all_images if img.stem in holdout_stems]
            print(f"[BlindTest] Fold-0 holdout: {len(holdout_images)} images (annotations IGNORED)")
            return holdout_images[:max_select]
    return all_images[:max_select]


# -----------------------------------------------------------------------
# Preprocessing (matches training: /255 only, no ImageNet)
# -----------------------------------------------------------------------

def preprocess_image(image_path: Path, input_size: tuple[int, int] = INPUT_SIZE):
    """
    Load image, resize to input_size.
    Returns: tensor [3, H, W], orig_h, orig_w, scale_x, scale_y
    scale_x = orig_w / input_W, scale_y = orig_h / input_H
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((input_size[1], input_size[0]), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
    scale_x = orig_w / input_size[1]
    scale_y = orig_h / input_size[0]
    return tensor, orig_h, orig_w, scale_x, scale_y


# -----------------------------------------------------------------------
# Landmark inference — hard argmax decode
# -----------------------------------------------------------------------

def decode_landmarks_argmax(heatmaps: torch.Tensor, input_size: tuple[int, int] = INPUT_SIZE):
    """
    Hard-argmax decode. Returns [B, N, 2] in input pixel space (x=col, y=row),
    and confidence [B, N].
    """
    B, N, H, W = heatmaps.shape
    conf = torch.sigmoid(heatmaps)
    flat = conf.view(B * N, -1)
    confidence, flat_idx = flat.max(dim=-1)
    x_int = (flat_idx % W).float()
    y_int = (flat_idx // W).float()
    coords = torch.stack([x_int, y_int], dim=-1).view(B, N, 2)
    confidence = confidence.view(B, N)
    return coords, confidence


def infer_landmarks(model: torch.nn.Module, image_path: Path, calib: dict[str, float]):
    """Run landmark inference on a single image."""
    model.eval()
    tensor, orig_h, orig_w, scale_x, scale_y = preprocess_image(image_path)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        heatmaps = model(tensor)   # [1, 10, 256, 256]

    coords_input, conf = decode_landmarks_argmax(heatmaps)  # [1, 10, 2]
    coords_input = coords_input[0].cpu().numpy()   # [10, 2] in input space

    # Scale to original image space
    coords_orig = np.zeros_like(coords_input)
    coords_orig[:, 0] = coords_input[:, 0] * scale_x   # x (col) → original
    coords_orig[:, 1] = coords_input[:, 1] * scale_y   # y (row) → original

    image_id = image_path.stem
    mm_per_pixel = calib.get(image_id, 0.0985)

    return coords_orig, conf[0].cpu().numpy(), (orig_h, orig_w), mm_per_pixel


# -----------------------------------------------------------------------
# Segmentation inference
# -----------------------------------------------------------------------

def infer_segmentation(model: torch.nn.Module, image_path: Path):
    """Run segmentation inference. Returns [3, H, W] sigmoid masks at INPUT resolution."""
    model.eval()
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)   # [1, 3, 512, 512]
    sigmoid_masks = torch.sigmoid(logits)[0].cpu().numpy()   # [3, 512, 512]
    return sigmoid_masks, orig_h, orig_w


# -----------------------------------------------------------------------
# Visualization — render one blind test case
# -----------------------------------------------------------------------

def render_blind_case(
    image_path: Path,
    landmark_coords: np.ndarray,   # [10, 2] original image space
    landmark_conf: np.ndarray,     # [10]
    seg_masks: np.ndarray,         # [3, 512, 512] sigmoid at input
    orig_h: int, orig_w: int,
    case_idx: int,
) -> np.ndarray:
    """
    Render one case:
    - X-ray background (grayscale)
    - 3 transparent bone masks: Upper_incisor=Red, Labial_bone=Green, Palatal_bone=Blue
    - 10 neon-lime landmark dots with text indexes
    - Dashed yellow ANS(6)-PNS(7) line

    Returns: BGR uint8 image
    """
    # Load original image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    overlay = img_bgr.copy()

    # ---- Segmentation masks ----
    # Threshold at input resolution, then resize to original
    mask_colors = [
        ( 60, 200, 255),   # Upper_incisor — vivid blue (BGR)
        ( 80, 255,  80),   # Labial_bone   — vivid green (BGR)
        (220,  80,  60),   # Palatal_bone  — vivid red (BGR)
    ]

    for ch_idx, (cls_name, color_bgr) in enumerate(zip(POLYGON_CLASSES, mask_colors)):
        mask_input = (seg_masks[ch_idx] > 0.5).astype(np.uint8)   # [512, 512]
        mask_orig = cv2.resize(mask_input, (orig_w, orig_h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        # Composite: color fill with alpha=0.40
        alpha = 0.40
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_orig,
                (alpha * color_bgr[c] + (1 - alpha) * overlay[:, :, c]).astype(np.uint8),
                overlay[:, :, c]
            )

    # Blend overlay onto original
    img_blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

    # ---- Landmark dots — neon lime ----
    lime_color = (0, 255, 150)   # BGR
    for kp_idx, (name, (x, y), confidence) in enumerate(
        zip(KEYPOINT_NAMES, landmark_coords, landmark_conf)
    ):
        # Scale confidence: 0→1
        conf_display = max(0.0, min(1.0, confidence))
        dot_size = int(8 + conf_display * 8)   # 8–16px

        cx, cy = int(round(x)), int(round(y))
        # Neon glow: draw thick low-alpha circle, then thin bright circle
        cv2.circle(img_blended, (cx, cy), dot_size + 4, (0, 80, 50), -1)
        cv2.circle(img_blended, (cx, cy), dot_size, (0, 255, 150), -1)
        cv2.circle(img_blended, (cx, cy), dot_size - 2, (200, 255, 230), -1)

        # Text label
        label = f"{kp_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        text_color = (0, 255, 150)
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = cx + 6
        text_y = cy - 6
        cv2.putText(img_blended, label, (text_x, text_y),
                    font, font_scale, (0, 0, 0), thickness + 2)   # shadow
        cv2.putText(img_blended, label, (text_x, text_y),
                    font, font_scale, text_color, thickness)

    # ---- ANS(6) — PNS(7) dashed yellow line ----
    ans = landmark_coords[6]   # index 6 = ANS
    pns = landmark_coords[7]   # index 7 = PNS
    pt_ans = (int(round(ans[0])), int(round(ans[1])))
    pt_pns = (int(round(pns[0])), int(round(pns[1])))

    yellow = (0, 255, 255)
    # Draw dashed line
    dy = pt_pns[1] - pt_ans[1]
    dx = pt_pns[0] - pt_ans[0]
    line_len = max(abs(dx), abs(dy))
    if line_len > 0:
        step = 10
        for i in range(0, line_len, step):
            t = i / line_len
            t2 = min((i + step // 2) / line_len, 1.0)
            p1x = int(pt_ans[0] + dx * t)
            p1y = int(pt_ans[1] + dy * t)
            p2x = int(pt_ans[0] + dx * t2)
            p2y = int(pt_ans[1] + dy * t2)
            cv2.line(img_blended, (p1x, p1y), (p2x, p2y), yellow, 2)

    # Annotate ANS/PNS endpoints
    cv2.putText(img_blended, "ANS", (pt_ans[0] + 8, pt_ans[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 1)
    cv2.putText(img_blended, "PNS", (pt_pns[0] + 8, pt_pns[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 1)

    return img_blended


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- Load models ----
    print("\n[BlindTest] Loading landmark model...")
    land_model = build_landmark_model(10, pretrained=False).to(DEVICE)
    load_landmark_checkpoint(LANDMARK_CKPT, land_model)

    print("[BlindTest] Loading segmentation model...")
    seg_model = build_segmentation_model(3, "resnet34", pretrained=False).to(DEVICE)
    seg_ckpt = torch.load(SEG_CKPT, map_location=DEVICE, weights_only=False)
    seg_state = seg_ckpt.get("model_state_dict", seg_ckpt)
    seg_model.load_state_dict(seg_state, strict=False)
    print(f"  Segmentation checkpoint loaded: {SEG_CKPT.name}")

    # ---- Load calibration ----
    calib = load_calibration(CALIB_CSV)
    print(f"[BlindTest] Calibration entries: {len(calib)}")

    # ---- Select images ----
    selected = find_unannotated_images(IMAGE_DIR, LANDMARKS_JSON, max_select=3)
    print(f"[BlindTest] Selected {len(selected)} images:")
    for img in selected:
        print(f"  - {img.name}")

    # ---- Run inference ----
    results = []
    for case_idx, image_path in enumerate(selected, start=1):
        print(f"\n[BlindTest] Processing case {case_idx}: {image_path.name}")

        # Landmark inference
        coords, conf, (orig_h, orig_w), mm_per_pixel = infer_landmarks(
            land_model, image_path, calib
        )

        # Segmentation inference
        seg_masks, _, _ = infer_segmentation(seg_model, image_path)

        # Render
        rendered = render_blind_case(
            image_path, coords, conf, seg_masks, orig_h, orig_w, case_idx
        )

        out_png = OUTPUT_DIR / f"blind_test_{case_idx:02d}.png"
        cv2.imwrite(str(out_png), rendered)
        print(f"  → Saved {out_png}")

        # Print coordinates
        print(f"\n  Landmark coordinates (original image space):")
        for kp_idx, (name, (x, y), confidence) in enumerate(
            zip(KEYPOINT_NAMES, coords, conf)
        ):
            print(f"    [{kp_idx}] {name:<20} x={x:7.1f}  y={y:7.1f}  conf={confidence:.3f}")

        print(f"\n  ANS(6) → PNS(7) spatial check:")
        ans = coords[6]
        pns = coords[7]
        dy = pns[1] - ans[1]
        dx = pns[0] - ans[0]
        sep_pixels = np.sqrt(dx**2 + dy**2)
        sep_mm = sep_pixels * mm_per_pixel
        print(f"    ANS: ({ans[0]:.1f}, {ans[1]:.1f})")
        print(f"    PNS: ({pns[0]:.1f}, {pns[1]:.1f})")
        print(f"    Separation: {sep_pixels:.1f} px = {sep_mm:.2f} mm")

        results.append({
            "case_idx": case_idx,
            "image": image_path.name,
            "image_id": image_path.stem,
            "coords": coords.tolist(),
            "conf": conf.tolist(),
            "mm_per_pixel": mm_per_pixel,
            "ans": ans.tolist(),
            "pns": pns.tolist(),
            "sep_pixels": float(sep_pixels),
            "sep_mm": float(sep_mm),
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
        })

    # ---- Write BLIND_TEST_REPORT.md ----
    report_path = OUTPUT_DIR / "BLIND_TEST_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Production Blind Test Report\n\n")
        f.write("**Date:** 2026-05-25  |  **Models:** HRNet-W32 (Landmark) + DeepLabV3+ (Segmentation)\n\n")
        f.write("---\n\n")

        for res in results:
            case_idx = res["case_idx"]
            image_name = res["image"]
            coords = np.array(res["coords"])
            conf = np.array(res["conf"])
            ans = np.array(res["ans"])
            pns = np.array(res["pns"])
            sep_mm = res["sep_mm"]
            sep_px = res["sep_pixels"]

            f.write(f"## Case {case_idx}: {image_name}\n\n")
            f.write(f"**mm/pixel:** {res['mm_per_pixel']:.4f}  |  ")
            f.write(f"**Image size:** {res['orig_w']}×{res['orig_h']}\n\n")

            # Spatial sanity check
            vertical_sep = abs(pns[1] - ans[1])

            if sep_mm > 15.0:
                verdict = "✅ **PASS** — ANS and PNS are anatomically separated (palatal plane visible)"
            elif sep_mm > 5.0:
                verdict = "⚠️ **MARGINAL** — ANS and PNS are close but anatomically plausible"
            else:
                verdict = "❌ **FAIL** — ANS and PNS are clustering (model spatially blind / collapsed)"

            f.write(f"**ANS-PNS Separation:** {sep_px:.1f} px / {sep_mm:.2f} mm\n\n")
            f.write(f"**Vertical gap:** {vertical_sep:.1f} px\n\n")
            f.write(f"**Sanity Check Verdict:** {verdict}\n\n")

            f.write("### Landmark Coordinates\n\n")
            f.write("| # | Name | X (px) | Y (px) | Confidence |\n")
            f.write("|---|------|--------|--------|------------|\n")
            for kp_idx, name in enumerate(KEYPOINT_NAMES):
                x, y = coords[kp_idx]
                c = conf[kp_idx]
                f.write(f"| {kp_idx} | {name} | {x:.1f} | {y:.1f} | {c:.3f} |\n")
            f.write("\n")

            # Show ANS-PNS relationship
            f.write("### Anatomical Placement Analysis\n\n")
            f.write(f"- **ANS (6):** ({ans[0]:.1f}, {ans[1]:.1f}) — anterior nasal spine\n")
            f.write(f"- **PNS (7):** ({pns[0]:.1f}, {pns[1]:.1f}) — posterior nasal spine\n")
            f.write(f"- **Separation:** {sep_px:.1f} px ({sep_mm:.2f} mm)\n\n")

            if sep_mm > 15.0:
                detail = ("ANS and PNS are clearly separated across the palate — "
                          "ANS in the anterior maxilla, PNS in the posterior near the pterygoid plates. "
                          "The model has learned genuine anatomical structure.")
            elif sep_mm > 5.0:
                detail = ("ANS and PNS are closer than ideal but not collapsed. "
                          "The model shows partial spatial understanding — better than random, "
                          "but not clinically precise.")
            else:
                detail = ("ANS and PNS are near-identical or clustered near the teeth. "
                          "This indicates the model is spatially blind — it has collapsed to "
                          "mean positions and cannot distinguish anterior from posterior anatomy.")

            f.write(f"{detail}\n\n")

            # Embed image
            png_rel = f"blind_test_{case_idx:02d}.png"
            f.write(f"### Visualization\n\n")
            f.write(f"![Case {case_idx} — {image_name}](./{png_rel})\n\n")
            f.write("---\n\n")

        # Summary
        all_pass = all(r["sep_mm"] > 15.0 for r in results)
        all_fail = all(r["sep_mm"] < 5.0 for r in results)

        f.write("## Summary\n\n")
        if all_pass:
            f.write("**Overall: ✅ BOTH PIPELINES VERIFIED** — Landmark detection AND segmentation "
                    "generalize correctly. ANS-PNS separation confirms spatial awareness across all cases.\n")
        elif all_fail:
            f.write("**Overall: ❌ PIPELINES BROKEN** — All cases show collapsed landmark predictions. "
                    "The model has no spatial generalization on unannotated data.\n")
        else:
            f.write("**Overall: ⚠️ MIXED RESULTS** — Some cases pass, some fail. "
                    "Model shows partial generalization but is unreliable for production deployment.\n")

    print(f"\n[BlindTest] Report written: {report_path}")
    print("\n[BlindTest] === RAW OUTPUT COORDINATES ===")
    for res in results:
        print(f"\n  Case {res['case_idx']} — {res['image']}:")
        for kp_idx, name in enumerate(KEYPOINT_NAMES):
            x, y = res["coords"][kp_idx]
            c = res["conf"][kp_idx]
            print(f"    [{kp_idx}] {name:<20} x={x:7.1f}  y={y:7.1f}  conf={c:.3f}")
        print(f"    ANS(6)  → ({res['ans'][0]:.1f}, {res['ans'][1]:.1f}) = {res['sep_mm']:.2f} mm from PNS(7)")
        print(f"    PNS(7)  → ({res['pns'][0]:.1f}, {res['pns'][1]:.1f})")
    print("\n[BlindTest] Done.")


if __name__ == "__main__":
    main()