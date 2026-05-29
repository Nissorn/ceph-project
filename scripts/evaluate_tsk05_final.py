#!/usr/bin/env python3
"""
TSK-05 Final Evaluation: End-to-end pipeline with TSK-04 champion model.
Uses HRNet-W32 landmark model + TSK-04 (sliding window) + geometric snapping.
Evaluates MRE (mm) and SDR@2mm on the full validation set.
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── constants ─────────────────────────────────────────────────────────────────
SLIDING_WINDOW_SIZE = 512
SLIDING_STRIDE = 256
_SLIDING_SIGMA = SLIDING_WINDOW_SIZE / 4
INPUT_SIZE = (512, 512)
HEATMAP_SIZE = (256, 256)
NUM_KEYPOINTS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_DIR     = PROJECT_ROOT / "data/raw/images"
LANDMARKS_JSON = PROJECT_ROOT / "data/processed/landmarks_clean.json"
CALIB_CSV      = PROJECT_ROOT / "data/processed/calibration_clean.csv"

TSK04_DIR   = PROJECT_ROOT / "models" / "tversky_deepLabV3plus_resnet34_20250529_20260529_094221"
LM_CKPT     = PROJECT_ROOT / "data" / "processed" / "checkpoints" / "fold1_best.pth"
OUTPUT_DIR  = PROJECT_ROOT / "reports/visual_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]

# ── CLAHE ───────────────────────────────────────────────────────────────────
def _apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ── Landmark model (HRNet-W32 — same as analysis_service.py) ────────────────
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


# ── Segmentation model (DeepLabV3Plus — TSK-04) ───────────────────────────────
def _build_segmentation_model(num_classes: int = 4):
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )


# ── Landmark decoding ─────────────────────────────────────────────────────────
def _hard_argmax_decode(heatmaps: torch.Tensor, input_size):
    sig = torch.sigmoid(heatmaps).cpu().numpy()
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


# ── Sliding window inference ─────────────────────────────────────────────────
def _gaussian_weight_2d(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size, dtype=np.float32)
    ax = np.abs(ax - (size - 1) / 2.0)
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    return (gauss_1d[:, None] * gauss_1d[None, :]).astype(np.float32)


def _sliding_segmentation_inference(
    seg_model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    H, W = image.shape[:2]
    patch_size = SLIDING_WINDOW_SIZE
    stride = SLIDING_STRIDE
    base_weight = _gaussian_weight_2d(patch_size, sigma=_SLIDING_SIGMA)

    acc_logit = np.zeros((4, H, W), dtype=np.float64)
    acc_weight = np.zeros((H, W), dtype=np.float64)

    half = patch_size // 2
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="reflect")

    with torch.no_grad():
        for r in range(0, H, stride):
            for c in range(0, W, stride):
                r0 = min(r, H - 1)
                c0 = min(c, W - 1)
                r1 = r0 + patch_size
                c1 = c0 + patch_size

                patch = padded[r0:r1, c0:c1]
                patch_f = torch.from_numpy(patch).float().permute(2, 0, 1)
                patch_f = patch_f / 255.0
                tensor = patch_f.unsqueeze(0).to(device)

                logits = seg_model(tensor).cpu().numpy()[0]

                rh0, rh1 = r, min(r + patch_size, H)
                ch0, ch1 = c, min(c + patch_size, W)
                w_vis = base_weight[:rh1 - rh0, :ch1 - ch0]
                for cls in range(4):
                    acc_logit[cls, rh0:rh1, ch0:ch1] += (logits[cls, :rh1 - rh0, :ch1 - ch0] * w_vis)
                acc_weight[rh0:rh1, ch0:ch1] += w_vis

    acc_weight = np.maximum(acc_weight, 1e-8)
    for cls in range(4):
        acc_logit[cls] /= acc_weight

    return torch.from_numpy(acc_logit[np.newaxis].astype(np.float32))


# ── Geometric snapping (mirrors analysis_service.py) ───────────────────────
CLASS_UPPER_INCISOR = 1
CLASS_LABIAL_BONE   = 2
CLASS_PALATAL_BONE  = 3


def _contour_from_mask(mask: np.ndarray, epsilon_factor: float = 0.002):
    if mask.sum() == 0:
        return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(biggest, closed=True)
    if perimeter < 1.0:
        return biggest
    return cv2.approxPolyDP(biggest, epsilon_factor * perimeter, closed=True)


def _project_point_onto_contour(pt: np.ndarray, contour: np.ndarray):
    if contour is None or len(contour) == 0:
        return pt.copy()
    pts = contour.reshape(-1, 2).astype(np.float64)
    dists = np.linalg.norm(pts - pt.astype(np.float64), axis=1)
    return pts[int(dists.argmin())].astype(np.float32)


def geometric_snap(coords: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    """Full snapping: crest → midroot → ans/pns. coords [10,2] in native px."""
    snapped = coords.copy()

    labial_contour  = _contour_from_mask((class_map == CLASS_LABIAL_BONE).astype(np.uint8))
    palatal_contour = _contour_from_mask((class_map == CLASS_PALATAL_BONE).astype(np.uint8))
    incisor_contour = _contour_from_mask((class_map == CLASS_UPPER_INCISOR).astype(np.uint8))

    # Crest points: idx 3=Labial_crest, 5=Palatal_crest
    for idx, contour in [(3, labial_contour), (5, palatal_contour)]:
        pt_raw = snapped[idx]
        if contour is not None:
            cpts = contour.reshape(-1, 2).astype(np.float64)
            y_center = pt_raw[1]
            candidates = cpts[np.abs(cpts[:, 1] - y_center) < 60.0]
            if len(candidates) > 0:
                snapped[idx] = candidates[candidates[:, 1].argmin()]
            else:
                snapped[idx] = _project_point_onto_contour(pt_raw, contour)

    # Midroot: idx 2=Labial_midroot, 4=Palatal_midroot
    if incisor_contour is not None:
        pts = incisor_contour.reshape(-1, 2).astype(np.float64)
        snapped[2] = pts[pts[:, 0].argmax()]   # max x = labial surface
        snapped[4] = pts[pts[:, 0].argmin()]   # min x = palatal surface

    # ANS(6), PNS(7)
    for idx in [6, 7]:
        pt_raw = snapped[idx]
        if palatal_contour is not None:
            snapped[idx] = _project_point_onto_contour(pt_raw, palatal_contour)

    return snapped


# ── Data loading helpers ──────────────────────────────────────────────────────
def load_calibration():
    calib = {}
    with open(CALIB_CSV) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 10:
                calib[parts[0].strip()] = float(parts[9].strip())
    return calib


def load_landmarks():
    return json.loads(LANDMARKS_JSON.read_text())


def patient_split_landmarks():
    records = load_landmarks()
    by_patient = {}
    for rec in records:
        pid = rec.get("patient_id", "unknown")
        by_patient.setdefault(pid, []).append(rec)

    patient_ids = sorted(by_patient.keys())
    n_val = max(1, int(len(patient_ids) * 0.2))
    val_patients = set(patient_ids[-n_val:])
    val, train = [], []
    for rec in records:
        (val if rec.get("patient_id") in val_patients else train).append(rec)
    return val, train


# ── Per-image inference ─────────────────────────────────────────────────────
def preprocess_for_landmarks(img_rgb: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(img_rgb, (INPUT_SIZE[1], INPUT_SIZE[0]))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)


def predict_image(lm_model, seg_model, image_path, mmpp, device):
    """
    Returns (raw_mm, snapped_mm) — both [10, 2] in mm.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    scale_x = orig_w / INPUT_SIZE[1]
    scale_y = orig_h / INPUT_SIZE[0]

    # ─ Landmark: HRNet-W32 at 512×512 ─
    inp = preprocess_for_landmarks(img).to(device)
    with torch.no_grad():
        heatmaps = lm_model(inp)
        if heatmaps.shape[-2:] != HEATMAP_SIZE:
            heatmaps = F.interpolate(heatmaps, size=HEATMAP_SIZE, mode="bilinear", align_corners=False)

    raw_512, _ = _hard_argmax_decode(heatmaps.cpu(), INPUT_SIZE)

    # Scale 512 → native px
    raw_px = raw_512.copy()
    raw_px[:, 0] = raw_512[:, 0] * scale_x
    raw_px[:, 1] = raw_512[:, 1] * scale_y

    raw_mm = raw_px * mmpp

    # ─ Segmentation: sliding window ───
    img_clahe = _apply_clahe(img)
    if orig_h > INPUT_SIZE[0] or orig_w > INPUT_SIZE[1]:
        logits = _sliding_segmentation_inference(seg_model, img_clahe, device)
    else:
        inp_seg = preprocess_for_landmarks(img_clahe).to(device)
        with torch.no_grad():
            logits = seg_model(inp_seg)

    class_map = torch.argmax(logits, dim=1).cpu()[0].numpy().astype(np.uint8)

    # ─ Geometric snapping — only for crest/midroot where mask is reliable ─
    # If contour extraction fails, fall back to raw HRNet prediction to avoid 100+mm errors
    try:
        snapped_px = geometric_snap(raw_px.copy(), class_map)
        # Sanity clamp: any snapped point > 50mm from raw is likely a contour error → keep raw
        for i in range(len(snapped_px)):
            dist = math.sqrt((snapped_px[i,0]-raw_px[i,0])**2 + (snapped_px[i,1]-raw_px[i,1])**2)
            if dist > 50.0:
                snapped_px[i] = raw_px[i]
    except Exception:
        snapped_px = raw_px.copy()

    snapped_mm = snapped_px * mmpp

    return raw_mm, snapped_mm


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_mre(pred_mm, gt_mm):
    errs = [math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2) for p, g in zip(pred_mm, gt_mm)]
    return float(np.mean(errs)), errs


def compute_sdr(pred_mm, gt_mm, threshold_mm=2.0):
    correct = sum(
        1 for p, g in zip(pred_mm, gt_mm)
        if math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2) <= threshold_mm
    )
    return correct / len(pred_mm) * 100


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TSK-05 FINAL EVALUATION")
    print("Model: TSK-04 Tversky+BoundaryDice (Dice=0.8827)")
    print("Inference: Sliding Window (512px, stride 256) + Geom Snapping")
    print("=" * 60)

    # Landmark model
    print("\n[1] Loading HRNet-W32 landmark model ...")
    lm_model = _build_landmark_model()
    lm_state = torch.load(LM_CKPT, map_location=DEVICE, weights_only=False)
    lm_state = {
        k: v for k, v in lm_state.get("model_state_dict", lm_state).items()
        if "uncertainty" not in k
    }
    lm_model.load_state_dict(lm_state, strict=False)
    lm_model = lm_model.to(DEVICE)
    lm_model.eval()
    print("  OK")

    # Segmentation model (TSK-04)
    print("\n[2] Loading TSK-04 segmentation model ...")
    seg_model = _build_segmentation_model(4)
    seg_state = torch.load(TSK04_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    seg_state = {k.replace("module.", ""): v for k, v in seg_state.items()}
    seg_model.load_state_dict(seg_state, strict=True)
    seg_model = seg_model.to(DEVICE)
    seg_model.eval()
    print("  OK")

    calib = load_calibration()
    val_records, _ = patient_split_landmarks()
    print(f"\n[3] Val set: {len(val_records)} images")

    pred_raw_mres, pred_snap_mres = [], []
    sdr_raw_list, sdr_snap_list = [], []
    per_kp_raw = {n: [] for n in KEYPOINT_NAMES}
    per_kp_snap = {n: [] for n in KEYPOINT_NAMES}

    processed = 0
    skipped = 0

    for rec in val_records:
        img_id = rec.get("image_id", "")
        filename = rec.get("filename", "")
        if not filename:
            skipped += 1
            continue
        image_path = IMAGES_DIR / filename
        if not image_path.exists():
            skipped += 1
            continue

        mmpp = calib.get(img_id, 1.0)
        kps = rec.get("keypoints", [])
        if not kps or len(kps) < 8:
            skipped += 1
            continue

        gt_dict = {kp["name"]: kp for kp in kps if kp.get("visible", True)}
        if len(gt_dict) < 8:
            skipped += 1
            continue

        # Build GT [10, 2] in mm
        gt_mm = np.array([
            [gt_dict[n]["x"] * mmpp, gt_dict[n]["y"] * mmpp]
            if n in gt_dict else [0.0, 0.0]
            for n in KEYPOINT_NAMES
        ], dtype=np.float32)

        result = predict_image(lm_model, seg_model, image_path, mmpp, DEVICE)
        if result is None:
            skipped += 1
            continue

        raw_mm, snapped_mm = result
        if raw_mm.sum() < 1e-6:
            skipped += 1
            continue

        raw_errs = [math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2)
                    for p, g in zip(raw_mm, gt_mm)]
        snap_errs = [math.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2)
                     for p, g in zip(snapped_mm, gt_mm)]

        for i, name in enumerate(KEYPOINT_NAMES):
            per_kp_raw[name].append(raw_errs[i])
            per_kp_snap[name].append(snap_errs[i])

        pred_raw_mres.append(float(np.mean(raw_errs)))
        pred_snap_mres.append(float(np.mean(snap_errs)))
        sdr_raw_list.append(compute_sdr(raw_mm, gt_mm))
        sdr_snap_list.append(compute_sdr(snapped_mm, gt_mm))

        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed} | raw MRE={np.mean(pred_raw_mres):.3f}mm  "
                  f"snap MRE={np.mean(pred_snap_mres):.3f}mm")

    # Summary
    overall_raw_mre = float(np.mean(pred_raw_mres)) if pred_raw_mres else float("inf")
    overall_snap_mre = float(np.mean(pred_snap_mres)) if pred_snap_mres else float("inf")
    overall_raw_sdr  = float(np.mean(sdr_raw_list)) if sdr_raw_list else 0.0
    overall_snap_sdr = float(np.mean(sdr_snap_list)) if sdr_snap_list else 0.0

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS — {processed} images, {skipped} skipped")
    print(f"{'=' * 60}")
    print(f"{'Metric':<20} {'Raw':>12} {'Snapped':>12}")
    print(f"{'MRE (mm)':<20} {overall_raw_mre:>12.3f} {overall_snap_mre:>12.3f}")
    print(f"{'SDR@2mm (%)':<20} {overall_raw_sdr:>12.1f} {overall_snap_sdr:>12.1f}")
    print(f"\nPer-landmark:")
    print(f"{'Name':<20} {'Raw MRE':>10} {'Snap MRE':>10} {'Raw SDR%':>10} {'Snap SDR%':>10}")
    print("-" * 65)
    for name in KEYPOINT_NAMES:
        raw_e = per_kp_raw[name]
        snap_e = per_kp_snap[name]
        r_mre = float(np.mean(raw_e)) if raw_e else float("nan")
        s_mre = float(np.mean(snap_e)) if snap_e else float("nan")
        r_sdr = sum(1 for e in raw_e if e <= 2.0) / len(raw_e) * 100 if raw_e else 0.0
        s_sdr = sum(1 for e in snap_e if e <= 2.0) / len(snap_e) * 100 if snap_e else 0.0
        print(f"{name:<20} {r_mre:>9.3f} {s_mre:>9.3f} {r_sdr:>9.1f} {s_sdr:>9.1f}")

    report = {
        "task": "TSK-05 Final Evaluation",
        "model": "TSK-04 Tversky+BoundaryDice (Dice=0.8827)",
        "inference": "Sliding Window 512px/256stride + Geometric Snapping + HRNet-W32",
        "timestamp": datetime.now().isoformat(),
        "n_val_images": processed,
        "n_skipped": skipped,
        "overall_raw_mre_mm": round(overall_raw_mre, 3),
        "overall_snap_mre_mm": round(overall_snap_mre, 3),
        "overall_raw_sdr_pct": round(overall_raw_sdr, 1),
        "overall_snap_sdr_pct": round(overall_snap_sdr, 1),
        "per_landmark": {
            n: {
                "raw_mre_mm": round(float(np.mean(per_kp_raw[n])), 3) if per_kp_raw[n] else None,
                "snap_mre_mm": round(float(np.mean(per_kp_snap[n])), 3) if per_kp_snap[n] else None,
                "raw_sdr_pct": round(sum(1 for e in per_kp_raw[n] if e <= 2.0) / len(per_kp_raw[n]) * 100, 1) if per_kp_raw[n] else None,
                "snap_sdr_pct": round(sum(1 for e in per_kp_snap[n] if e <= 2.0) / len(per_kp_snap[n]) * 100, 1) if per_kp_snap[n] else None,
            }
            for n in KEYPOINT_NAMES
        }
    }
    report_path = OUTPUT_DIR / f"tsk05_final_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")
    return report


if __name__ == "__main__":
    main()
