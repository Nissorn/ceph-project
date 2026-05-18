#!/usr/bin/env python3
"""
evaluate_tta.py — Compare TTA predictions vs baseline (no-TTA) against ground truth.
Loads GT from data/processed/landmarks_clean.json and calibration from calibration.csv.
Usage:
    python scripts/evaluate_tta.py
"""

import json, csv, sys, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]


def load_calibration(csv_path: Path) -> dict[str, float]:
    """image_id → mm_per_pixel"""
    mm_map = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mm_map[row["image_id"]] = float(row["mm_per_pixel"])
    return mm_map


def load_gt_landmarks(json_path: Path) -> dict[str, dict]:
    """
    Load GT landmarks. Structure: {"images": [{image_id, filename, keypoints: [...]}, ...]}
    Returns: {img_id: {landmark_name: (x, y), ...}, ...}
    """
    with open(json_path) as f:
        data = json.load(f)
    result = {}
    for img_entry in data["images"]:
        img_id = img_entry["image_id"]   # e.g. "Patient01_T1"
        result[img_id] = {}
        for lm in img_entry["keypoints"]:
            result[img_id][lm["name"]] = (lm["x"], lm["y"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta", default="outputs/predictions_tta.json",
                        help="TTA predictions JSON")
    parser.add_argument("--baseline", default="outputs/predictions.json",
                        help="Baseline (no-TTA) predictions JSON")
    parser.add_argument("--calibration", default="data/processed/calibration.csv")
    parser.add_argument("--landmarks", default="data/processed/landmarks_clean.json")
    args = parser.parse_args()

    # Load data
    mm_map = load_calibration(ROOT / args.calibration)
    gt_images = load_gt_landmarks(ROOT / args.landmarks)
    print(f"[INFO] {len(gt_images)} GT images, {len(mm_map)} calibration entries")

    with open(ROOT / args.tta) as f:
        tta_preds = json.load(f)
    print(f"[INFO] TTA predictions: {len(tta_preds)} images")

    baseline_preds = None
    if Path(args.baseline).exists():
        with open(ROOT / args.baseline) as f:
            baseline_preds = json.load(f)
        print(f"[INFO] Baseline predictions: {len(baseline_preds)} images")

    # Per-image MRE
    print("\n=== Per-Image MRE (mm) ===")
    print(f"{'Image':<22} {'TTA MRE':>10} {'Base MRE':>10} {'Delta':>10}")
    print("-" * 55)

    tta_errors = []
    base_errors = []

    for img_id in sorted(gt_images.keys()):
        gt_kps = gt_images[img_id]  # {name: (x, y), ...}

        # Match by image_id in predictions (predictions keyed by filename)
        tta_img = None
        for k in tta_preds:
            if k.replace(".jpg", "").replace(".png", "") == img_id:
                tta_img = k
                break
        if tta_img is None:
            continue

        tta_kps = tta_preds[tta_img]

        mm_per_px = mm_map.get(img_id, 0.1)
        tta_total = 0.0
        base_total = 0.0
        count = 0

        for name in KEYPOINT_NAMES:
            if name not in gt_kps or name not in tta_kps:
                continue
            gx, gy = gt_kps[name]
            tx, ty = tta_kps[name]["x"], tta_kps[name]["y"]
            err = np.sqrt((tx - gx)**2 + (ty - gy)**2) * mm_per_px
            tta_total += err
            count += 1

            if baseline_preds:
                base_img = None
                for k in baseline_preds:
                    if k.replace(".jpg", "").replace(".png", "") == img_id:
                        base_img = k
                        break
                if base_img and name in baseline_preds[base_img]:
                    bx = baseline_preds[base_img][name]["x"]
                    by = baseline_preds[base_img][name]["y"]
                    berr = np.sqrt((bx - gx)**2 + (by - gy)**2) * mm_per_px
                    base_total += berr
                else:
                    base_total = None

        tta_mre_mm = tta_total / count if count else 0.0
        if count == 0:
            continue
        if base_total is not None and base_total > 0:
            base_mre_mm = base_total / count
            delta = tta_mre_mm - base_mre_mm
            print(f"{img_id:<22} {tta_mre_mm:>10.3f} {base_mre_mm:>10.3f} {delta:>+10.3f}")
            tta_errors.append(tta_mre_mm)
            base_errors.append(base_mre_mm)
        else:
            print(f"{img_id:<22} {tta_mre_mm:>10.3f} {'N/A':>10} {'N/A':>10}")

    if baseline_preds:
        overall_tta = np.mean(tta_errors)
        overall_base = np.mean(base_errors)
        print("-" * 55)
        print(f"{'OVERALL MEAN':<22} {overall_tta:>10.3f} {overall_base:>10.3f} {overall_tta - overall_base:>+10.3f}")
        winner = "TTA IMPROVED" if overall_tta < overall_base else "TTA REGRESSED"
        print(f"\n==> {winner} by {abs(overall_tta - overall_base):.3f}mm")
    else:
        overall_tta = np.mean(tta_errors)
        print("-" * 55)
        print(f"{'OVERALL MEAN':<22} {overall_tta:>10.3f}")

    # Per-landmark breakdown
    print("\n=== Per-Landmark MRE (mm) ===")
    landmark_errors = {name: [] for name in KEYPOINT_NAMES}

    for img_id, gt_kps in gt_images.items():
        # Match by image_id in predictions
        tta_img = None
        for k in tta_preds:
            if k.replace(".jpg", "").replace(".png", "") == img_id:
                tta_img = k
                break
        if tta_img is None:
            continue
        tta_kps = tta_preds[tta_img]
        mm_per_px = mm_map.get(img_id, 0.1)

        for name in KEYPOINT_NAMES:
            if name not in gt_kps or name not in tta_kps:
                continue
            gx, gy = gt_kps[name]
            tx, ty = tta_kps[name]["x"], tta_kps[name]["y"]
            err = np.sqrt((tx - gx)**2 + (ty - gy)**2) * mm_per_px
            landmark_errors[name].append(err)

    print(f"{'Landmark':<20} {'Mean (mm)':>10} {'Min':>8} {'Max':>8} {'Count':>6}")
    print("-" * 55)
    for name, errors in landmark_errors.items():
        if errors:
            print(f"{name:<20} {np.mean(errors):>10.3f} {np.min(errors):>8.3f} {np.max(errors):>8.3f} {len(errors):>6}")


if __name__ == "__main__":
    main()