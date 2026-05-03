"""Write Phase 1 outputs: landmarks_clean.json and calibration.csv."""

import csv
from pathlib import Path

from src.phase1.calibration import calibration_from_record
from src.utils.io import save_json


QUALITY_REJECT_TAG = "Quality_Reject"


def build_landmarks_json(records: list[dict], reject_bad: bool = True) -> dict:
    """
    Filter records and build the landmarks_clean.json structure.
    Excludes Quality_Reject images when reject_bad=True.
    """
    images = []
    for rec in records:
        if reject_bad and QUALITY_REJECT_TAG in rec.get("quality_flags", []):
            continue
        # Drop calibration_points from output (stored separately in CSV)
        entry = {k: v for k, v in rec.items() if k != "calibration_points"}
        images.append(entry)
    return {"images": images}


def build_calibration_rows(records: list[dict]) -> list[dict]:
    """Build rows for calibration.csv — one row per image, including rejected ones."""
    rows = []
    for rec in records:
        pts = rec.get("calibration_points")
        mm_per_pixel = calibration_from_record(rec)
        pt1 = pts[0] if pts else (None, None)
        pt2 = pts[1] if pts else (None, None)

        dist_px = None
        if pts:
            import math
            dist_px = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

        rows.append({
            "image_id": rec["image_id"],
            "file_name": rec["file_name"],
            "patient_id": rec["patient_id"],
            "timepoint": rec["timepoint"],
            "pt1_x": pt1[0],
            "pt1_y": pt1[1],
            "pt2_x": pt2[0],
            "pt2_y": pt2[1],
            "distance_px": dist_px,
            "mm_per_pixel": mm_per_pixel,
        })
    return rows


def write_calibration_csv(rows: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "file_name", "patient_id", "timepoint",
                  "pt1_x", "pt1_y", "pt2_x", "pt2_y", "distance_px", "mm_per_pixel"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_export(records: list[dict], landmarks_json_path: str, calibration_csv_path: str) -> dict:
    """Run Phase 1 export. Returns summary stats dict."""
    landmarks_data = build_landmarks_json(records, reject_bad=True)
    save_json(landmarks_data, landmarks_json_path)

    cal_rows = build_calibration_rows(records)
    write_calibration_csv(cal_rows, calibration_csv_path)

    n_total = len(records)
    n_rejected = sum(1 for r in records if QUALITY_REJECT_TAG in r.get("quality_flags", []))
    n_with_landmarks = sum(1 for r in records if r.get("has_landmarks"))
    n_with_calibration = sum(1 for r in records if r.get("has_calibration"))

    return {
        "total_images": n_total,
        "rejected": n_rejected,
        "exported": n_total - n_rejected,
        "with_landmarks": n_with_landmarks,
        "with_calibration": n_with_calibration,
    }
