#!/usr/bin/env python3
"""Impeccable extraction and ingestion script for CephCanvasEditor JSON exports.

Parses manual clinical annotations exported from the React Konva editor
and seamlessly merges them into the authoritative landmarks_clean.json dataset,
preserving strict keypoint ordering, patient-aware structures, and calibration references.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Hardcoded keypoint order matching repository rules
KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS",
    "LB", "PB",
]

def split_patient_timepoint(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("T1", "T2"):
        return parts[0], parts[1]
    return stem, "unknown"

def main():
    parser = argparse.ArgumentParser(description="Ingest CephCanvasEditor JSON exports into the main dataset.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a specific exported JSON file or a directory containing exported JSON files.")
    parser.add_argument("--clean-json", type=str, default="data/processed/landmarks_clean.json",
                        help="Path to the standardized landmarks_clean.json file to update.")
    parser.add_argument("--calibration-csv", type=str, default="data/processed/calibration.csv",
                        help="Path to calibration.csv to look up calibration points.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    clean_json_path = project_root / args.clean_json if not Path(args.clean_json).is_absolute() else Path(args.clean_json)
    calib_csv_path = project_root / args.calibration_csv if not Path(args.calibration_csv).is_absolute() else Path(args.calibration_csv)

    # 1. Load existing clean records to allow merging/updating
    existing_records: list[dict] = []
    if clean_json_path.exists():
        try:
            with open(clean_json_path, "r", encoding="utf-8") as f:
                existing_records = json.load(f)
            log.info("Loaded %d existing records from %s", len(existing_records), clean_json_path.relative_to(project_root))
        except Exception as e:
            log.warning("Could not parse existing clean JSON (%s), starting fresh.", e)

    records_map = {r["image_id"]: r for r in existing_records if "image_id" in r}

    # 2. Load calibration DataFrame if available
    calib_df = None
    if calib_csv_path.exists():
        try:
            calib_df = pd.read_csv(calib_csv_path, index_col="image_id")
            log.info("Loaded calibration data for %d images.", len(calib_df))
        except Exception as e:
            log.warning("Could not load calibration CSV: %s", e)

    # 3. Gather target files to process
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        files_to_process.extend(input_path.glob("*.json"))
    else:
        log.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if not files_to_process:
        log.warning("No JSON export files found to process at %s", input_path)
        return

    log.info("Processing %d exported annotation file(s)...", len(files_to_process))
    updated_count = 0
    added_count = 0

    for file_path in files_to_process:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                export_data = json.load(f)
        except Exception as e:
            log.error("Failed to read JSON file %s: %s", file_path, e)
            continue

        filename = export_data.get("imageName")
        if not filename:
            log.warning("Skipping %s: missing 'imageName' attribute.", file_path.name)
            continue

        image_id = Path(filename).stem
        patient_id, timepoint = split_patient_timepoint(filename)
        width = export_data.get("imageWidth", 0)
        height = export_data.get("imageHeight", 0)

        # Parse keypoints into standard dict mapping
        kp_export_list = export_data.get("keypoints", [])
        kp_export_map = {k.get("name"): (k.get("x"), k.get("y")) for k in kp_export_list if isinstance(k, dict)}

        std_keypoints = []
        valid_mask = []
        for name in KEYPOINT_NAMES:
            if name in kp_export_map and kp_export_map[name][0] is not None and kp_export_map[name][1] is not None:
                x, y = kp_export_map[name]
                std_keypoints.append({"name": name, "x": float(x), "y": float(y), "visible": True})
                valid_mask.append(1)
            else:
                std_keypoints.append({"name": name, "x": 0.0, "y": 0.0, "visible": False})
                valid_mask.append(0)

        has_landmarks = any(v == 1 for v in valid_mask)

        # Parse polygons
        poly_export_list = export_data.get("polygons", [])
        std_polygons = {}
        for p in poly_export_list:
            if not isinstance(p, dict):
                continue
            p_name = p.get("name")
            p_pts = p.get("points", [])
            if p_name and isinstance(p_pts, list) and len(p_pts) >= 6:
                # Group flat array into pairs
                paired_pts = [[float(p_pts[i]), float(p_pts[i+1])] for i in range(0, len(p_pts)-1, 2)]
                std_polygons[p_name] = paired_pts

        # Merge into existing record or create new
        if image_id in records_map:
            rec = records_map[image_id]
            rec["width"] = width
            rec["height"] = height
            rec["keypoints"] = std_keypoints
            rec["valid_mask"] = valid_mask
            rec["has_landmarks"] = has_landmarks
            rec["polygons"] = std_polygons
            updated_count += 1
            log.info("Updated existing record for %s", image_id)
        else:
            rec = {
                "image_id": image_id,
                "filename": filename,
                "patient_id": patient_id,
                "timepoint": timepoint,
                "width": width,
                "height": height,
                "calibration_pts": None,
                "has_calibration": False,
                "keypoints": std_keypoints,
                "valid_mask": valid_mask,
                "has_landmarks": has_landmarks,
                "polygons": std_polygons,
                "treatment": [],
                "quality_flags": [],
            }
            records_map[image_id] = rec
            existing_records.append(rec)
            added_count += 1
            log.info("Added new record for %s", image_id)

        # Re-verify calibration data from CSV if present
        if calib_df is not None and image_id in calib_df.index:
            try:
                row = calib_df.loc[image_id]
                # In case of duplicate index rows, take the first one
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                rec["calibration_pts"] = [[float(row["pt1_x"]), float(row["pt1_y"])], [float(row["pt2_x"]), float(row["pt2_y"])]]
                rec["has_calibration"] = True
            except Exception as e:
                log.debug("Could not attach calibration points for %s: %s", image_id, e)

    # Save final cleanly combined dataset
    clean_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clean_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_records, f, indent=2)

    log.info("\n✨ Impeccable Extraction Complete!")
    log.info("   - Total records in clean JSON: %d", len(existing_records))
    log.info("   - Records updated: %d", updated_count)
    log.info("   - Records added:   %d", added_count)
    log.info("💾 Saved to: %s", clean_json_path.resolve())

if __name__ == "__main__":
    main()
