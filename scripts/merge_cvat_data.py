#!/usr/bin/env python3
"""
Merge CVAT annotation batches (batch01–batch04) into unified JSON files.

Outputs:
  data/processed/landmarks_clean.json   — all records (keypoints + polygons + tags)
  data/processed/segmentation_train.json — records that have ≥1 polygon annotation

The parser is schema-compatible with src/data/cvat_parser.py and
src/phase2b/segmentation_dataset.py so no changes needed downstream.

Usage:
    python scripts/merge_cvat_data.py
"""

import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("merge_cvat_data")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "raw" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCHES = [
    ANNOTATION_DIR / "annotations_batch01.xml",
    ANNOTATION_DIR / "annotations_batch02.xml",
    ANNOTATION_DIR / "annotations_batch03.xml",
    ANNOTATION_DIR / "annotations_batch04.xml",
]

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]

CALIBRATION_LABEL = "Calibration_30mm"
SKELETON_LABEL = "Incisor_Maxilla_Complex"
TREATMENT_TAGS = frozenset({
    "Uncontrolled_tipping", "Controlled_tipping", "Translation",
    "Root_torque", "Extrusion", "Intrusion",
})
QUALITY_TAGS = frozenset({"Quality_Reject", "Low_Visibility"})
POLYGON_LABELS = {"Upper_incisor", "Labial_bone", "Palatal_bone"}


def _parse_points(points_str: str) -> list[tuple[float, float]]:
    result = []
    for pair in points_str.strip().split(";"):
        x_str, y_str = pair.split(",")
        result.append((float(x_str), float(y_str)))
    return result


def _split_patient_timepoint(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("T1", "T2"):
        return parts[0], parts[1]
    return stem, "unknown"


def parse_cvat_xml(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []

    for img_el in root.findall(".//image"):
        filename = img_el.get("name", "")
        patient_id, timepoint = _split_patient_timepoint(filename)
        image_id = Path(filename).stem

        record = {
            "image_id": image_id,
            "filename": filename,
            "patient_id": patient_id,
            "timepoint": timepoint,
            "width": int(img_el.get("width", 0)),
            "height": int(img_el.get("height", 0)),
            "calibration_pts": None,
            "has_calibration": False,
            "keypoints": [],
            "valid_mask": [0] * len(KEYPOINT_NAMES),
            "has_landmarks": False,
            "polygons": {},
            "treatment": [],
            "quality_flags": [],
        }

        for child in img_el:
            tag = child.tag
            label = child.get("label", "")
            points_str = child.get("points", "")

            if tag == "polyline" and label == CALIBRATION_LABEL:
                try:
                    pts = _parse_points(points_str)
                    if len(pts) == 2:
                        record["calibration_pts"] = pts
                        record["has_calibration"] = True
                except Exception as exc:
                    log.warning("%s: malformed calibration: %s", filename, exc)

            elif tag == "skeleton" and label == SKELETON_LABEL:
                kp_map = {}
                for pt_el in child.findall("points"):
                    kp_label = pt_el.get("label", "")
                    try:
                        pts = _parse_points(pt_el.get("points", ""))
                        if pts:
                            kp_map[kp_label] = pts[0]
                    except Exception as exc:
                        log.warning("%s: malformed keypoint '%s': %s", filename, kp_label, exc)

                keypoints = []
                valid_mask = []
                for name in KEYPOINT_NAMES:
                    if name in kp_map:
                        x, y = kp_map[name]
                        keypoints.append({"name": name, "x": x, "y": y, "visible": True})
                        valid_mask.append(1)
                    else:
                        keypoints.append({"name": name, "x": 0.0, "y": 0.0, "visible": False})
                        valid_mask.append(0)

                record["keypoints"] = keypoints
                record["valid_mask"] = valid_mask
                record["has_landmarks"] = any(v == 1 for v in valid_mask)

            elif tag == "polygon" and label in POLYGON_LABELS:
                try:
                    pts = _parse_points(points_str)
                    record["polygons"][label] = [[float(x), float(y)] for x, y in pts]
                except Exception as exc:
                    log.warning("%s: malformed polygon '%s': %s", filename, label, exc)

            elif tag == "tag":
                if label in TREATMENT_TAGS:
                    record["treatment"].append(label)
                elif label in QUALITY_TAGS:
                    record["quality_flags"].append(label)

        records.append(record)

    return records


def main():
    print("=" * 62)
    print("CVAT Annotation Batch Merger")
    print("=" * 62)

    all_records: list[dict] = []
    seen_ids: dict[str, int] = {}

    for batch_path in BATCHES:
        if not batch_path.exists():
            log.error("Batch file not found: %s", batch_path)
            continue
        log.info("Parsing %s", batch_path.name)
        records = parse_cvat_xml(batch_path)
        n_new = 0
        for rec in records:
            image_id = rec["image_id"]
            if image_id not in seen_ids:
                seen_ids[image_id] = len(all_records)
                all_records.append(rec)
                n_new += 1
            else:
                # Merge: prefer record with more data
                existing = all_records[seen_ids[image_id]]
                # Take whichever has more keypoints
                if sum(rec.get("valid_mask", [])) > sum(existing.get("valid_mask", [])):
                    all_records[seen_ids[image_id]] = rec
                log.debug("Duplicate image_id %s — kept record with more annotations", image_id)
        log.info("  +%d new records (total now %d)", n_new, len(all_records))

    # ── Verification counts ───────────────────────────────────────────
    n_landmarks  = sum(1 for r in all_records if r["has_landmarks"])
    n_calib      = sum(1 for r in all_records if r["has_calibration"])
    n_polygons   = sum(1 for r in all_records if r["polygons"])
    n_treatment  = sum(1 for r in all_records if r["treatment"])
    n_reject     = sum(1 for r in all_records if "Quality_Reject" in r["quality_flags"])
    n_landmarks_batch01 = sum(1 for r in all_records if r["has_landmarks"])

    # Per-polygon class counts
    poly_counts = defaultdict(int)
    for r in all_records:
        for poly_cls in r["polygons"]:
            poly_counts[poly_cls] += 1

    # Per-tag counts
    tag_counts = defaultdict(int)
    for r in all_records:
        for t in r["treatment"]:
            tag_counts[t] += 1

    # Batch breakdown for landmarks
    batch_landmark_counts = {}
    batch_image_counts = {}
    for batch_path in BATCHES:
        batch_name = batch_path.stem.replace("annotations_", "")
        tree = ET.parse(batch_path)
        root = tree.getroot()
        n_imgs = len(root.findall(".//image"))
        batch_image_counts[batch_name] = n_imgs

    print("\n" + "=" * 62)
    print("MERGE COMPLETE — VERIFIED COUNTS")
    print("=" * 62)
    print(f"  Total unique images        : {len(all_records)}")
    print(f"  With landmark annotations  : {n_landmarks}")
    print(f"  With calibration data      : {n_calib}")
    print(f"  With polygon annotations   : {n_polygons}")
    print(f"  With treatment tags         : {n_treatment}")
    print(f"  Quality_Reject flagged     : {n_reject}")
    print()
    print("  Per-polygon class:")
    for cls in ["Upper_incisor", "Labial_bone", "Palatal_bone"]:
        print(f"    {cls:<20s}: {poly_counts[cls]}")
    print()
    print("  Per-treatment tag:")
    for tag, cnt in sorted(tag_counts.items()):
        print(f"    {tag:<25s}: {cnt}")
    print()
    print("  Per-batch image counts:")
    for bname, bcnt in batch_image_counts.items():
        print(f"    {bname:<25s}: {bcnt}")

    # ── Write output files ─────────────────────────────────────────────
    landmarks_path = OUTPUT_DIR / "landmarks_clean.json"
    with open(landmarks_path, "w") as f:
        json.dump(all_records, f, indent=2)
    log.info("Written: %s  (%d records)", landmarks_path, len(all_records))

    seg_records = [r for r in all_records if r["polygons"]]
    seg_path = OUTPUT_DIR / "segmentation_train.json"
    with open(seg_path, "w") as f:
        json.dump(seg_records, f, indent=2)
    log.info("Written: %s  (%d records with polygons)", seg_path, len(seg_records))

    print()
    print("  OUTPUT FILES:")
    print(f"    landmark_train.json      : {landmarks_path}  ({len(all_records)} records)")
    print(f"    segmentation_train.json : {seg_path}  ({len(seg_records)} records)")
    print()
    print("✓ Merge complete.")


if __name__ == "__main__":
    main()