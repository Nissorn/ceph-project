"""Parse CVAT XML 1.1 annotation files into structured Python dicts."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# Confirmed keypoint order — never infer from file
KEYPOINT_NAMES = [
    "Upper_tip",
    "Upper_apex",
    "Labial_midroot",
    "Labial_crest",
    "Palatal_midroot",
    "Palatal_crest",
    "ANS",
    "PNS",
]

SKELETON_LABEL = "Incisor_Maxilla_Complex_Skeleton"
CALIBRATION_LABEL = "Calibration_30mm"
QUALITY_REJECT_TAG = "Quality_Reject"
LOW_VISIBILITY_TAG = "Low_Visibility"

TREATMENT_TAGS = {
    "Uncontrolled_tipping",
    "Controlled_tipping",
    "Translation",
    "Root_torque",
    "Extrusion",
    "Intrusion",
}


def _parse_points_str(points_str: str) -> list[tuple[float, float]]:
    """'x1,y1;x2,y2;...' → [(x1,y1), (x2,y2), ...]"""
    result = []
    for pair in points_str.strip().split(";"):
        x, y = pair.split(",")
        result.append((float(x), float(y)))
    return result


def _extract_timepoint(image_name: str) -> tuple[str, str]:
    """'Patient03_T1.jpg' → ('Patient03', 'T1')"""
    stem = Path(image_name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("T1", "T2"):
        return parts[0], parts[1]
    return stem, "unknown"


def parse_cvat_xml(xml_path: str) -> list[dict]:
    """
    Parse a CVAT XML 1.1 file.

    Returns a list of image dicts with structure matching landmarks_clean.json spec.
    Images tagged Quality_Reject are included but flagged — filtering happens in export.py.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for img_el in root.findall(".//image"):
        image_id_raw = img_el.get("name", "")
        patient_id, timepoint = _extract_timepoint(image_id_raw)
        image_id = Path(image_id_raw).stem

        record: dict = {
            "image_id": image_id,
            "file_name": image_id_raw,
            "patient_id": patient_id,
            "timepoint": timepoint,
            "width": int(img_el.get("width", 0)),
            "height": int(img_el.get("height", 0)),
            "has_landmarks": False,
            "has_calibration": False,
            "keypoints": [],
            "valid_mask": [0] * len(KEYPOINT_NAMES),
            "calibration_points": None,
            "treatment": [],
            "quality_flags": [],
            "polygons": {},
        }

        # Tags (treatment + quality)
        for tag_el in img_el.findall("tag"):
            label = tag_el.get("label", "")
            if label in TREATMENT_TAGS:
                record["treatment"].append(label)
            elif label in (QUALITY_REJECT_TAG, LOW_VISIBILITY_TAG):
                record["quality_flags"].append(label)

        # Calibration polyline
        for poly_el in img_el.findall("polyline"):
            if poly_el.get("label") == CALIBRATION_LABEL:
                pts = _parse_points_str(poly_el.get("points", ""))
                if len(pts) == 2:
                    record["calibration_points"] = pts
                    record["has_calibration"] = True

        # Skeleton keypoints
        for skel_el in img_el.findall("skeleton"):
            if skel_el.get("label") == SKELETON_LABEL:
                kp_map: dict[str, tuple[float, float]] = {}
                for pt_el in skel_el.findall("points"):
                    kp_label = pt_el.get("label", "")
                    pts = _parse_points_str(pt_el.get("points", ""))
                    if pts:
                        kp_map[kp_label] = pts[0]

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

        # Segmentation polygons
        for poly_el in img_el.findall("polygon"):
            label = poly_el.get("label", "")
            if label in ("Upper_incisor", "Labial_bone", "Palatal_bone"):
                pts = _parse_points_str(poly_el.get("points", ""))
                record["polygons"][label] = [[x, y] for x, y in pts]

        records.append(record)

    return records
