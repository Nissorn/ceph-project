"""Parse CVAT XML 1.1 annotation exports for the cephalometric pipeline.

Designed to be re-runnable: handles calibration-only exports now and will
pick up skeleton landmarks automatically when Dr. completes annotations.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Hardcoded order — never infer from file content
KEYPOINT_NAMES: list[str] = [
    "Upper_tip",        # 0 — crown tip
    "Upper_apex",       # 1 — root apex
    "Labial_midroot",   # 2
    "Labial_crest",     # 3
    "Palatal_midroot",  # 4
    "Palatal_crest",    # 5
    "ANS",              # 6 — superimposition reference
    "PNS",              # 7 — superimposition reference
    "LB",               # 8 — labial bone level (annotated in CVAT v2+)
    "PB",               # 9 — palatal bone level (annotated in CVAT v2+)
]

CALIBRATION_LABEL = "Calibration_30mm"
SKELETON_LABEL = "Incisor_Maxilla_Complex"
TREATMENT_TAGS = frozenset({
    "Uncontrolled_tipping",
    "Controlled_tipping",
    "Translation",
    "Root_torque",
    "Extrusion",
    "Intrusion",
})
QUALITY_TAGS = frozenset({"Quality_Reject", "Low_Visibility"})


def _parse_points(points_str: str) -> list[tuple[float, float]]:
    """'x1,y1;x2,y2;...' → [(x1, y1), (x2, y2), ...]"""
    result: list[tuple[float, float]] = []
    for pair in points_str.strip().split(";"):
        x_str, y_str = pair.split(",")
        result.append((float(x_str), float(y_str)))
    return result


def _split_patient_timepoint(filename: str) -> tuple[str, str]:
    """'Patient03_T1.jpg' → ('Patient03', 'T1')"""
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("T1", "T2"):
        return parts[0], parts[1]
    log.warning("Cannot parse patient/timepoint from filename: %s", filename)
    return stem, "unknown"


def parse_cvat_xml(xml_path: str | Path) -> list[dict]:
    """
    Parse a CVAT XML 1.1 export and return one record per image.

    Record schema:
        image_id        str    'Patient03_T1'
        filename        str    'Patient03_T1.jpg'
        patient_id      str    'Patient03'
        timepoint       str    'T1' | 'T2' | 'unknown'
        width           int
        height          int
        calibration_pts Optional[list[tuple[float,float]]]  — [(x1,y1),(x2,y2)] or None
        has_calibration bool
        keypoints       list[dict]   [{"name":str,"x":float,"y":float,"visible":bool}]
        valid_mask      list[int]    1=present, 0=missing — same order as KEYPOINT_NAMES
        has_landmarks   bool
        polygons        dict[str, list[list[float]]]
        treatment       list[str]
        quality_flags   list[str]
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"CVAT XML not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()
    records: list[dict] = []

    for img_el in root.findall(".//image"):
        filename = img_el.get("name", "")
        patient_id, timepoint = _split_patient_timepoint(filename)
        image_id = Path(filename).stem

        record: dict = {
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
                    if len(pts) != 2:
                        log.warning(
                            "%s: calibration polyline has %d points (expected 2) — skipping",
                            filename, len(pts),
                        )
                    else:
                        record["calibration_pts"] = pts
                        record["has_calibration"] = True
                except Exception as exc:
                    log.warning("%s: malformed calibration points: %s", filename, exc)

            elif tag == "skeleton" and label == SKELETON_LABEL:
                kp_map: dict[str, tuple[float, float]] = {}
                for pt_el in child.findall("points"):
                    kp_label = pt_el.get("label", "")
                    try:
                        pts = _parse_points(pt_el.get("points", ""))
                        if pts:
                            kp_map[kp_label] = pts[0]
                    except Exception as exc:
                        log.warning("%s: malformed keypoint '%s': %s", filename, kp_label, exc)

                keypoints: list[dict] = []
                valid_mask: list[int] = []
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

            elif tag == "polygon" and label in ("Upper_incisor", "Labial_bone", "Palatal_bone"):
                try:
                    pts = _parse_points(points_str)
                    record["polygons"][label] = [[x, y] for x, y in pts]
                except Exception as exc:
                    log.warning("%s: malformed polygon '%s': %s", filename, label, exc)

            elif tag == "tag":
                if label in TREATMENT_TAGS:
                    record["treatment"].append(label)
                elif label in QUALITY_TAGS:
                    record["quality_flags"].append(label)

        records.append(record)

    return records
