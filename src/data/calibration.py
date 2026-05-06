"""Compute mm_per_pixel from Calibration_30mm polyline endpoints."""

from __future__ import annotations

import math

CALIBRATION_MM: float = 30.0


def pixel_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def compute_mm_per_pixel(
    p1: tuple[float, float],
    p2: tuple[float, float],
    ruler_mm: float = CALIBRATION_MM,
) -> float:
    """ruler_mm / pixel_distance(p1, p2) — raises ValueError on zero-length."""
    dist = pixel_distance(p1, p2)
    if dist == 0.0:
        raise ValueError(f"Zero-length calibration polyline: {p1} → {p2}")
    return ruler_mm / dist


def calibration_row(record: dict, ruler_mm: float = CALIBRATION_MM) -> dict:
    """
    Build one row for calibration.csv from a parsed record.
    Fields: image_id, filename, patient_id, timepoint,
            pt1_x, pt1_y, pt2_x, pt2_y, pixel_length, mm_per_pixel.
    All numeric fields are None when calibration data is absent or invalid.
    """
    pts = record.get("calibration_pts")
    base = {
        "image_id": record["image_id"],
        "filename": record["filename"],
        "patient_id": record["patient_id"],
        "timepoint": record["timepoint"],
    }

    if not pts or len(pts) < 2:
        return {**base,
                "pt1_x": None, "pt1_y": None,
                "pt2_x": None, "pt2_y": None,
                "pixel_length": None,
                "mm_per_pixel": None}

    p1, p2 = pts[0], pts[1]
    try:
        dist = pixel_distance(p1, p2)
        mpp = compute_mm_per_pixel(p1, p2, ruler_mm)
    except ValueError:
        dist = 0.0
        mpp = None

    return {**base,
            "pt1_x": p1[0], "pt1_y": p1[1],
            "pt2_x": p2[0], "pt2_y": p2[1],
            "pixel_length": dist,
            "mm_per_pixel": mpp}


def build_calibration_rows(records: list[dict], ruler_mm: float = CALIBRATION_MM) -> list[dict]:
    return [calibration_row(r, ruler_mm) for r in records]
