"""Compute mm_per_pixel from Calibration_30mm polyline."""

import math
from typing import Optional


CALIBRATION_MM = 30.0


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def compute_mm_per_pixel(
    p1: tuple[float, float],
    p2: tuple[float, float],
    ruler_mm: float = CALIBRATION_MM,
) -> float:
    """
    Compute mm_per_pixel from two endpoints of the calibration ruler.
    Ruler is nearly vertical (Δx small, Δy large) in all clinic images.
    """
    dist_px = euclidean_distance(p1, p2)
    if dist_px == 0:
        raise ValueError(f"Calibration polyline has zero length: {p1}, {p2}")
    return ruler_mm / dist_px


def calibration_from_record(record: dict, ruler_mm: float = CALIBRATION_MM) -> Optional[float]:
    """Extract mm_per_pixel from a parsed image record. Returns None if not available."""
    pts = record.get("calibration_points")
    if not pts or len(pts) < 2:
        return None
    return compute_mm_per_pixel(pts[0], pts[1], ruler_mm)
