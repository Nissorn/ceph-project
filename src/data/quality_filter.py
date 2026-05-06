"""Quality filter for Phase 1 calibration data.

Splits calibration rows into passing and rejected sets.
Writes calibration_clean.csv (passing) and rejection_log.txt (failures with reason).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

MM_PER_PIXEL_MIN: float = 0.05   # below this → ruler annotation error or extreme zoom
MM_PER_PIXEL_MAX: float = 0.30   # above this → implausible for clinical ceph X-ray

_CALIBRATION_FIELDS = [
    "image_id", "filename", "patient_id", "timepoint",
    "pt1_x", "pt1_y", "pt2_x", "pt2_y", "pixel_length", "mm_per_pixel",
]


def filter_calibration(
    rows: list[dict],
    mpp_min: float = MM_PER_PIXEL_MIN,
    mpp_max: float = MM_PER_PIXEL_MAX,
) -> tuple[list[dict], list[dict]]:
    """
    Split calibration rows into (passing, rejected).

    Rejection criteria (checked in order):
    1. Duplicate image_id — only the first occurrence passes
    2. Missing calibration polyline (mm_per_pixel is None, pt1_x is None)
    3. Zero-length polyline (mm_per_pixel is None, pt1_x is not None)
    4. mm_per_pixel < mpp_min
    5. mm_per_pixel > mpp_max

    Rejected rows include an extra "rejection_reason" key.
    """
    seen_ids: set[str] = set()
    passing: list[dict] = []
    rejected: list[dict] = []

    for row in rows:
        image_id = row["image_id"]
        mpp: Optional[float] = row.get("mm_per_pixel")
        reason: Optional[str] = None

        if image_id in seen_ids:
            reason = f"duplicate image_id '{image_id}'"
        elif mpp is None and row.get("pt1_x") is None:
            reason = "missing calibration polyline"
        elif mpp is None:
            reason = "zero-length calibration polyline"
        elif mpp < mpp_min:
            reason = f"mm_per_pixel {mpp:.4f} < min {mpp_min} (ruler too long in pixels)"
        elif mpp > mpp_max:
            reason = f"mm_per_pixel {mpp:.4f} > max {mpp_max} (ruler too short in pixels)"

        if reason:
            rejected.append({**row, "rejection_reason": reason})
        else:
            passing.append(row)
            seen_ids.add(image_id)

    return passing, rejected


def write_calibration_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CALIBRATION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_rejection_log(rejected: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{'image_id':<30}  reason\n")
        f.write("-" * 80 + "\n")
        for row in rejected:
            f.write(f"{row['image_id']:<30}  {row.get('rejection_reason', 'unknown')}\n")
        f.write(f"\nTotal rejected: {len(rejected)}\n")
