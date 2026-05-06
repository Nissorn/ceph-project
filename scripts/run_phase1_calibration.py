#!/usr/bin/env python3
"""Phase 1 calibration pipeline.

Usage:
    python scripts/run_phase1_calibration.py --cvat_xml data/annotations.xml --output_dir data/processed/

Outputs:
    calibration.csv        all 104 rows (including rejects)
    calibration_clean.csv  QC-passing rows only
    rejection_log.txt      rejected image_ids with reason
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cvat_parser import parse_cvat_xml
from src.data.calibration import build_calibration_rows
from src.data.quality_filter import (
    MM_PER_PIXEL_MIN,
    MM_PER_PIXEL_MAX,
    filter_calibration,
    write_calibration_csv,
    write_rejection_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: CVAT XML → calibration.csv + QC filtering"
    )
    parser.add_argument(
        "--cvat_xml", required=True,
        help="Path to CVAT XML export (e.g. data/annotations.xml)"
    )
    parser.add_argument(
        "--output_dir", default="data/processed",
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--ruler_mm", type=float, default=30.0,
        help="Physical ruler length in mm (default: 30.0)"
    )
    parser.add_argument(
        "--mpp_min", type=float, default=MM_PER_PIXEL_MIN,
        help=f"Min valid mm_per_pixel (default: {MM_PER_PIXEL_MIN})"
    )
    parser.add_argument(
        "--mpp_max", type=float, default=MM_PER_PIXEL_MAX,
        help=f"Max valid mm_per_pixel (default: {MM_PER_PIXEL_MAX})"
    )
    args = parser.parse_args()

    xml_path = Path(args.cvat_xml)
    out_dir = Path(args.output_dir)

    if not xml_path.exists():
        print(f"ERROR: CVAT XML not found: {xml_path}", file=sys.stderr)
        sys.exit(1)

    # --- Parse ---
    print(f"Parsing: {xml_path}")
    records = parse_cvat_xml(xml_path)
    print(f"  {len(records)} image records")

    n_calibration = sum(1 for r in records if r["has_calibration"])
    n_landmarks = sum(1 for r in records if r["has_landmarks"])
    print(f"  {n_calibration} have calibration polyline")
    print(f"  {n_landmarks} have landmark annotations")

    # --- Compute calibration ---
    rows = build_calibration_rows(records, ruler_mm=args.ruler_mm)

    # --- Quality filter ---
    passing, rejected = filter_calibration(rows, mpp_min=args.mpp_min, mpp_max=args.mpp_max)

    # --- Write outputs ---
    cal_path = out_dir / "calibration.csv"
    clean_path = out_dir / "calibration_clean.csv"
    log_path = out_dir / "rejection_log.txt"

    write_calibration_csv(rows, cal_path)
    write_calibration_csv(passing, clean_path)
    write_rejection_log(rejected, log_path)

    # --- Summary ---
    mpp_values = [r["mm_per_pixel"] for r in passing if r["mm_per_pixel"] is not None]

    print(f"\nPhase 1 calibration complete")
    print(f"  Total parsed:       {len(records)}")
    print(f"  QC passed:          {len(passing)}")
    print(f"  QC rejected:        {len(rejected)}")

    if mpp_values:
        print(f"\n  mm/pixel  min:      {min(mpp_values):.4f}")
        print(f"  mm/pixel  max:      {max(mpp_values):.4f}")
        print(f"  mm/pixel  mean:     {statistics.mean(mpp_values):.4f}")
        if len(mpp_values) > 1:
            print(f"  mm/pixel  std:      {statistics.stdev(mpp_values):.4f}")

    if rejected:
        print(f"\n  Rejected images:")
        for r in rejected:
            print(f"    {r['image_id']:<28} {r.get('rejection_reason', '')}")

    print(f"\n  → {cal_path}")
    print(f"  → {clean_path}")
    print(f"  → {log_path}")


if __name__ == "__main__":
    main()
