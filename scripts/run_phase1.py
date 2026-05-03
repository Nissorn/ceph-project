#!/usr/bin/env python3
"""Phase 1: Parse CVAT XML annotations → landmarks_clean.json + calibration.csv"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase1.cvat_parser import parse_cvat_xml
from src.phase1.export import run_export
from src.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Parse annotations and calibration")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    xml_path = cfg["data"]["annotation_file"]
    landmarks_json = cfg["data"]["landmarks_json"]
    calibration_csv = cfg["data"]["calibration_csv"]

    if not Path(xml_path).exists():
        print(f"ERROR: Annotation file not found: {xml_path}")
        sys.exit(1)

    print(f"Parsing: {xml_path}")
    records = parse_cvat_xml(xml_path)
    print(f"  Found {len(records)} image records")

    stats = run_export(records, landmarks_json, calibration_csv)

    print("\nPhase 1 complete:")
    print(f"  Total images:        {stats['total_images']}")
    print(f"  Quality rejected:    {stats['rejected']}")
    print(f"  Exported:            {stats['exported']}")
    print(f"  With landmarks:      {stats['with_landmarks']}")
    print(f"  With calibration:    {stats['with_calibration']}")
    print(f"\n  Output: {landmarks_json}")
    print(f"  Output: {calibration_csv}")


if __name__ == "__main__":
    main()
