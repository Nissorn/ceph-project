#!/usr/bin/env python3
"""
scripts/convert_predictions_for_frontend.py
=============================================

Convert the predictions.json produced by predict_all.py
into the format required by the Astro frontend's
public/data/predictions.json.

predict_all.py output (already correct format):
    {"image_001.jpg": {"Upper_tip": {"x": 1450.5, "y": 800.2, "confidence": 0.98}, ...}, ...}

Frontend expects the exact same format, so this script is mostly a copy —
use it when you want to:
  1. Copy from outputs/predictions.json → frontend/public/data/predictions.json
  2. Or validate the structure before deploying

Usage
-----
    python scripts/convert_predictions_for_frontend.py \\
        --input outputs/predictions.json \\
        --output frontend/public/data/predictions.json

Or just cp the file directly:
    cp outputs/predictions.json frontend/public/data/predictions.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.io import load_json, save_json


EXPECTED_KEYPOINTS = {
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
}


def validate(data: dict) -> list[str]:
    """Return list of warnings; empty list = clean."""
    warnings = []
    for fname, landmarks in data.items():
        if fname.startswith("_"):
            continue  # skip metadata keys
        if not isinstance(landmarks, dict):
            warnings.append(f'  [{fname}] value is not a dict — skipping')
            continue
        for kp_name, kp_data in landmarks.items():
            if not isinstance(kp_data, dict):
                warnings.append(f'  [{fname}] "{kp_name}" is not a dict — skipping')
                continue
            for field in ("x", "y", "confidence"):
                if field not in kp_data:
                    warnings.append(f'  [{fname}] "{kp_name}" missing field "{field}"')
        extra = set(landmarks.keys()) - EXPECTED_KEYPOINTS
        if extra:
            warnings.append(f'  [{fname}] unexpected keypoint(s): {extra}')
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and copy predictions.json to the Astro frontend public directory"
    )
    parser.add_argument("--input",  type=str, required=True, help="Source predictions.json (from predict_all.py)")
    parser.add_argument("--output", type=str, required=True, help="Destination — frontend/public/data/predictions.json")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        print(f"[ERROR] Input file not found: {src}")
        sys.exit(1)

    print(f"Loading: {src}")
    data = load_json(str(src))
    print(f"  Images: {len(data)}")

    # Validate
    warnings = validate(data)
    if warnings:
        print(f"Validation warnings ({len(warnings)}):")
        for w in warnings:
            print(w)
    else:
        print("  Validation: PASSED — all 10 keypoints present in all images")

    # Write to destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_json(data, str(dst))
    print(f"\nCopied → {dst}")
    print(f"  The Astro dev server will serve this at /data/predictions.json")


if __name__ == "__main__":
    main()