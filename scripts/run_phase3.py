#!/usr/bin/env python3
"""Phase 3: Classify treatment type for a T1/T2 patient pair."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.phase3.superimposition import superimpose_on_ans_pns
from src.phase3.heuristics import classify_treatment
from src.utils.io import load_config


def load_predictions(json_path: str, patient_id: str, timepoint: str) -> tuple:
    with open(json_path) as f:
        data = json.load(f)
    for img in data["images"]:
        if img["patient_id"] == patient_id and img["timepoint"] == timepoint:
            kp = img["keypoints"]
            coords = np.array([[k["x"], k["y"]] for k in kp], dtype=np.float64)
            valid = np.array([k["visible"] for k in kp], dtype=bool)
            return coords, valid
    raise KeyError(f"Patient {patient_id} timepoint {timepoint} not found in {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Treatment classification")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--patient", required=True, help="Patient ID (e.g. Patient01)")
    parser.add_argument("--predictions", default=None,
                        help="Predictions JSON (default: use landmarks_clean.json ground truth)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_json = args.predictions or cfg["data"]["landmarks_json"]

    kp_t1, valid_t1 = load_predictions(pred_json, args.patient, "T1")
    kp_t2, valid_t2 = load_predictions(pred_json, args.patient, "T2")

    kp_t1_reg, kp_t2_reg, ok = superimpose_on_ans_pns(kp_t1, kp_t2, valid_t1, valid_t2)
    if not ok:
        print("WARNING: ANS/PNS not available in T1 — superimposition skipped")

    cal_df = pd.read_csv(cfg["data"]["calibration_csv"], index_col="image_id")
    mm_per_pixel_t1 = float(cal_df.loc[f"{args.patient}_T1", "mm_per_pixel"])

    p3_cfg = cfg["phase3"]
    result = classify_treatment(
        kp_t1_reg, kp_t2_reg, valid_t1, valid_t2,
        tipping_threshold_deg=p3_cfg["tipping_threshold_deg"],
        translation_threshold_mm=p3_cfg["translation_threshold_mm"],
        mm_per_pixel_t1=mm_per_pixel_t1,
    )

    print(f"\nPhase 3 result for {args.patient}:")
    print(f"  Treatment class:   {result['treatment_class']}")
    print(f"  Angle change:      {result['angle_change_deg']:.2f}°" if result["angle_change_deg"] else "  Angle change: N/A")
    if result["delta_tip_mm"]:
        print(f"  Δ tip (mm):        {result['delta_tip_mm']}")
        print(f"  Δ apex (mm):       {result['delta_apex_mm']}")


if __name__ == "__main__":
    main()
