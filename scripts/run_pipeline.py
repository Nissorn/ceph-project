#!/usr/bin/env python3
"""Full pipeline: Phase 1→3 + Phase 4 clinical output for a patient pair."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.phase3.superimposition import superimpose_on_ans_pns
from src.phase3.heuristics import classify_treatment
from src.phase4.convert import load_calibration, build_clinical_report
from src.phase4.visualize import draw_tracing_overlay, save_overlay
from src.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: landmarks → clinical report + overlay")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--patient", required=True, help="Patient ID (e.g. Patient01)")
    parser.add_argument("--predictions", default=None,
                        help="Predictions JSON (default: landmarks_clean.json)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_json = args.predictions or cfg["data"]["landmarks_json"]
    output_dir = Path(args.output_dir) / args.patient
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pred_json) as f:
        all_images = json.load(f)

    def find_image(pid, tp):
        for img in all_images:
            if img["patient_id"] == pid and img["timepoint"] == tp:
                return img
        raise KeyError(f"{pid}_{tp} not found")

    rec_t1 = find_image(args.patient, "T1")
    rec_t2 = find_image(args.patient, "T2")

    kp_t1 = np.array([[k["x"], k["y"]] for k in rec_t1["keypoints"]], dtype=np.float64)
    kp_t2 = np.array([[k["x"], k["y"]] for k in rec_t2["keypoints"]], dtype=np.float64)
    valid_t1 = np.array([k["visible"] for k in rec_t1["keypoints"]], dtype=bool)
    valid_t2 = np.array([k["visible"] for k in rec_t2["keypoints"]], dtype=bool)

    kp_t1_reg, kp_t2_reg, ok = superimpose_on_ans_pns(kp_t1, kp_t2, valid_t1, valid_t2)

    cal_df = load_calibration(cfg["data"]["calibration_csv"])
    mm_per_pixel_t1 = float(cal_df.loc[rec_t1["image_id"], "mm_per_pixel"])

    p3_cfg = cfg["phase3"]
    classification = classify_treatment(
        kp_t1_reg, kp_t2_reg, valid_t1, valid_t2,
        tipping_threshold_deg=p3_cfg["tipping_threshold_deg"],
        translation_threshold_mm=p3_cfg["translation_threshold_mm"],
        mm_per_pixel_t1=mm_per_pixel_t1,
    )

    # Placeholder confidence arrays (1.0 = ground truth annotations)
    confidence_t1 = np.ones(len(cfg["keypoints"]["names"]))
    confidence_t2 = np.ones(len(cfg["keypoints"]["names"]))

    report = build_clinical_report(
        image_id_t1=rec_t1["image_id"],
        image_id_t2=rec_t2["image_id"],
        classification_result=classification,
        calibration_df=cal_df,
        confidence_t1=confidence_t1,
        confidence_t2=confidence_t2,
        keypoint_names=cfg["keypoints"]["names"],
        low_confidence_threshold=cfg["evaluation"]["confidence_low_threshold"],
    )

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    t2_img_path = str(Path(cfg["data"]["image_dir"]) / rec_t2["filename"])
    if Path(t2_img_path).exists():
        overlay = draw_tracing_overlay(
            t2_img_path, kp_t1, kp_t2, valid_t1, valid_t2,
            cfg["keypoints"]["names"],
            low_confidence_landmarks=report["low_confidence_landmarks"],
        )
        overlay_path = str(output_dir / "tracing_overlay.jpg")
        save_overlay(overlay, overlay_path)
        print(f"  Overlay saved:  {overlay_path}")
    else:
        print(f"  NOTE: T2 image not found at {t2_img_path} — overlay skipped")

    print(f"\nPipeline complete for {args.patient}:")
    print(f"  Treatment class: {report['treatment_class']}")
    print(f"  Report saved:   {report_path}")


if __name__ == "__main__":
    main()
