"""Convert pixel-space measurements to millimeters using per-image calibration."""

import numpy as np
import pandas as pd
from pathlib import Path


def load_calibration(calibration_csv: str) -> pd.DataFrame:
    return pd.read_csv(calibration_csv, index_col="image_id")


def get_mm_per_pixel(calibration_df: pd.DataFrame, image_id: str) -> float:
    """Look up mm_per_pixel for a specific image. Raises KeyError if not found."""
    return float(calibration_df.loc[image_id, "mm_per_pixel"])


def pixels_to_mm(value_px: float, mm_per_pixel: float) -> float:
    return value_px * mm_per_pixel


def vector_pixels_to_mm(vec_px: np.ndarray, mm_per_pixel: float) -> np.ndarray:
    return vec_px * mm_per_pixel


def build_clinical_report(
    image_id_t1: str,
    image_id_t2: str,
    classification_result: dict,
    calibration_df: pd.DataFrame,
    confidence_t1: np.ndarray,
    confidence_t2: np.ndarray,
    keypoint_names: list[str],
    low_confidence_threshold: float = 0.3,
) -> dict:
    """
    Build the final clinical report dict for a patient pair.

    Returns JSON-serializable dict with all measurements in mm.
    """
    mm_per_pixel = get_mm_per_pixel(calibration_df, image_id_t1)

    report = {
        "patient_id": image_id_t1.rsplit("_", 1)[0],
        "image_t1": image_id_t1,
        "image_t2": image_id_t2,
        "mm_per_pixel": mm_per_pixel,
        "treatment_class": classification_result.get("treatment_class"),
        "angle_change_deg": classification_result.get("angle_change_deg"),
        "delta_tip_mm": classification_result.get("delta_tip_mm"),
        "delta_apex_mm": classification_result.get("delta_apex_mm"),
        "confidence": {},
        "low_confidence_landmarks": [],
    }

    for i, name in enumerate(keypoint_names):
        conf_t1 = float(confidence_t1[i]) if i < len(confidence_t1) else 0.0
        conf_t2 = float(confidence_t2[i]) if i < len(confidence_t2) else 0.0
        report["confidence"][name] = {"t1": conf_t1, "t2": conf_t2}
        if conf_t1 < low_confidence_threshold or conf_t2 < low_confidence_threshold:
            report["low_confidence_landmarks"].append(name)

    return report
