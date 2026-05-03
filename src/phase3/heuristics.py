"""Geometric heuristics for orthodontic treatment classification.

Classification thresholds (tipping_threshold_deg, translation_threshold_mm)
are PENDING confirmation from Dr. Code handles null gracefully.
"""

import math
import numpy as np
from typing import Optional


TREATMENT_CLASSES = [
    "Uncontrolled_tipping",
    "Controlled_tipping",
    "Translation",
    "Root_torque",
    "Extrusion",
    "Intrusion",
]

PENDING_THRESHOLD = "pending_threshold"

# Keypoint indices
TIP_IDX = 0
APEX_IDX = 1


def _long_axis_angle_deg(tip: np.ndarray, apex: np.ndarray) -> float:
    """Angle of the tooth long axis (apex→tip) relative to vertical, in degrees."""
    vec = tip - apex
    return math.degrees(math.atan2(vec[0], -vec[1]))


def _magnitude(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def classify_treatment(
    kp_t1_reg: np.ndarray,
    kp_t2_reg: np.ndarray,
    valid_t1: np.ndarray,
    valid_t2: np.ndarray,
    tipping_threshold_deg: Optional[float],
    translation_threshold_mm: Optional[float],
    mm_per_pixel_t1: float = 1.0,
) -> dict:
    """
    Classify orthodontic treatment from registered T1/T2 landmarks.

    Args:
        kp_t1_reg, kp_t2_reg: [N, 2] landmarks in ANS-PNS registered pixel space
        valid_t1, valid_t2: [N] bool masks
        tipping_threshold_deg: from config — null until Dr. confirms
        translation_threshold_mm: from config — null until Dr. confirms
        mm_per_pixel_t1: calibration for T1 image

    Returns dict with:
        treatment_class: str or "pending_threshold"
        angle_change_deg: float
        delta_tip_mm: [2] vector
        delta_apex_mm: [2] vector
    """
    required = [TIP_IDX, APEX_IDX]
    for idx in required:
        if not (valid_t1[idx] and valid_t2[idx]):
            return {
                "treatment_class": "insufficient_landmarks",
                "angle_change_deg": None,
                "delta_tip_mm": None,
                "delta_apex_mm": None,
            }

    tip_t1 = kp_t1_reg[TIP_IDX]
    apex_t1 = kp_t1_reg[APEX_IDX]
    tip_t2 = kp_t2_reg[TIP_IDX]
    apex_t2 = kp_t2_reg[APEX_IDX]

    angle_t1 = _long_axis_angle_deg(tip_t1, apex_t1)
    angle_t2 = _long_axis_angle_deg(tip_t2, apex_t2)
    angle_change_deg = angle_t2 - angle_t1

    delta_tip_px = tip_t2 - tip_t1
    delta_apex_px = apex_t2 - apex_t1
    delta_tip_mm = delta_tip_px * mm_per_pixel_t1
    delta_apex_mm = delta_apex_px * mm_per_pixel_t1

    result = {
        "angle_change_deg": float(angle_change_deg),
        "delta_tip_mm": delta_tip_mm.tolist(),
        "delta_apex_mm": delta_apex_mm.tolist(),
        "treatment_class": None,
    }

    if tipping_threshold_deg is None or translation_threshold_mm is None:
        result["treatment_class"] = PENDING_THRESHOLD
        return result

    tip_magnitude_mm = _magnitude(delta_tip_mm)
    apex_magnitude_mm = _magnitude(delta_apex_mm)
    vertical_delta_tip = float(delta_tip_mm[1])

    if abs(angle_change_deg) > tipping_threshold_deg:
        if apex_magnitude_mm < translation_threshold_mm:
            result["treatment_class"] = "Uncontrolled_tipping"
        else:
            result["treatment_class"] = "Controlled_tipping"
    elif abs(delta_apex_mm[0]) > translation_threshold_mm and tip_magnitude_mm > translation_threshold_mm:
        result["treatment_class"] = "Translation"
    elif abs(angle_change_deg) > tipping_threshold_deg * 0.5:
        result["treatment_class"] = "Root_torque"
    elif vertical_delta_tip < -translation_threshold_mm:
        result["treatment_class"] = "Intrusion"
    elif vertical_delta_tip > translation_threshold_mm:
        result["treatment_class"] = "Extrusion"
    else:
        result["treatment_class"] = PENDING_THRESHOLD

    return result
