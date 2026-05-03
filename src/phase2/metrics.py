"""Evaluation metrics: MRE (Mean Radial Error) and SDR (Success Detection Rate)."""

import numpy as np


def radial_error(pred: np.ndarray, gt: np.ndarray, mm_per_pixel: np.ndarray) -> np.ndarray:
    """
    Compute per-landmark radial error in mm.

    Args:
        pred: [N, 2] predicted (x, y) in pixels
        gt:   [N, 2] ground truth (x, y) in pixels
        mm_per_pixel: [N] or scalar — calibration factor per image

    Returns:
        errors: [N] float32 in mm
    """
    diff = pred - gt
    dist_px = np.sqrt((diff**2).sum(axis=-1))
    return dist_px * mm_per_pixel


def mean_radial_error(errors: list[np.ndarray]) -> tuple[float, float]:
    """
    Aggregate radial errors across all images.
    Returns (mean_mm, std_mm).
    """
    all_errors = np.concatenate([e.flatten() for e in errors])
    return float(all_errors.mean()), float(all_errors.std())


def per_landmark_mre(errors: list[np.ndarray], num_keypoints: int = 8) -> np.ndarray:
    """
    Compute per-landmark MRE across all images.
    Returns array of shape [num_keypoints] in mm.
    """
    stacked = np.stack(errors, axis=0)  # [n_images, N]
    return stacked.mean(axis=0)


def sdr(errors: list[np.ndarray], threshold_mm: float) -> float:
    """
    Success Detection Rate — fraction of predictions within threshold_mm.
    """
    all_errors = np.concatenate([e.flatten() for e in errors])
    return float((all_errors <= threshold_mm).mean())


def compute_all_metrics(
    errors: list[np.ndarray],
    sdr_thresholds: list[float] = (2.0, 2.5, 3.0, 4.0),
    keypoint_names: list[str] | None = None,
) -> dict:
    """Compute full evaluation report dict."""
    mre_mean, mre_std = mean_radial_error(errors)
    per_kp = per_landmark_mre(errors)

    report: dict = {
        "mre_mean_mm": mre_mean,
        "mre_std_mm": mre_std,
        "per_landmark_mre": {},
    }

    names = keypoint_names or [str(i) for i in range(len(per_kp))]
    for name, val in zip(names, per_kp):
        report["per_landmark_mre"][name] = float(val)

    for t in sdr_thresholds:
        key = f"sdr_{t}mm".replace(".", "_")
        report[key] = sdr(errors, t)

    return report
