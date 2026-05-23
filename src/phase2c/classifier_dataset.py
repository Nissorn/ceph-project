"""
Phase 2c — Treatment Recommendation Engine Dataset
===================================================
Loads T1 images, computes Phase 3 scalars from GT keypoints, and returns
treatment label vectors for fusion model training.

Scalar computation pipeline per record:
    GT keypoints (+ optional Gaussian noise) → calculate_metrics()
    → u1_pp_angle_deg, lb_apex_dist_mm, pb_apex_dist_mm
    → _get_apex_position() → one-hot [3]
    → scalar tensor [6]

Filtering rules:
    1. T1 images only — T2 images have no treatment labels.
    2. Exclude records with "Quality_Reject" in quality_flags.
    3. Skip records missing required keypoints (Upper_tip, Upper_apex, ANS, PNS, LB, PB).

Noise injection (exposure bias mitigation):
    Enabled via inject_training_noise=True. Adds Gaussian noise to keypoint
    pixel coords before computing Phase 3 scalars. sigma_pixels should be
    calibrated to the Phase 2 HRNet MRE once v6 training benchmarks are available.
    Default sigma_pixels ~= 15 px (≈ 1.5 mm at 0.0984 mm/px) as placeholder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.phase2c.classifier import (
    TREATMENT_CLASSES,
    NUM_TREATMENT_CLASSES,
    apex_position_to_onehot,
    SCALAR_DIM,
)
from src.phase3.biomechanics import calculate_metrics, _get_apex_position

_REQUIRED_KEYPOINTS = ("Upper_tip", "Upper_apex", "ANS", "PNS", "LB", "PB")


def _build_landmarks(
    keypoints: list[dict],
    noise_sigma_px: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Optional[dict[str, tuple[float, float]]]:
    """
    Convert landmarks_clean.json keypoints list to {name: (x, y)} dict.
    Returns None if any required keypoint is missing or not visible.
    If noise_sigma_px > 0, adds Gaussian noise to pixel coords.
    """
    lm: dict[str, tuple[float, float]] = {}
    for kp in keypoints:
        if kp.get("visible", False):
            lm[kp["name"]] = (float(kp["x"]), float(kp["y"]))

    if any(k not in lm for k in _REQUIRED_KEYPOINTS):
        return None

    if noise_sigma_px > 0.0:
        _rng = rng if rng is not None else np.random.default_rng()
        noised: dict[str, tuple[float, float]] = {}
        for name, (x, y) in lm.items():
            dx, dy = _rng.normal(0.0, noise_sigma_px, size=2)
            noised[name] = (x + float(dx), y + float(dy))
        return noised

    return lm


def _compute_scalars(
    landmarks: dict[str, tuple[float, float]],
    mm_per_pixel: float,
) -> torch.Tensor:
    """
    Run Phase 3 biomechanics on a landmarks dict and return the 6-dim scalar tensor.
    """
    metrics = calculate_metrics(landmarks, mm_per_pixel=mm_per_pixel)
    apex_pos = _get_apex_position(
        metrics["lb_apex_dist_mm"],
        metrics["pb_apex_dist_mm"],
    )
    one_hot = apex_position_to_onehot(apex_pos)
    vec = [
        metrics["u1_pp_angle_deg"],
        metrics["lb_apex_dist_mm"],
        metrics["pb_apex_dist_mm"],
        *one_hot,
    ]
    return torch.tensor(vec, dtype=torch.float32)


class ClassificationDataset(Dataset):
    """
    Multi-label treatment recommendation dataset for Phase 2c fusion model.

    Args:
        records:              Parsed records from landmarks_clean.json.
        image_dir:            Path to directory containing raw images.
        calibration:          Dict mapping image_id → mm_per_pixel.
        input_size:           (H, W). Default (512, 512).
        transform:            Optional albumentations Compose transform.
        image_ids:            Optional subset of image_ids to include.
        inject_training_noise: If True, add Gaussian noise to keypoint coords
                              before computing Phase 3 scalars.
        noise_sigma_pixels:   Std of noise in pixels. Calibrate to Phase 2 MRE
                              benchmarks. Default 15.0 px ≈ 1.5 mm.
        seed:                 RNG seed for reproducible noise injection.
    """

    def __init__(
        self,
        records: list[dict],
        image_dir: str,
        calibration: dict[str, float],
        input_size: tuple[int, int] = (512, 512),
        transform=None,
        image_ids: Optional[list[str]] = None,
        inject_training_noise: bool = False,
        noise_sigma_pixels: float = 15.0,
        seed: int = 42,
    ):
        if image_ids is not None:
            id_set = set(image_ids)
            records = [r for r in records if r["image_id"] in id_set]

        records = [r for r in records if r.get("timepoint") == "T1"]
        records = [
            r for r in records
            if "Quality_Reject" not in r.get("quality_flags", [])
        ]

        # Drop records where required keypoints are missing
        valid = []
        for r in records:
            lm = _build_landmarks(r.get("keypoints", []))
            if lm is not None:
                valid.append(r)
        records = valid

        self.records = records
        self.image_dir = Path(image_dir)
        self.calibration = calibration
        self.input_size = input_size
        self.transform = transform
        self.inject_training_noise = inject_training_noise
        self.noise_sigma_pixels = noise_sigma_pixels
        self._rng = np.random.default_rng(seed) if inject_training_noise else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        rec = self.records[idx]
        image_id = rec["image_id"]

        # --- Image ---
        img_path = self.image_dir / rec["filename"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = self.input_size
        img = cv2.resize(img, (W, H))

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # --- Phase 3 scalars ---
        sigma = self.noise_sigma_pixels if self.inject_training_noise else 0.0
        landmarks = _build_landmarks(rec.get("keypoints", []), noise_sigma_px=sigma, rng=self._rng)
        mm_per_pixel = self.calibration.get(image_id, 0.0984)
        scalars = _compute_scalars(landmarks, mm_per_pixel)

        # --- Treatment labels ---
        treatments = set(rec.get("treatment", []))
        label = torch.zeros(NUM_TREATMENT_CLASSES, dtype=torch.float32)
        for i, cls in enumerate(TREATMENT_CLASSES):
            if cls in treatments:
                label[i] = 1.0

        meta = {"image_id": image_id, "patient_id": rec["patient_id"]}
        return img_tensor, scalars, label, meta
