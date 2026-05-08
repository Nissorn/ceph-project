"""Dataset for EfficientNet-B3 multi-label treatment classification.

Loads T1 images and returns multi-hot label vectors for treatment classes.

Filtering rules (critical for correctness):
  1. Only T1 images — T2 images have no treatment labels.
  2. Exclude records with "Quality_Reject" in quality_flags.
  3. Exclude records with no image file (guard against missing data).

Output:
    image:  [3, 512, 512] float32 in [0, 1]
    labels: [6] float32  — multi-hot binary vector (one entry per treatment class)
    meta:   dict with image_id, patient_id
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.phase2c.classifier import TREATMENT_CLASSES, NUM_TREATMENT_CLASSES


class ClassificationDataset(Dataset):
    """
    Multi-label classification dataset for orthodontic treatment types.

    Args:
        records:   List of parsed record dicts (from landmarks_clean.json).
        image_dir: Path to directory containing raw images.
        input_size: (H, W). Default (512, 512).
        transform: Optional albumentations Compose transform.
        image_ids: Optional list of image_ids to filter to.
    """

    def __init__(
        self,
        records: list[dict],
        image_dir: str,
        input_size: tuple[int, int] = (512, 512),
        transform=None,
        image_ids: Optional[list[str]] = None,
    ):
        # Filter by image_ids if provided
        if image_ids is not None:
            id_set = set(image_ids)
            records = [r for r in records if r["image_id"] in id_set]

        # Rule 1: T1 only
        records = [r for r in records if r.get("timepoint") == "T1"]

        # Rule 2: Exclude Quality_Reject
        records = [
            r for r in records
            if "Quality_Reject" not in r.get("quality_flags", [])
        ]

        self.records = records
        self.image_dir = Path(image_dir)
        self.input_size = input_size  # (H, W)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        rec = self.records[idx]

        # --- Load image ---
        img_path = self.image_dir / rec["filename"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W = self.input_size
        img_resized = cv2.resize(img, (W, H))

        # --- Build multi-hot label vector ---
        treatments = set(rec.get("treatment", []))
        label = torch.zeros(NUM_TREATMENT_CLASSES, dtype=torch.float32)
        for i, cls in enumerate(TREATMENT_CLASSES):
            if cls in treatments:
                label[i] = 1.0

        # --- Optional transform ---
        if self.transform is not None:
            result = self.transform(image=img_resized)
            img_resized = result["image"]

        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0

        meta = {
            "image_id": rec["image_id"],
            "patient_id": rec["patient_id"],
        }

        return img_tensor, label, meta
