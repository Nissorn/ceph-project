"""PyTorch Dataset with patient-aware LOPO split for cephalometric landmark detection."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Confirmed keypoint order — must match cvat_parser.KEYPOINT_NAMES
KEYPOINT_NAMES = [
    "Upper_tip",
    "Upper_apex",
    "Labial_midroot",
    "Labial_crest",
    "Palatal_midroot",
    "Palatal_crest",
    "ANS",
    "PNS",
    "LB",
    "PB"
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)


def get_lopo_splits(records: list[dict]) -> list[tuple[list[str], list[str]]]:
    """
    Build Leave-One-Patient-Out splits.

    Returns list of (train_image_ids, test_image_ids) — one tuple per patient.
    T1 and T2 of the same patient are always in the same split.
    """
    patient_ids = sorted({r["patient_id"] for r in records})
    id_to_images: dict[str, list[str]] = {}
    for r in records:
        id_to_images.setdefault(r["patient_id"], []).append(r["image_id"])

    splits = []
    for test_patient in patient_ids:
        test_ids = id_to_images[test_patient]
        train_ids = [
            img_id
            for pid, imgs in id_to_images.items()
            if pid != test_patient
            for img_id in imgs
        ]
        splits.append((train_ids, test_ids))
    return splits


class CephalometricDataset(Dataset):
    """
    Dataset for landmark detection.

    Each item returns:
        image: torch.Tensor [3, H, W] float32, normalized to [0, 1]
        keypoints: torch.Tensor [N, 2] float32, coordinates in original image space
        valid_mask: torch.Tensor [N] bool, True = landmark is annotated
        meta: dict with image_id, patient_id, original_size
    """

    def __init__(
        self,
        records: list[dict],
        image_dir: str,
        input_size: tuple[int, int] = (512, 512),
        transform=None,
        image_ids: Optional[list[str]] = None,
    ):
        if image_ids is not None:
            id_set = set(image_ids)
            self.records = [r for r in records if r["image_id"] in id_set]
        else:
            self.records = [r for r in records if r.get("has_landmarks")]

        self.image_dir = Path(image_dir)
        self.input_size = input_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        import cv2

        rec = self.records[idx]
        img_path = self.image_dir / rec["file_name"]

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        scale_x = self.input_size[1] / orig_w
        scale_y = self.input_size[0] / orig_h

        img_resized = cv2.resize(img, (self.input_size[1], self.input_size[0]))

        kp_list = rec.get("keypoints", [])
        keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        valid_mask = np.zeros(NUM_KEYPOINTS, dtype=bool)

        for i, kp in enumerate(kp_list):
            if kp.get("visible", False):
                keypoints[i, 0] = kp["x"] * scale_x
                keypoints[i, 1] = kp["y"] * scale_y
                valid_mask[i] = True

        if self.transform is not None:
            transformed = self.transform(
                image=img_resized,
                keypoints=[(kp[0], kp[1]) for kp in keypoints],
            )
            img_resized = transformed["image"]
            transformed_kps = transformed["keypoints"]
            for i, (x, y) in enumerate(transformed_kps):
                keypoints[i] = [x, y]

        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0

        meta = {
            "image_id": rec["image_id"],
            "patient_id": rec["patient_id"],
            "original_size": (orig_h, orig_w),
            "scale": (scale_x, scale_y),
        }

        return (
            img_tensor,
            torch.from_numpy(keypoints),
            torch.from_numpy(valid_mask),
            meta,
        )
