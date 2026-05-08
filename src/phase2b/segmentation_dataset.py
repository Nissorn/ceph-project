"""Dataset for U-Net polygon segmentation.

Reads polygon data from landmarks_clean.json and converts them to
binary raster masks at load time using cv2.fillPoly.

Output:
    image:  [3, 512, 512] float32 in [0, 1]
    masks:  [3, 512, 512] float32 binary — channel order = POLYGON_CLASSES
    meta:   dict with image_id, patient_id

NOTE: Only records where record["polygons"] is non-empty will produce
meaningful mask data. Skipping records with no polygons is controlled
by the `require_polygons` flag (default True).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.phase2b.segmentation import POLYGON_CLASSES, NUM_SEG_CLASSES


class SegmentationDataset(Dataset):
    """
    Loads images and converts CVAT polygon annotations to binary masks.

    Args:
        records:         List of parsed record dicts (from landmarks_clean.json).
        image_dir:       Path to the directory containing the raw images.
        input_size:      (H, W) — target image size. Default (512, 512).
        transform:       Optional albumentations Compose transform.
        require_polygons: If True (default), only include records that have
                          at least one polygon annotation.
        image_ids:       Optional list of image_ids to filter records.
    """

    def __init__(
        self,
        records: list[dict],
        image_dir: str,
        input_size: tuple[int, int] = (512, 512),
        transform=None,
        require_polygons: bool = True,
        image_ids: Optional[list[str]] = None,
    ):
        if image_ids is not None:
            id_set = set(image_ids)
            records = [r for r in records if r["image_id"] in id_set]

        if require_polygons:
            records = [r for r in records if r.get("polygons")]

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

        orig_h, orig_w = img.shape[:2]
        H, W = self.input_size

        img_resized = cv2.resize(img, (W, H))

        # --- Build masks ---
        # Each polygon is in original image coords → scale + rasterise
        masks = np.zeros((NUM_SEG_CLASSES, H, W), dtype=np.float32)
        polygons_data = rec.get("polygons", {})

        for ch_idx, class_name in enumerate(POLYGON_CLASSES):
            if class_name not in polygons_data:
                continue

            # Scale polygon points to resized image coordinates
            raw_pts = np.array(polygons_data[class_name], dtype=np.float32)  # [N, 2] (x, y)
            raw_pts[:, 0] *= W / orig_w
            raw_pts[:, 1] *= H / orig_h
            pts_int = raw_pts.astype(np.int32)

            canvas = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(canvas, [pts_int], color=1)
            masks[ch_idx] = canvas.astype(np.float32)

        # --- Optional albumentations transform ---
        if self.transform is not None:
            # albumentations expects masks as [H, W, C] or list of [H, W]
            aug_masks = [masks[c] for c in range(NUM_SEG_CLASSES)]
            result = self.transform(image=img_resized, masks=aug_masks)
            img_resized = result["image"]
            aug_masks = result["masks"]
            masks = np.stack(aug_masks, axis=0)

        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        masks_tensor = torch.from_numpy(masks)

        meta = {
            "image_id": rec["image_id"],
            "patient_id": rec["patient_id"],
            "original_size": (orig_h, orig_w),
        }

        return img_tensor, masks_tensor, meta
