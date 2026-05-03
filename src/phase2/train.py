"""LOPO training loop for HRNet landmark detection — MPS backend."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.phase2.dataset import CephalometricDataset, get_lopo_splits
from src.phase2.heatmap import encode_heatmaps, decode_heatmaps
from src.phase2.metrics import radial_error, compute_all_metrics
from src.phase2.model import CephalometricModel
from src.utils.io import load_config


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    heatmap_size: tuple[int, int],
    sigma: float,
    input_size: tuple[int, int],
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for imgs, keypoints, valid_mask, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        keypoints_np = keypoints.numpy()
        valid_np = valid_mask.numpy()

        # Encode ground truth heatmaps
        gt_heatmaps = []
        for b in range(len(keypoints_np)):
            hm = encode_heatmaps(keypoints_np[b], valid_np[b], heatmap_size, sigma, input_size)
            gt_heatmaps.append(hm)
        gt_tensor = torch.from_numpy(
            __import__("numpy").stack(gt_heatmaps, axis=0)
        ).to(device)

        pred_heatmaps = model(imgs)

        # Resize prediction to match gt if needed
        if pred_heatmaps.shape[-2:] != gt_tensor.shape[-2:]:
            pred_heatmaps = torch.nn.functional.interpolate(
                pred_heatmaps, size=heatmap_size, mode="bilinear", align_corners=False
            )

        # Apply valid mask — only compute loss on annotated landmarks
        mask = torch.from_numpy(valid_np).bool().to(device)  # [B, N]
        mask_4d = mask.unsqueeze(-1).unsqueeze(-1).expand_as(gt_tensor)

        loss = criterion(pred_heatmaps[mask_4d], gt_tensor[mask_4d])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    heatmap_size: tuple[int, int],
    input_size: tuple[int, int],
    calibration_lookup: dict[str, float],
) -> list:
    """Returns list of per-image radial error arrays (in mm)."""
    model.eval()
    all_errors = []

    with torch.no_grad():
        for imgs, keypoints_gt, valid_mask, metas in loader:
            imgs = imgs.to(device)
            pred_heatmaps = model(imgs)

            if pred_heatmaps.shape[-2:] != (heatmap_size[0], heatmap_size[1]):
                pred_heatmaps = torch.nn.functional.interpolate(
                    pred_heatmaps, size=heatmap_size, mode="bilinear", align_corners=False
                )

            coords, confidence = decode_heatmaps(pred_heatmaps.cpu(), input_size)

            for b in range(imgs.shape[0]):
                image_id = metas["image_id"][b]
                mm_per_px = calibration_lookup.get(image_id, 1.0)
                valid = valid_mask[b].numpy()

                errors = radial_error(
                    coords[b].numpy()[valid],
                    keypoints_gt[b].numpy()[valid],
                    mm_per_px,
                )
                all_errors.append(errors)

    return all_errors


def run_lopo_training(config_path: str = "config.yaml") -> dict:
    """Run full Leave-One-Patient-Out cross validation. Returns aggregated metrics."""
    cfg = load_config(config_path)

    import json
    import pandas as pd

    with open(cfg["data"]["landmarks_json"]) as f:
        landmarks_data = json.load(f)
    records = [r for r in landmarks_data["images"] if r.get("has_landmarks")]

    cal_df = pd.read_csv(cfg["data"]["calibration_csv"]).set_index("image_id")
    calibration_lookup = cal_df["mm_per_pixel"].to_dict()

    device = torch.device(cfg["training"]["device"])
    heatmap_size = tuple(cfg["model"]["heatmap_size"])
    input_size = tuple(cfg["model"]["input_size"])

    splits = get_lopo_splits(records)
    all_fold_errors = []

    for fold_idx, (train_ids, test_ids) in enumerate(splits):
        from src.phase2.augmentation import build_train_transform, build_val_transform
        aug_cfg = cfg["augmentation"]
        train_transform = build_train_transform(
            rotation_limit=aug_cfg["rotation_limit"],
            zoom_limit=aug_cfg["zoom_limit"],
            brightness_limit=aug_cfg["brightness_limit"],
            contrast_limit=aug_cfg["contrast_limit"],
            clahe=aug_cfg["clahe"],
            horizontal_flip=False,
        )

        train_ds = CephalometricDataset(
            records, cfg["data"]["image_dir"], input_size, train_transform, image_ids=train_ids
        )
        test_ds = CephalometricDataset(
            records, cfg["data"]["image_dir"], input_size, build_val_transform(), image_ids=test_ids
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg["training"]["batch_size"],
            shuffle=True, num_workers=0
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        model = CephalometricModel(
            num_keypoints=cfg["keypoints"]["num_keypoints"],
            pretrained=(fold_idx == 0),  # download once, reuse checkpoint thereafter
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        for epoch in range(cfg["training"]["epochs"]):
            train_one_epoch(
                model, train_loader, optimizer, device,
                heatmap_size, cfg["model"]["sigma"], input_size,
            )

        fold_errors = evaluate(model, test_loader, device, heatmap_size, input_size, calibration_lookup)
        all_fold_errors.extend(fold_errors)

        patient_id = test_ids[0].rsplit("_", 1)[0]
        print(f"Fold {fold_idx + 1}/{len(splits)} — patient {patient_id} done")

    kp_names = cfg["keypoints"]["names"]
    metrics = compute_all_metrics(
        all_fold_errors,
        sdr_thresholds=cfg["evaluation"]["sdr_thresholds_mm"],
        keypoint_names=kp_names,
    )
    return metrics
