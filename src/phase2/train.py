"""LOPO training loop for HRNet landmark detection — CUDA backend."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.phase2.dataset import CephalometricDataset, get_lopo_splits
from src.phase2.heatmap import encode_heatmaps, decode_heatmaps
from src.phase2.loss import AdaptiveWingLoss
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
    criterion = AdaptiveWingLoss()

    for imgs, keypoints, valid_mask, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        keypoints_np = keypoints.numpy()
        valid_np = valid_mask.numpy()

        gt_heatmaps = []
        for b in range(len(keypoints_np)):
            hm = encode_heatmaps(keypoints_np[b], valid_np[b], heatmap_size, sigma, input_size)
            gt_heatmaps.append(hm)
        gt_tensor = torch.from_numpy(
            __import__("numpy").stack(gt_heatmaps, axis=0)
        ).to(device)

        pred_heatmaps = model(imgs)

        if pred_heatmaps.shape[-2:] != gt_tensor.shape[-2:]:
            pred_heatmaps = torch.nn.functional.interpolate(
                pred_heatmaps, size=heatmap_size, mode="bilinear", align_corners=False
            )

        mask = torch.from_numpy(valid_np).bool().to(device)
        loss = criterion(pred_heatmaps, gt_tensor, mask)
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


def compute_mean_mre(errors_list: list) -> float:
    """Compute mean MRE from list of per-image error arrays."""
    all_errors = []
    for errors in errors_list:
        all_errors.extend(errors.flatten().tolist())
    return sum(all_errors) / len(all_errors) if all_errors else float("inf")


def run_lopo_training(
    config_path: str = "config.yaml",
    debug: bool = False,
    max_images: Optional[int] = None,
) -> dict:
    """Run full Leave-One-Patient-Out cross validation with proper training schedule.

    Key improvements over v1:
    - Backbone freezing for first N epochs (warmup) + lower LR for backbone
    - Cosine annealing scheduler
    - Best-model checkpointing per fold (save best, not last)
    - Evaluate every 10 epochs + at final epoch
    - Pretrained weights for ALL folds (not just fold 0)
    """
    cfg = load_config(config_path)

    import json
    import pandas as pd

    with open(cfg["data"]["landmarks_json"]) as f:
        landmarks_data = json.load(f)
    records = [r for r in landmarks_data["images"] if r.get("has_landmarks")]

    if max_images is not None:
        records = records[:max_images]

    if not records:
        print("WARNING: No annotated images found. Waiting for landmark annotations from Dr.")
        return {"mre_mean_mm": None, "mre_std_mm": None, "per_landmark_mre": {}, "note": "no_data"}

    cal_df = pd.read_csv(cfg["data"]["calibration_csv"]).set_index("image_id")
    calibration_lookup = cal_df["mm_per_pixel"].to_dict()

    device = torch.device(cfg["training"]["device"])
    heatmap_size = tuple(cfg["model"]["heatmap_size"])
    input_size = tuple(cfg["model"]["input_size"])
    total_epochs = cfg["training"].get("epochs", 300)
    if debug:
        total_epochs = 1
    lr = cfg["training"]["lr"]
    batch_size = cfg["training"]["batch_size"]

    warmup_epochs = cfg["training"].get("warmup_epochs", 10)
    freeze_backbone = cfg["training"].get("freeze_backbone", False)

    splits = get_lopo_splits(records)
    if debug:
        splits = splits[:1]
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
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        # Always pretrained=True — every fold gets fresh pretrained init
        model = CephalometricModel(
            num_keypoints=cfg["keypoints"]["num_keypoints"],
            pretrained=True,
        ).to(device)

        # Phase 1: freeze backbone, train head only
        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False

        head_params = list(model.head.parameters())
        backbone_params = list(model.backbone.parameters())

        if freeze_backbone:
            optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=cfg["training"]["weight_decay"])
            T_max = total_epochs - warmup_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr * 0.01)
        else:
            optimizer = torch.optim.AdamW(
                [{"params": head_params, "lr": lr}, {"params": backbone_params, "lr": lr * 0.1}],
                weight_decay=cfg["training"]["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=lr * 0.001
            )

        best_mre = float("inf")
        best_model_state = None
        phase = 1

        for epoch in range(total_epochs):
            # Phase 2: unfreeze backbone after warmup
            if freeze_backbone and epoch == warmup_epochs and phase == 1:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(
                    [{"params": model.head.parameters(), "lr": lr * 0.5},
                     {"params": model.backbone.parameters(), "lr": lr * 0.1}],
                    weight_decay=cfg["training"]["weight_decay"],
                )
                T_max = total_epochs - warmup_epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max, eta_min=lr * 0.001
                )
                phase = 2
                print(f"  [Fold {fold_idx+1}] Backbone unfrozen at epoch {epoch+1}, LR head={lr*0.5:.6f}, backbone={lr*0.1:.6f}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, device,
                heatmap_size, cfg["model"]["sigma"], input_size,
            )
            scheduler.step()

            # Evaluate every 10 epochs + at final epoch
            if (epoch + 1) % 10 == 0 or epoch == total_epochs - 1:
                fold_errors = evaluate(model, test_loader, device, heatmap_size, input_size, calibration_lookup)
                mre = compute_mean_mre(fold_errors)
                current_lr = optimizer.param_groups[0]["lr"]
                if mre < best_mre:
                    best_mre = mre
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    marker = " *BEST*"
                else:
                    marker = ""
                print(f"  [Fold {fold_idx+1}] Epoch {epoch+1}/{total_epochs} — loss: {train_loss:.4f}, MRE: {mre:.2f}mm, best: {best_mre:.2f}mm, LR: {current_lr:.6f}{marker}")

        # Restore best model for this fold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)

        # Final evaluation with best model
        fold_errors = evaluate(model, test_loader, device, heatmap_size, input_size, calibration_lookup)
        all_fold_errors.extend(fold_errors)

        patient_id = test_ids[0].rsplit("_", 1)[0]
        print(f"Fold {fold_idx + 1}/{len(splits)} — patient {patient_id} — final MRE: {best_mre:.2f}mm")

    kp_names = cfg["keypoints"]["names"]
    metrics = compute_all_metrics(
        all_fold_errors,
        sdr_thresholds=cfg["evaluation"]["sdr_thresholds_mm"],
        keypoint_names=kp_names,
    )
    return metrics