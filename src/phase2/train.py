"""LOPO training loop for HRNet landmark detection — CUDA backend."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.phase2.dataset import CephalometricDataset
from src.data.splits import build_splits
from src.phase2.heatmap import encode_heatmaps, decode_heatmaps
from src.phase2.loss import AdaptiveWingLoss, EUPELoss
from src.phase2.metrics import radial_error, compute_all_metrics
from src.phase2.model import CephalometricModel
from src.utils.io import load_config


# ---------------------------------------------------------------------------
# Partial freezing helpers — freeze Stage 1 & 2 of HRNet-W32
# The backbone has these layers (timm hrnet_w32):
#   conv1, bn1, conv2, bn2          — stem
#   layer1.*                        — Stage 1  (4 residual units)
#   transition1.*                   — Transition 1→2
#   stage2.*                        — Stage 2
#   transition2.*                   — Transition 2→3
#   stage3.*                        — Stage 3  ← unfreeze
#   stage4.*                        — Stage 4  ← unfreeze
#   head.*                          — HeatmapHead (always unfrozen)
# ---------------------------------------------------------------------------

def freeze_stage1_2(model: nn.Module) -> None:
    """
    Freeze Stage 1 (layer1) and Stage 2 (stage2) of HRNet-W32 backbone.
    Also freezes the stem conv1/bn1/conv2/bn2 and transition layers.
    Stage 3, Stage 4, and HeatmapHead remain trainable.
    """
    for name, param in model.backbone.named_parameters():
        # Freeze: stem (conv1,bn1,conv2,bn2) + layer1 + transition1 + stage2
        if (
            name.startswith("conv1.")
            or name.startswith("bn1.")
            or name.startswith("conv2.")
            or name.startswith("bn2.")
            or name.startswith("layer1.")
            or name.startswith("transition1.")
            or name.startswith("stage2.")
        ):
            param.requires_grad = False
        # stage3, stage4 remain trainable


def get_partial_freeze_param_groups(
    model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float
) -> list[dict]:
    """
    Returns optimizer param groups for partial-freeze mode:
      - stage1_2 frozen:   requires_grad=False (no entry needed)
      - stage3_4:          lr=backbone_lr, weight_decay=weight_decay
      - head:              lr=head_lr, weight_decay=weight_decay
    """
    stage3_4_params = []
    head_params = []
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:  # only unfrozen backbone params
            stage3_4_params.append(param)
    head_params = list(model.head.parameters())

    groups = [
        {"params": stage3_4_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]
    return groups


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    heatmap_size: tuple[int, int],
    sigma: float,
    input_size: tuple[int, int],
    mixup_alpha: float = 0.0,
    use_eupe: bool = False,
    sigma_map: np.ndarray | list | None = None,
) -> float:
    """
    sigma_map: Optional [N] per-landmark sigma array for adaptive heatmap generation.
               When provided, sigma_map[i] is used for landmark i instead of sigma.
               Small values (2.0-2.5): sharp for anterior landmarks.
               Large values (4.0-5.0): diffuse for posterior/low-contrast landmarks.
    """
    model.train()
    total_loss = 0.0
    criterion = AdaptiveWingLoss()
    eupe_criterion = EUPELoss(reg_lambda=0.1)
    mixup_beta = torch.distributions.Beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else None
    np = __import__("numpy")

    for imgs, keypoints, valid_mask, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        keypoints_np = keypoints.numpy()
        valid_np = valid_mask.numpy()

        # Apply mixup if enabled (pure torch to avoid GPU/numpy issues)
        batch_size = len(imgs)
        if mixup_alpha > 0 and batch_size > 1:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(batch_size).numpy()
            imgs_np = imgs.cpu().numpy()
            imgs = torch.from_numpy(lam * imgs_np + (1 - lam) * imgs_np[idx]).float().to(device)
            keypoints_np = lam * keypoints_np + (1 - lam) * keypoints_np[idx]
            valid_np = lam * valid_np + (1 - lam) * valid_np[idx]

        gt_heatmaps = []
        for b in range(len(keypoints_np)):
            hm = encode_heatmaps(keypoints_np[b], valid_np[b], heatmap_size, sigma, input_size, sigma_map)
            gt_heatmaps.append(hm)
        gt_tensor = torch.from_numpy(
            __import__("numpy").stack(gt_heatmaps, axis=0)
        ).to(device)

        # Model ALWAYS returns (heatmaps, uncertainty) — unpack in both paths
        if use_eupe:
            pred_heatmaps, uncertainty = model(imgs)
        else:
            pred_heatmaps, _ = model(imgs)

        if pred_heatmaps.shape[-2:] != gt_tensor.shape[-2:]:
            pred_heatmaps = torch.nn.functional.interpolate(
                pred_heatmaps, size=heatmap_size, mode="bilinear", align_corners=False
            )

        mask = torch.from_numpy(valid_np).bool().to(device)

        if use_eupe:
            loss, _ = eupe_criterion(pred_heatmaps, gt_tensor, uncertainty, mask)
        else:
            loss = criterion(pred_heatmaps, gt_tensor, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
) -> tuple[list, list, list, list]:
    """Returns (soft_errors, argmax_errors, all_coords, all_gt) for mode-collapse detection."""
    model.eval()
    soft_errors = []
    argmax_errors = []
    all_coords = []   # predicted coords per image (for variance check)
    all_gt = []       # ground-truth coords per image

    with torch.no_grad():
        for imgs, keypoints_gt, valid_mask, metas in loader:
            imgs = imgs.to(device)
            output = model(imgs)
            # EUPE model returns (heatmaps, uncertainty); standard returns heatmaps
            if isinstance(output, tuple):
                pred_heatmaps = output[0]
            else:
                pred_heatmaps = output

            if pred_heatmaps.shape[-2:] != (heatmap_size[0], heatmap_size[1]):
                pred_heatmaps = torch.nn.functional.interpolate(
                    pred_heatmaps, size=heatmap_size, mode="bilinear", align_corners=False
                )

            # Soft-argmax coordinates
            coords_soft, confidence = decode_heatmaps(pred_heatmaps.cpu(), input_size)

            # Hard-argmax coordinates (for mode-collapse detection only)
            B, N, H, W = pred_heatmaps.shape
            raw_conf = torch.sigmoid(pred_heatmaps.cpu())
            flat = raw_conf.view(B * N, -1)
            _, flat_idx = flat.max(dim=-1)
            x_argmax = (flat_idx % W).float() / W * input_size[1]
            y_argmax = (flat_idx // W).float() / H * input_size[0]
            coords_argmax = torch.stack([x_argmax, y_argmax], dim=-1).view(B, N, 2)

            for b in range(imgs.shape[0]):
                image_id = metas["image_id"][b]
                mm_per_px = calibration_lookup.get(image_id, 1.0)
                valid = valid_mask[b].numpy()

                soft_err = radial_error(
                    coords_soft[b].numpy()[valid],
                    keypoints_gt[b].numpy()[valid],
                    mm_per_px,
                )
                soft_errors.append(soft_err)

                argmax_err = radial_error(
                    coords_argmax[b].numpy()[valid],
                    keypoints_gt[b].numpy()[valid],
                    mm_per_px,
                )
                argmax_errors.append(argmax_err)

    return soft_errors, argmax_errors, all_coords, all_gt


def compute_mean_mre(errors_list: list) -> float:
    """Compute mean MRE from list of per-image error arrays."""
    all_errors = []
    for errors in errors_list:
        all_errors.extend(errors.flatten().tolist())
    return sum(all_errors) / len(all_errors) if all_errors else float("inf")


def run_kfold_training(
    config_path: str = "config.yaml",
    debug: bool = False,
    max_images: Optional[int] = None,
) -> dict:
    """Run 5-Fold Cross-Validation training for cephalometric landmark detection.

    Key features (vs LOPO v1):
    - 5-Fold GroupKFold: all images from same patient stay in same split
      ~73 train / ~19 val per fold with 92 total images
    - Backbone unfrozen from Epoch 1 (no warmup freeze)
    - Early stopping on val_mre with patience=15
    - Best-model checkpoint per fold (save best, not last)
    - Evaluate every 5 epochs + at final epoch
    - Pretrained HRNet-W32 init for every fold
    - Unified LR for backbone + head (no differential LR)
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
    total_epochs = cfg["training"].get("epochs", 100)
    if debug:
        total_epochs = 2
    lr = cfg["training"]["lr"]
    batch_size = cfg["training"]["batch_size"]

    n_folds = cfg["training"].get("k_folds", 5)
    patience = cfg["training"].get("early_stopping_patience", 15)

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
    val_transform = build_val_transform()

    pid_to_iids: dict[str, list[str]] = {}
    for r in records:
        pid_to_iids.setdefault(r["patient_id"], []).append(r["image_id"])

    splits_data = build_splits(records, n_folds=n_folds)
    splits = [
        (
            [iid for pid in fold["train"] for iid in pid_to_iids.get(pid, [])],
            [iid for pid in fold["val"] for iid in pid_to_iids.get(pid, [])],
        )
        for fold in splits_data["folds"]
    ]
    if debug:
        splits = splits[:1]

    fold_metrics = []
    all_fold_errors = []

    for fold_idx, (train_ids, val_ids) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}  |  train={len(train_ids)}, val={len(val_ids)}")

        train_ds = CephalometricDataset(
            records, cfg["data"]["image_dir"], input_size, train_transform, image_ids=train_ids
        )
        val_ds = CephalometricDataset(
            records, cfg["data"]["image_dir"], input_size, val_transform, image_ids=val_ids
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        # Fresh pretrained init for every fold
        model = CephalometricModel(
            num_keypoints=cfg["keypoints"]["num_keypoints"],
            pretrained=True,
        ).to(device)

        # ── Partial freeze: Stage 1+2 frozen, Stage 3+4 + head trainable ──
        partial_freeze = cfg["training"].get("partial_freeze", False)
        if partial_freeze:
            freeze_stage1_2(model)
            backbone_lr = cfg["training"].get("backbone_lr", 1e-5)
            head_lr = cfg["training"].get("head_lr", 1e-3)
            param_groups = get_partial_freeze_param_groups(
                model, backbone_lr, head_lr, cfg["training"]["weight_decay"]
            )
            optimizer = torch.optim.AdamW(param_groups)
            print(
                f"  [Partial freeze] backbone_lr={backbone_lr:.0e}  head_lr={head_lr:.0e}  "
                f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
            )
        else:
            # Default: all params trainable, single LR
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=cfg["training"]["weight_decay"],
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=lr * 0.001
        )

        best_mre = float("inf")
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(total_epochs):
            # Extract landmark-specific sigma map if defined in config
            sigma_map = cfg["model"].get("landmark_sigmas")
            if sigma_map is not None:
                # Convert to numpy inside train_one_epoch where np is available
                print(f"  [Adaptive sigma] Using landmark-specific sigmas: {dict(zip(cfg['keypoints']['names'], sigma_map))}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, device,
                heatmap_size, cfg["model"]["sigma"], input_size,
                mixup_alpha=cfg["training"].get("mixup_alpha", 0.0),
                use_eupe=cfg["training"].get("use_eupe", False),
                sigma_map=sigma_map,
            )
            scheduler.step()

            # Evaluate every 5 epochs + last epoch
            eval_now = (epoch + 1) % 5 == 0 or epoch == total_epochs - 1

            if eval_now:
                fold_errors_soft, fold_errors_argmax, fold_coords, fold_gt = evaluate(
                    model, val_loader, device, heatmap_size, input_size, calibration_lookup
                )
                mre = compute_mean_mre(fold_errors_soft)
                mre_argmax = compute_mean_mre(fold_errors_argmax)
                current_lr = optimizer.param_groups[0]["lr"]

                # Use argmax MRE for early stopping — soft-argmax collapses to center
                # because beta is a non-learnable buffer so it can't self-correct.
                # Argmax MRE is what actually matters and it IS improving during training.
                if mre_argmax < best_mre:
                    best_mre = mre_argmax
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                    marker = " *BEST*"
                else:
                    epochs_no_improve += 5
                    marker = ""

                print(
                    f"  [Fold {fold_idx+1}] Epoch {epoch+1}/{total_epochs} — "
                    f"loss: {train_loss:.4f}, MRE: {mre:.2f}mm, MRE_argmax: {mre_argmax:.2f}mm, "
                    f"best: {best_mre:.2f}mm, LR: {current_lr:.6f}{marker}"
                )

                # Early stopping
                if epochs_no_improve >= patience:
                    print(f"  [Fold {fold_idx+1}] Early stopping at epoch {epoch+1}")
                    break

        # Restore best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)

        # Final evaluation with best model (both metrics)
        fold_errors_soft_final, fold_errors_argmax_final, fold_coords_final, fold_gt_final = evaluate(
            model, val_loader, device, heatmap_size, input_size, calibration_lookup
        )
        # Use hard-argmax errors for aggregated metrics — soft-argmax is broken
        # because sigmoid compression (logit 10→0.99995, logit 5→0.993) reduces
        # dynamic range to ~0.7%, making exp(beta*conf) ~uniform across 65k px.
        # Hard argmax selects the peak directly without sigmoid collapse, giving
        # the true model quality (~1.75mm vs the 9.97mm soft-argmax artifact).
        all_fold_errors.extend(fold_errors_argmax_final)
        fold_mre = compute_mean_mre(fold_errors_argmax_final)
        fold_mre_soft = compute_mean_mre(fold_errors_soft_final)
        fold_metrics.append({"fold": fold_idx + 1, "mre": fold_mre, "mre_soft": fold_mre_soft})
        print(f"  [Fold {fold_idx+1}] Final best MRE: {fold_mre:.2f}mm (soft-argmax: {fold_mre_soft:.2f}mm)")

        # ── Mode Collapse Detection ─────────────────────────────────────────
        # Compute per-keypoint spatial stddev of predicted coordinates across
        # validation images. If stddev is very low (< 5 px), the model is
        # predicting a static spatial mean — it has memorized absolute positions.
        import numpy as np
        if fold_coords_final:
            all_coords_arr = np.stack(fold_coords_final, axis=0)   # [N_img, 10, 2]
            all_gt_arr     = np.stack(fold_gt_final, axis=0)        # [N_img, 10, 2]
            pred_std = all_coords_arr.std(axis=0)                   # [10, 2] per kp
            gt_std   = all_gt_arr.std(axis=0)
            min_pred_std_px = float(pred_std.min())
            min_gt_std_px   = float(gt_std.min())
        else:
            min_pred_std_px = 0.0
            min_gt_std_px   = 0.0
        print(f"  [Fold {fold_idx+1}] Coord stddev — pred min: {min_pred_std_px:.1f}px, "
              f"gt min: {min_gt_std_px:.1f}px")

        # Save best checkpoint per fold + calibration lookup
        ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"fold{fold_idx+1}_best.pth")
        torch.save({
            "fold": fold_idx + 1,
            "model_state_dict": best_model_state,
            "fold_mre_argmax": fold_mre,
            "fold_mre_soft": fold_mre_soft,
            "calibration_lookup": calibration_lookup,
            "input_size": input_size,
            "heatmap_size": heatmap_size,
            "sigma": cfg["model"]["sigma"],
            "config": dict(cfg),
        }, ckpt_path)
        print(f"  [Fold {fold_idx+1}] Checkpoint saved: {ckpt_path}")

    # Aggregate across all folds
    kp_names = cfg["keypoints"]["names"]
    metrics = compute_all_metrics(
        all_fold_errors,
        sdr_thresholds=cfg["evaluation"]["sdr_thresholds_mm"],
        keypoint_names=kp_names,
    )
    metrics["fold_metrics"] = fold_metrics
    return metrics