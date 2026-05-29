#!/usr/bin/env python3
"""
TSK-04: Tversky + BoundaryDice Fine-Tuning on 512px baseline.

Fine-tune the champion 512px DeepLabV3+ (Dice=0.8588) using:
  - Tversky Loss (alpha=0.7, beta=0.3) — penalize FN 2.3x more than FP
  - BoundaryDice Loss — focus on bone edge accuracy
  - 50 epochs, patient-level train/val split, early stopping patience=10
  - GPU 4 (our assigned worker)

Usage:
  python scripts/train_tversky.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.phase2b.segmentation import POLYGON_CLASSES, NUM_SEG_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tversky")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGE_DIR = DATA_DIR / "raw" / "images"
LANDMARKS_JSON = DATA_DIR / "processed" / "landmarks_clean.json"
SEG_JSON = DATA_DIR / "processed" / "segmentation_train.json"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky Loss — generalisation of Dice that handles FP/FN asymmetry.

    alpha=0.7, beta=0.3:
      - FN (missed bone) penalised ~2.3x more than FP (false bone)
      - Forces model to avoid missing bone regions.
    tau=0.5 prevents division by zero.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, tau: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)

        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))

        numerator = (1 + self.tau) * tp
        denominator = numerator + self.alpha * fn + self.beta * fp + self.smooth
        score = numerator / denominator

        return 1 - score.mean()


class BoundaryDiceLoss(nn.Module):
    """
    Boundary Dice — computes Dice only on the boundary strip of each mask.

    boundary = dilate(mask, kernel=3) XOR erode(mask, kernel=3)
    Focuses training signal on bone edges rather than interior fill.
    """

    def __init__(self, kernel_size: int = 3, smooth: float = 1e-6):
        super().__init__()
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        self.smooth = smooth

    def _get_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Return binary boundary mask."""
        # Ensure uint8 for cv2 operations
        mask_u8 = (mask * 255).astype(np.uint8)
        dilated = cv2.dilate(mask_u8, self.kernel)
        eroded = cv2.erode(mask_u8, self.kernel)
        boundary = cv2.bitwise_xor(dilated, eroded)
        return boundary.astype(np.float32) / 255.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        losses = []

        for b in range(pred.shape[0]):
            for c in range(pred.shape[1]):
                p = pred[b, c].detach().cpu().numpy()
                t = target[b, c].cpu().numpy()

                boundary_t = self._get_boundary(t)
                if boundary_t.sum() == 0:
                    # No boundary in GT — ignore this channel
                    continue

                # Apply same boundary mask to prediction
                p_boundary = p * boundary_t

                intersection = (p_boundary * boundary_t).sum()
                size = boundary_t.sum()
                dice = (2 * intersection + self.smooth) / (p_boundary.sum() + size + self.smooth)
                losses.append(1 - dice)

        if not losses:
            return pred.sum() * 0.0  # zero loss if no boundaries found
        return torch.tensor(losses, device=pred.device, dtype=pred.dtype).mean()


class CombinedTverskyBoundaryLoss(nn.Module):
    """
    Combined loss: Tversky (global) + BoundaryDice (boundary-focused).
    Weight: 0.6 * Tversky + 0.4 * BoundaryDice
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.boundary = BoundaryDiceLoss(kernel_size=3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.6 * self.tversky(pred, target) + 0.4 * self.boundary(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET (same pattern as auto_research_segmentation.py)
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDatasetAuto(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_dir: Path,
        input_size: tuple[int, int] = (512, 512),
        transform=None,
    ):
        self.records = records
        self.image_dir = Path(image_dir)
        self.H, self.W = input_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        img_path = self.image_dir / rec["filename"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (self.W, self.H))
# Build masks with background channel (model outputs 4 classes)
        masks = np.zeros((NUM_SEG_CLASSES, self.H, self.W), dtype=np.float32)
        for ch_idx, cls in enumerate(POLYGON_CLASSES):
            if cls not in rec.get("polygons", {}):
                continue
            pts = np.array(rec["polygons"][cls], dtype=np.float32)
            pts[:, 0] *= self.W / orig_w
            pts[:, 1] *= self.H / orig_h
            canvas = np.zeros((self.H, self.W), dtype=np.uint8)
            cv2.fillPoly(canvas, [pts.astype(np.int32)], color=1)
            masks[ch_idx] = canvas.astype(np.float32)

        # Background = 1 - OR of all foreground masks
        foreground_mask = (masks.sum(axis=0) > 0).astype(np.float32)
        # Pad to 4 channels: [Background, Upper_incisor, Labial_bone, Palatal_bone]
        masks = np.concatenate([
            1 - foreground_mask[np.newaxis, :, :],  # background channel
            masks  # 3 class channels
        ], axis=0).astype(np.float32)

        if self.transform is not None:
            # masks now has 4 channels [bg, UI, LB, PB]
            aug_masks = [masks[c] for c in range(4)]
            result = self.transform(image=img_resized, masks=aug_masks)
            img_resized = result["image"]
            aug_masks = result["masks"]
            # aug_masks remain np arrays (ToTensorV2 only converts image)
            masks_tensor = torch.from_numpy(np.stack(aug_masks, axis=0))
        else:
            masks_tensor = torch.from_numpy(masks)

        # img_resized may be numpy (no transform) or torch.Tensor (after ToTensorV2)
        if isinstance(img_resized, np.ndarray):
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        else:
            img_tensor = img_resized.float() / 255.0 if img_resized.max() > 1.0 else img_resized.float()

        meta = {"image_id": rec["image_id"], "patient_id": rec["patient_id"]}
        return img_tensor, masks_tensor, meta


AUG_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussNoise(std_range=(0.01, 0.05), p=0.25),
    A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.9, 1.1), p=0.3),
    ToTensorV2(),
], additional_targets={f"mask{i}": "mask" for i in range(NUM_SEG_CLASSES - 1)})

AUG_VAL = A.Compose([ToTensorV2()], additional_targets={f"mask{i}": "mask" for i in range(NUM_SEG_CLASSES - 1)})


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT-LEVEL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def patient_split(records: list[dict], val_frac: float = 0.2, seed: int = 42):
    """Split records by patient_id so no patient leaks across train/val."""
    import random
    random.seed(seed)

    patients = list({r["patient_id"] for r in records})
    random.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])

    train_records = [r for r in records if r["patient_id"] not in val_patients]
    val_records   = [r for r in records if r["patient_id"] in val_patients]
    return train_records, val_records


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute mean Dice across classes."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    dice_scores = []
    smooth = 1e-6
    for c in range(pred.shape[1]):
        intersection = (pred_bin[:, c] * target[:, c]).sum()
        union = pred_bin[:, c].sum() + target[:, c].sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    return np.mean(dice_scores)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []

    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        dice = compute_dice(logits, masks)
        dice_scores.append(dice)

    return total_loss / len(loader), np.mean(dice_scores)


# ─────────────────────────────────────────────────────────────────────────────
# GIT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GIT_WORK_BRANCH = "optimize"


def git_current_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()


def git_create_branch(branch: str) -> None:
    subprocess.run(["git", "checkout", "-b", branch], check=True, capture_output=True)


def git_add_commit(message: str, files: list[str]) -> None:
    for f in files:
        subprocess.run(["git", "add", f], check=True)
    try:
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        log.warning("Git commit failed — continuing (non-blocking).")


def git_checkout(branch: str) -> None:
    subprocess.run(["git", "checkout", branch], check=True, capture_output=True)


def commit_model_checkpoint(model_name: str, dice_score: float, extra_files: list[str]) -> None:
    """Branch + commit + return to work branch."""
    branch = f"experiment/{model_name}-dice{int(dice_score*10000):04d}"
    result = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True)
    if result.stdout.strip():
        subprocess.run(["git", "stash"], capture_output=True)
    try:
        git_create_branch(branch)
        git_add_commit(
            f"perf: TSK-04 Tversky+BoundaryDice Dice={dice_score:.4f} {model_name}",
            extra_files,
        )
        log.info("Branch '%s' committed.", branch)
        git_checkout(GIT_WORK_BRANCH)
        result2 = subprocess.run(["git", "stash", "list"], text=True, capture_output=True)
        if result2.stdout.strip():
            subprocess.run(["git", "stash", "pop"], capture_output=True)
    except subprocess.CalledProcessError as exc:
        log.warning("Git operation failed (non-blocking): %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TSK-04: Tversky + BoundaryDice fine-tuning")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--gpu", type=int, default=4, help="GPU index")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    log.info("Using GPU %d", args.gpu)

    # ── Load data ──────────────────────────────────────────────────────────
    with open(SEG_JSON, "r") as f:
        records = json.load(f)

    # Keep only records with polygon data
    records_with_polys = [r for r in records if r.get("polygons")]
    log.info("Records with polygons: %d / %d", len(records_with_polys), len(records))

    train_records, val_records = patient_split(records_with_polys, val_frac=0.2)
    log.info("Train: %d records, Val: %d records", len(train_records), len(val_records))

    # ── Dataloaders ────────────────────────────────────────────────────────
    train_ds = SegmentationDatasetAuto(train_records, RAW_IMAGE_DIR, (512, 512), AUG_TRAIN)
    val_ds   = SegmentationDatasetAuto(val_records,   RAW_IMAGE_DIR, (512, 512), AUG_VAL)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    baseline_path = MODEL_OUTPUT_DIR / "exp0000_DeepLabV3Plus_resnet34_clahe1_20260527_075717" / "best_model.pt"
    log.info("Loading baseline: %s", baseline_path)

    model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=4)
    state = torch.load(baseline_path, map_location="cpu")
    # Handle both wrapped (DP) and unwrapped states
    if list(state.keys())[0].startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        log.info("Missing keys (will use random init for new head classes): %s", result.missing_keys[:3])
    if result.unexpected_keys:
        log.info("Unexpected keys (will be ignored): %s", result.unexpected_keys[:3])
    model = model.to(device)
    log.info("Model loaded: %d params", sum(p.numel() for p in model.parameters()))

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CombinedTverskyBoundaryLoss(alpha=0.7, beta=0.3)
    val_criterion = CombinedTverskyBoundaryLoss(alpha=0.7, beta=0.3)  # same loss for eval

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"tversky_deepLabV3plus_resnet34_20250529_{timestamp}"
    run_dir = MODEL_OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    patience_counter = 0

    log.info("=" * 60)
    log.info("TSK-04: Tversky+BoundaryDice Fine-Tuning")
    log.info("Run name: %s", run_name)
    log.info("Epochs: %d, Batch: %d, LR: %s, Patience: %d", args.epochs, args.batch_size, args.lr, args.patience)
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, val_criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        log.info(
            "epoch %3d/%3d | train_loss %.4f | val_loss %.4f | val_dice %.4f | elapsed %.1fs",
            epoch, args.epochs, train_loss, val_loss, val_dice, elapsed,
        )

        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            best_path = run_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            log.info("*** New best val_dice: %.4f ***", best_val_dice)
        else:
            patience_counter += 1

        # Early stop
        if patience_counter >= args.patience:
            log.info("Early stopping at epoch %d", epoch)
            break

    # ── Final checkpoint ───────────────────────────────────────────────────
    config = {
        "experiment_index": 99,
        "run_id": run_name,
        "arch_name": "DeepLabV3Plus",
        "encoder_name": "resnet34",
        "lr": args.lr,
        "weight_decay": 1e-3,
        "batch_size": args.batch_size,
        "loss": "Tversky(alpha=0.7,beta=0.3) + BoundaryDice (0.6+0.4)",
        "epochs_trained": epoch,
        "best_val_dice": float(best_val_dice),
        "epochs": args.epochs,
        "timestamp": datetime.now().isoformat(),
        "num_classes": NUM_SEG_CLASSES,
        "classes": POLYGON_CLASSES,
    }

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("=" * 60)
    log.info("Training complete. Best val_dice: %.4f", best_val_dice)
    log.info("Model: %s", run_dir / "best_model.pt")
    log.info("Config: %s", config_path)

    # Git branch on new best (vs baseline 0.8588)
    baseline_dice = 0.8588
    if best_val_dice > baseline_dice:
        log.info("New best! Committing to git branch...")
        commit_model_checkpoint(
            f"{run_name}",
            best_val_dice,
            [str(config_path)],
        )
    else:
        log.info("Val Dice %.4f <= baseline %.4f — no git commit.", best_val_dice, baseline_dice)

    return best_val_dice


if __name__ == "__main__":
    dice = main()
    sys.exit(0)