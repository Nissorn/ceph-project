#!/usr/bin/env python3
"""
Infinite Auto-Research Loop for Phase 2B — Alveolar Bone Segmentation.

Architecture search: UNet, DeepLabV3+, Attention UNet (SMP encoders)
Hyperparameters: LR, weight_decay, batch_size, augmentations
Exit: Manual Ctrl-C / SIGINT only.

Automated Git pipeline:
  - Track best validation Dice score
  - When new high score is reached: auto-branch, commit, return to main branch

Usage:
    python scripts/auto_research_segmentation.py
    python scripts/auto_research_segmentation.py --epochs-per-run 10 --max-train-images 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
import signal
import os
from datetime import datetime
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.phase2b.segmentation import POLYGON_CLASSES, NUM_SEG_CLASSES, SegmentationLoss

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGE_DIR = DATA_DIR / "raw" / "images"   # 381 images live here
LANDMARKS_JSON = DATA_DIR / "processed" / "landmarks_clean.json"
SEG_JSON = DATA_DIR / "processed" / "segmentation_train.json"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_research")

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER GRID
# ─────────────────────────────────────────────────────────────────────────────

ARCHITECTURES = [
    # (smp_factory, arch_name, encoder_name)
    (smp.Unet,               "Unet",              "resnet34"),
    (smp.Unet,               "Unet",              "efficientnet-b4"),
    (smp.DeepLabV3Plus,      "DeepLabV3Plus",     "resnet34"),
    (smp.Unet,               "AttentionUnet",     "resnet34"),
    (smp.Linknet,            "Linknet",           "resnet34"),
]

AUG_PRESETS = {
    "light": A.Compose([
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ToTensorV2(),
    ], additional_targets={f"mask{i}": "mask" for i in range(NUM_SEG_CLASSES - 1)}),
    "medium": A.Compose([
        A.HorizontalFlip(p=0.4),
        A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.25),
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.9, 1.1), p=0.3),
        ToTensorV2(),
    ], additional_targets={f"mask{i}": "mask" for i in range(NUM_SEG_CLASSES - 1)}),
    "heavy": A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.08), p=0.3),
        A.Affine(translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)}, scale=(0.88, 1.12), rotate=(-5, 5), p=0.4),
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        ToTensorV2(),
    ], additional_targets={f"mask{i}": "mask" for i in range(NUM_SEG_CLASSES - 1)}),
}

LOSS_CONFIGS = [
    ("dice_bce_50_50", 0.5, 0.5),
    ("dice_bce_70_30", 0.7, 0.3),
    ("dice_only",      1.0, 0.0),
]

LR_CANDIDATES     = [1e-4, 3e-4, 1e-3]
WD_CANDIDATES     = [1e-4, 5e-4, 1e-3]
BATCH_SIZE_CANDS  = [4, 8]
EPOCHS_PER_RUN    = 8
PATIENCE          = 3

# ─────────────────────────────────────────────────────────────────────────────
# GIT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GIT_WORK_BRANCH = "optimize"   # main working branch — loop stays here unless committing


def git_current_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()


def git_checkout(branch: str) -> None:
    subprocess.run(["git", "checkout", branch], check=True, capture_output=True)


def git_create_branch(branch_name: str) -> None:
    subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)


def git_add_commit(message: str, files: list[str]) -> None:
    for f in files:
        subprocess.run(["git", "add", f], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)


def ensure_clean_state() -> None:
    """Stash any uncommitted changes so we don't interfere with the loop."""
    result = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True)
    if result.stdout.strip():
        log.info("Uncommitted changes detected — stashing before branch operation.")
        subprocess.run(["git", "stash"], check=True, capture_output=True)


def commit_model_checkpoint(model_name: str, dice_score: float, extra_files: list[str]) -> None:
    """Create experiment branch, commit config+metrics+weights, return to work branch."""
    branch = f"experiment/{model_name}-dice{int(dice_score*100):03d}"
    ensure_clean_state()

    log.info("New best Dice %.4f — creating branch: %s", dice_score, branch)
    try:
        git_create_branch(branch)
        git_add_commit(
            f"perf: achieve new best validation dice score {dice_score:.4f} using {model_name}",
            extra_files,
        )
        log.info("Branch '%s' committed. Checking back to %s.", branch, GIT_WORK_BRANCH)
        git_checkout(GIT_WORK_BRANCH)
        # pop stash if we had one
        result = subprocess.run(["git", "stash", "list"], text=True, capture_output=True)
        if result.stdout.strip():
            subprocess.run(["git", "stash", "pop"], capture_output=True)
    except subprocess.CalledProcessError as exc:
        log.error("Git operation failed: %s", exc)
        log.warning("Continuing loop — not a training error.")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDatasetAuto(Dataset):
    """
    Reads records from segmentation_train.json and rasterises CVAT polygons
    into binary masks at load time.

    Compatible with the augmentation pipeline: passes masks as additional_targets.
    """
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

        if self.transform is not None:
            aug_masks = [masks[c] for c in range(NUM_SEG_CLASSES)]
            result = self.transform(image=img_resized, masks=aug_masks)
            img_resized = result["image"]
            aug_masks = result["masks"]
            masks = np.stack(aug_masks, axis=0)

        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        masks_tensor = torch.from_numpy(masks)

        meta = {"image_id": rec["image_id"], "patient_id": rec["patient_id"]}
        return img_tensor, masks_tensor, meta


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def build_aug(aug_name: str) -> A.Compose:
    """Return a pre-built albumentations transform. Adds mask targets automatically."""
    base = AUG_PRESETS.get(aug_name)
    if base is None:
        raise ValueError(f"Unknown augmentation preset: {aug_name}")
    # Clone so we don't mutate shared definitions
    targets = {f"mask{i}": "mask" for i in range(NUM_SEG_CLASSES - 1)}
    return A.Compose(base.transforms if hasattr(base, 'transforms') else [], additional_targets=targets)


def normalize_for_model(img: np.ndarray) -> torch.Tensor:
    """Apply ImageNet normalisation, then ToTensorV2."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    # Albumentations ToTensorV2 already does HWC→CHW and /255
    # We handle /255 above, then just transpose
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute mean Dice score across classes. pred is sigmoid-activated logits."""
    pred_sig = torch.sigmoid(pred)
    dice_per_class = []
    for c in range(pred.shape[1]):
        p = pred_sig[:, c].reshape(-1)
        t = target[:, c].reshape(-1)
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice_per_class.append((2.0 * intersection + smooth) / (union + smooth))
    return float(np.mean(dice_per_class))


def evaluate(model, dataloader, device) -> tuple[float, float]:
    """Return (dice, iou) on validation set."""
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks  = masks.to(device)
            pred   = model(images)
            dice   = compute_dice(pred, masks)
            dice_scores.append(dice)
            # IoU per class
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            for c in range(pred.shape[1]):
                p = pred_bin[:, c].reshape(-1).cpu().numpy()
                t = masks[:, c].reshape(-1).cpu().numpy()
                intersection = np.sum(p * t)
                union = np.sum(p) + np.sum(t) - intersection
                iou_scores.append(intersection / (union + 1e-6))
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_segmentation(
    arch_fn,
    arch_name: str,
    encoder_name: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    aug_name: str,
    loss_alpha: float,
    loss_beta: float,
    epochs: int,
    train_records: list[dict],
    val_records: list[dict],
    image_dir: Path,
    patience: int = PATIENCE,
) -> tuple[nn.Module, float, float, float]:
    """
    Train one configuration. Returns (best_model, best_dice, best_iou, elapsed_sec).
    """
    # Build model
    if arch_name == "AttentionUnet":
        model = smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet",
                         in_channels=3, classes=NUM_SEG_CLASSES, activation=None,
                         attention_mode={"attention_mode": "scse"})
    else:
        model = arch_fn(encoder_name=encoder_name, encoder_weights="imagenet",
                        in_channels=3, classes=NUM_SEG_CLASSES, activation=None)
    model = model.to(DEVICE)

    # Datasets
    train_ds = SegmentationDatasetAuto(train_records, image_dir, (512, 512), transform=None)
    val_ds   = SegmentationDatasetAuto(val_records,   image_dir, (512, 512), transform=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Loss
    class CombinedLoss(nn.Module):
        def __init__(self, alpha, beta):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.bce = nn.BCEWithLogitsLoss()
        def forward(self, pred, target):
            bce_loss = self.bce(pred, target)
            pred_sig = torch.sigmoid(pred)
            intersection = (pred_sig * target).sum(dim=(2, 3))
            union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice_loss = (1.0 - (2*intersection+1e-6)/(union+1e-6)).mean()
            return self.alpha * bce_loss + self.beta * dice_loss

    criterion = CombinedLoss(loss_alpha, loss_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_dice = 0.0
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_iou  = 0.0
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, masks, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 20 == 0:
                log.debug("  Epoch %d | Batch %d | loss=%.4f", epoch, batch_idx, loss.item())

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validate ──
        val_dice, val_iou = evaluate(model, val_loader, DEVICE)

        log.info(
            "  Epoch %d/%d | train_loss=%.4f | val_dice=%.4f | val_iou=%.4f | lr=%.6f",
            epoch, epochs, avg_train_loss, val_dice, val_iou,
            optimizer.param_groups[0]["lr"],
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_iou  = val_iou
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            log.info("  ★ New best dice: %.4f", best_dice)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  → Early stopping at epoch %d (no improvement for %d rounds)", epoch, patience)
                break

    elapsed = time.time() - start_time
    model.load_state_dict(best_model_state)
    model.to(DEVICE)

    return model, best_dice, best_iou, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def patient_split(records: list[dict], val_pct: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split records by patient_id (ensures T1+T2 of same patient stay together)."""
    import random
    random.seed(seed)
    patient_ids = list({r["patient_id"] for r in records})
    random.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * val_pct))
    val_patients = set(patient_ids[:n_val])
    train = [r for r in records if r["patient_id"] not in val_patients]
    val   = [r for r in records if r["patient_id"] in val_patients]
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    arch_fn, arch_name, encoder_name,
    lr, weight_decay, batch_size, aug_name,
    loss_alpha, loss_beta,
    train_records, val_records,
    image_dir,
    experiment_index: int,
):
    """Train one config, return result dict."""
    config_str = (
        f"{arch_name}({encoder_name}) lr={lr} wd={weight_decay} "
        f"bs={batch_size} aug={aug_name} loss=α{round(loss_alpha,1)}_β{round(loss_beta,1)}"
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"exp{experiment_index:04d}_{arch_name}_{encoder_name}_{ts}"

    log.info("")
    log.info("═══════════════════════════════════════════════════════")
    log.info("EXPERIMENT #%d: %s", experiment_index, config_str)
    log.info("═══════════════════════════════════════════════════════")

    model, dice, iou, elapsed = train_segmentation(
        arch_fn=arch_fn, arch_name=arch_name, encoder_name=encoder_name,
        lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        aug_name=aug_name, loss_alpha=loss_alpha, loss_beta=loss_beta,
        epochs=EPOCHS_PER_RUN,
        train_records=train_records, val_records=val_records,
        image_dir=image_dir,
    )

    log.info("  Result: Dice=%.4f | IoU=%.4f | Time=%.0fs", dice, iou, elapsed)

    # Save checkpoint
    ckpt_dir = MODEL_OUTPUT_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Save config
    config_path = ckpt_dir / "config.json"
    config = {
        "experiment_index": experiment_index,
        "run_id": run_id,
        "arch_name": arch_name,
        "encoder_name": encoder_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "aug_name": aug_name,
        "loss_alpha": loss_alpha,
        "loss_beta": loss_beta,
        "epochs": EPOCHS_PER_RUN,
        "val_dice": dice,
        "val_iou": iou,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return {"run_id": run_id, "config": config, "dice": dice, "iou": iou, "model": model, "ckpt_path": ckpt_path}


# ─────────────────────────────────────────────────────────────────────────────
# INFINITE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def build_grid():
    """Cartesian product of all hyperparameter combinations."""
    import itertools
    combos = []
    for arch_fn, arch_name, encoder_name in ARCHITECTURES:
        for lr in LR_CANDIDATES:
            for wd in WD_CANDIDATES:
                for bs in BATCH_SIZE_CANDS:
                    for aug_name in AUG_PRESETS:
                        for loss_name, loss_alpha, loss_beta in LOSS_CONFIGS:
                            combos.append((arch_fn, arch_name, encoder_name, lr, wd, bs, aug_name, loss_alpha, loss_beta))
    return combos


def main():
    global EPOCHS_PER_RUN
    parser = argparse.ArgumentParser(description="Infinite Auto-Research for Phase 2B Segmentation")
    parser.add_argument("--epochs-per-run", type=int, default=EPOCHS_PER_RUN, help="Epochs per training run")
    parser.add_argument("--max-train-images", type=int, default=None, help="Cap training images for fast debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    EPOCHS_PER_RUN = args.epochs_per_run

    print("=" * 65)
    print("  PHASE 2B — INFINITE AUTO-RESEARCH LOOP")
    print("  Press Ctrl-C to stop at any time")
    print("=" * 65)

    # ── Load data ───────────────────────────────────────────────────
    if not SEG_JSON.exists():
        log.error("segmentation_train.json not found at %s — run scripts/merge_cvat_data.py first.", SEG_JSON)
        sys.exit(1)

    with open(SEG_JSON) as f:
        all_records = json.load(f)

    log.info("Loaded %d segmentation records", len(all_records))

    if args.max_train_images:
        all_records = all_records[:args.max_train_images]
        log.info("Capped to %d records for fast debugging", len(all_records))

    train_records, val_records = patient_split(all_records, val_pct=0.2, seed=args.seed)
    log.info("Train: %d | Val: %d (patient-level split, seed=%d)", len(train_records), len(val_records), args.seed)

    if not RAW_IMAGE_DIR.exists():
        log.error("Image directory not found: %s", RAW_IMAGE_DIR)
        sys.exit(1)

    # ── Grid ────────────────────────────────────────────────────────
    grid = build_grid()
    log.info("Hyperparameter grid: %d combinations", len(grid))

    # Shuffle grid so we don't always run in same order
    random.seed(args.seed)
    random.shuffle(grid)

    best_dice_ever = 0.0
    experiment_index = 0
    experiment_iter = iter(grid)

    log.info("Starting infinite loop. Best Dice tracker initialised at 0.0")
    log.info("=" * 65)

    try:
        while True:
            try:
                arch_fn, arch_name, encoder_name, lr, wd, bs, aug_name, loss_alpha, loss_beta = next(experiment_iter)
            except StopIteration:
                # Grid exhausted — reshuffle and restart
                log.info("Grid exhausted. Reshuffling and restarting.")
                random.shuffle(grid)
                experiment_iter = iter(grid)
                continue

            result = run_experiment(
                arch_fn=arch_fn, arch_name=arch_name, encoder_name=encoder_name,
                lr=lr, weight_decay=wd, batch_size=bs,
                aug_name=aug_name, loss_alpha=loss_alpha, loss_beta=loss_beta,
                train_records=train_records, val_records=val_records,
                image_dir=RAW_IMAGE_DIR,
                experiment_index=experiment_index,
            )

            dice = result["dice"]
            iou  = result["iou"]
            run_id = result["run_id"]

            # ── Auto-Git Pipeline ──────────────────────────────────
            if dice > best_dice_ever:
                old_best = best_dice_ever
                best_dice_ever = dice
                model_name = f"{arch_name}_{encoder_name}"
                log.info("  ★★★ NEW BEST DICE: %.4f (was %.4f) — triggering auto-git ★★★", dice, old_best)

                commit_model_checkpoint(
                    model_name=model_name,
                    dice_score=dice,
                    extra_files=[
                        # Model .pt weights are gitignored — do NOT add them
                        str(MODEL_OUTPUT_DIR / run_id / "config.json"),
                        str(PROJECT_ROOT / "config.yaml"),
                    ],
                )
                log.info("  Git pipeline complete. Loop continuing...")

            experiment_index += 1

    except KeyboardInterrupt:
        log.info("")
        log.info("═══════════════════════════════════════════════════════")
        log.info("  KeyboardInterrupt received.")
        log.info("  Auto-research loop stopped.")
        log.info("  Experiments run: %d | Best Dice ever: %.4f", experiment_index, best_dice_ever)
        log.info("═══════════════════════════════════════════════════════")
        return


if __name__ == "__main__":
    main()