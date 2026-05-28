#!/usr/bin/env python3
"""
Phase 2C 1024x1024 Segmentation Model Sweep

Trains DeepLabV3+(ResNet50) then UNet++(ResNet50) sequentially.
Both models use:
  - IMAGE_SIZE = (1024, 1024)
  - 4-class segmentation (Background=0, Upper_Incisor=1, Labial=2, Palatal=3)
  - AMP (torch.cuda.amp.autocast + GradScaler)
  - Gradient Accumulation (effective batch size = 32)
  - EarlyStopping patience = 15
  - Max 100 epochs per model

GPU enforcement: ONLY GPUs 0, 1, 2, 3 are used.
External PIDs on GPU 5 and other GPUs are never touched.

Usage:
    python scripts/run_1024_sweep.py

Output:
    models/exp{run_id}_DeepLabV3Plus_resnet50_1024/  best_model.pt + config.json
    models/exp{run_id}_UnetPlusPlus_resnet50_1024/  best_model.pt + config.json
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

import albumentations as A
import cv2
import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cvat_parser import parse_all_cvat_batches

# ═══════════════════════════════════════════════════════════════════
# GPU ENFORCEMENT — STRICT
# ═══════════════════════════════════════════════════════════════════
ALLOWED_GPUS = [0, 1, 2, 3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in ALLOWED_GPUS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]
NUM_CLASSES     = len(POLYGON_CLASSES) + 1   # +1 Background
CLASS_TO_IDX    = {cls: i + 1 for i, cls in enumerate(POLYGON_CLASSES)}

IMAGE_SIZE      = (1024, 1024)
BATCH_SIZE_PER_GPU = 4          # safe for V100 32GB @ 1024px with AMP
ACCUMULATION_STEPS  = 2         # effective bs = 4 × 2(gpu) × 2(acc) = 16 ... wait 4×4×2=32
# effective bs = BATCH_SIZE_PER_GPU × GPU_COUNT × ACCUMULATION_STEPS
EFFECTIVE_BS   = BATCH_SIZE_PER_GPU * GPU_COUNT * ACCUMULATION_STEPS

LR          = 3e-4
WEIGHT_DECAY= 1e-3
DICE_WEIGHT = 0.5
MAX_EPOCHS  = 100
PATIENCE    = 15
SEED        = 42

DATA_DIR    = PROJECT_ROOT / "data"
RAW_DIR     = DATA_DIR / "raw" / "images"
ANNOTATIONS_DIR = DATA_DIR / "raw" / "annotations"
SEG_JSON    = DATA_DIR / "processed" / "segmentation_train.json"
MODEL_OUT   = PROJECT_ROOT / "models"

RUN_TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("p2c_sweep")


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════
def patient_split(records, val_pct=0.2, seed=42):
    random.seed(seed)
    pids = list({r["patient_id"] for r in records})
    random.shuffle(pids)
    n_val = max(1, int(len(pids) * val_pct))
    val_pids = set(pids[:n_val])
    return [r for r in records if r["patient_id"] not in val_pids], \
           [r for r in records if r["patient_id"] in val_pids]


class SegmentationDataset4Class(Dataset):
    def __init__(self, records, image_dir, input_size=(1024, 1024),
                 transform=None, apply_clahe=False):
        self.records    = records
        self.image_dir  = Path(image_dir)
        self.H = input_size[0]
        self.W = input_size[1]
        self.transform  = transform
        self.apply_clahe= apply_clahe

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(str(self.image_dir / rec["filename"]))
        oh, ow = img.shape[:2]
        img = cv2.resize(img, (self.W, self.H))
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            chs   = [clahe.apply(c) for c in cv2.split(img.astype(np.uint8))]
            img   = cv2.merge(chs)

        mask = np.zeros((self.H, self.W), dtype=np.int64)
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            if cls_name not in rec.get("polygons", {}):
                continue
            pts = np.array(rec["polygons"][cls_name], dtype=np.float32)
            pts[:, 0] *= self.W / ow
            pts[:, 1] *= self.H / oh
            canvas = np.zeros((self.H, self.W), dtype=np.uint8)
            cv2.fillPoly(canvas, [pts.astype(np.int32)], color=1)
            mask[canvas == 1] = cls_idx

        if self.transform:
            r = self.transform(image=img, mask=mask)
            img, mask = r["image"], r["mask"]

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = img.astype(np.float32) / 255.0
        img  = (img - mean) / std
        img  = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


# ═══════════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════════
def build_aug():
    t = {f"mask{i}": "mask" for i in range(NUM_CLASSES - 1)}
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.08), p=0.3),
        A.Affine(
            translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
            scale=(0.9, 1.1), rotate=(-8, 8), p=0.4,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        A.OpticalDistortion(distort_limit=0.1, p=0.15),
    ], additional_targets=t)


# ═══════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.dice_weight = dice_weight
        self.smooth      = 1e-6

    def forward(self, pred, target):
        ce   = self.ce(pred, target)
        dice = 0.0
        for c in range(1, NUM_CLASSES):
            p = F.softmax(pred, dim=1)[:, c].reshape(-1)
            t = (target == c).float().reshape(-1)
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice += 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return (1.0 - self.dice_weight) * ce + self.dice_weight * (dice / (NUM_CLASSES - 1))


def compute_dice(pred, target, nc):
    p    = torch.argmax(pred, dim=1)
    t    = target
    dice = []
    for c in range(nc):
        pcm = (p == c).cpu().float()
        tcm = (t == c).cpu().float()
        inter = (pcm * tcm).sum().item()
        union = pcm.sum().item() + tcm.sum().item() - inter
        dice.append(inter / (union + 1e-6))
    return float(np.mean(dice))


def compute_iou(pred, target, nc):
    p   = torch.argmax(pred, dim=1)
    t   = target
    ious = []
    for c in range(nc):
        pcm = (p == c).cpu().float()
        tcm = (t == c).cpu().float()
        inter = (pcm * tcm).sum().item()
        union = pcm.sum().item() + tcm.sum().item() - inter
        ious.append(inter / (union + 1e-6))
    return float(np.mean(ious))


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP with AMP + Gradient Accumulation
# ═══════════════════════════════════════════════════════════════════
def train_one_model(model, train_dl, val_dl, epochs, run_id, patience, accumulation_steps):
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = CrossEntropyDiceLoss(dice_weight=DICE_WEIGHT)
    scaler  = GradScaler()

    best_dice, patience_counter, best_state = 0.0, 0, None

    for ep in range(1, epochs + 1):
        model.train()
        t_loss  = 0.0
        n_steps = 0
        opt.zero_grad()

        for step, (imgs, masks) in enumerate(train_dl):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            with autocast():
                out   = model(imgs)
                loss  = loss_fn(out, masks)
                loss  = loss / accumulation_steps   # scale loss for accumulation

            scaler.scale(loss).backward()
            t_loss += loss.item() * accumulation_steps
            n_steps += 1

            if (step + 1) % accumulation_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

        # handle remaining steps
        if (step + 1) % accumulation_steps != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        sched.step()

        # Validate
        model.eval()
        dice_list, iou_list = [], []
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs  = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                with autocast():
                    out = model(imgs)
                dice_list.append(compute_dice(out, masks, NUM_CLASSES))
                iou_list.append(compute_iou(out, masks, NUM_CLASSES))

        dice = float(np.mean(dice_list))
        iou  = float(np.mean(iou_list))
        avg_loss = t_loss / n_steps

        log.info(
            "  Ep %d/%d | loss=%.4f | dice=%.4f | iou=%.4f | lr=%.2e",
            ep, epochs, avg_loss, dice, iou, opt.param_groups[0]["lr"],
        )

        if dice > best_dice:
            best_dice      = dice
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log.info("  Early stop @ ep %d (patience %d)", ep, patience)
            break

    model.load_state_dict(best_state)
    return model, best_dice, iou


# ═══════════════════════════════════════════════════════════════════
# BUILD MODEL (wraps in DataParallel)
# ═══════════════════════════════════════════════════════════════════
def build_model(arch_name):
    if arch_name == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        )
    elif arch_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        )
    else:
        raise ValueError(f"Unknown arch: {arch_name}")

    if GPU_COUNT > 1:
        model = nn.DataParallel(model)
    model.train()   # ensure batchnorm sees >1 sample per GPU
    return model.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 68)
    log.info("PHASE 2C  SWEEP  —  1024x1024  |  DeepLabV3+  vs  U-Net++")
    log.info("=" * 68)
    log.info("GPUs: %d (strict: %s) | Eff.bs=%d | img=%s | amp=ON | grad_acc=%d",
             GPU_COUNT, ALLOWED_GPUS, EFFECTIVE_BS, IMAGE_SIZE, ACCUMULATION_STEPS)

    # ── Load & split data ─────────────────────────────────────────
    log.info("Loading annotations via parse_all_cvat_batches ...")
    all_records = parse_all_cvat_batches(ANNOTATIONS_DIR)
    seg_records = [r for r in all_records if r["polygons"]]
    train_r, val_r = patient_split(seg_records, val_pct=0.2, seed=SEED)
    log.info("Train: %d | Val: %d | Total polygons: %d",
             len(train_r), len(val_r), len(seg_records))

    # Write merged segmentation JSON for traceability
    SEG_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SEG_JSON, "w") as f:
        json.dump(seg_records, f)
    log.info("Written merged segmentation JSON: %s", SEG_JSON)

    # ── Dataloaders ─────────────────────────────────────────────────
    train_ds = SegmentationDataset4Class(train_r, RAW_DIR, IMAGE_SIZE, build_aug(), apply_clahe=True)
    val_ds   = SegmentationDataset4Class(val_r,   RAW_DIR, IMAGE_SIZE, None,             apply_clahe=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_PER_GPU * GPU_COUNT,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_PER_GPU * GPU_COUNT,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    arch_list = ["DeepLabV3Plus", "UnetPlusPlus"]

    for arch in arch_list:
        log.info("=" * 68)
        log.info("  TRAINING: %s (resnet50) @ %s", arch, IMAGE_SIZE)
        log.info("=" * 68)

        run_id = f"exp{RUN_TS}_{arch}_resnet50_1024"
        ckpt_dir = MODEL_OUT / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(arch)
        log.info("Model built: %s | Params: %.1fM",
                 arch,
                 sum(p.numel() for p in model.parameters()) / 1e6)

        t0 = time.time()
        model, best_dice, best_iou = train_one_model(
            model, train_dl, val_dl,
            epochs=MAX_EPOCHS,
            run_id=run_id,
            patience=PATIENCE,
            accumulation_steps=ACCUMULATION_STEPS,
        )
        elapsed = time.time() - t0

        torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

        cfg = {
            "experiment_type": "phase2c_1024_sweep",
            "run_id": run_id,
            "arch_name": arch,
            "encoder_name": "resnet50",
            "image_size": list(IMAGE_SIZE),
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "effective_batch_size": EFFECTIVE_BS,
            "dice_weight": DICE_WEIGHT,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "use_clahe": True,
            "augmentation": "heavy",
            "val_dice": best_dice,
            "val_iou": best_iou,
            "num_classes": NUM_CLASSES,
            "classes": ["Background"] + POLYGON_CLASSES,
            "n_train": len(train_r),
            "n_val": len(val_r),
            "training_time_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        log.info(
            "  %s DONE | dice=%.4f | iou=%.4f | time=%.0fs | saved: %s",
            arch, best_dice, best_iou, elapsed, ckpt_dir / "best_model.pt",
        )
        log.info("")

        # free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    log.info("=" * 68)
    log.info("  PHASE 2C SWEEP COMPLETE — both models trained.")
    log.info("=" * 68)
    log.info(
        "  DeepLabV3Plus & UnetPlusPlus results saved under models/exp%s_*_1024/",
        RUN_TS,
    )


if __name__ == "__main__":
    main()
