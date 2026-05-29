#!/usr/bin/env python3
"""
Final Deep Run — 150-Epoch Single-Config Training
Config: DeepLabV3Plus(resnet34) | lr=0.0003 | wd=0.001 | bs=16 (4/GPU x4)
        CLAHE=True | Augmentation=heavy | num_classes=4
        EarlyStopping patience=25

On completion: auto-runs evaluate_and_plot.py on Patient01_T1.jpg
"""
from __future__ import annotations
import argparse, json, logging, os, random, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

# ─── CONFIG ──────────────────────────────────────────────────────────────────
POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]
NUM_CLASSES    = len(POLYGON_CLASSES) + 1  # +1 Background
CLASS_TO_IDX   = {cls: i+1 for i, cls in enumerate(POLYGON_CLASSES)}
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw" / "images"
SEG_JSON       = DATA_DIR / "processed" / "segmentation_train.json"
MODEL_OUT      = PROJECT_ROOT / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("final_deep")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
log.info("Device: %s | GPUs: %d", DEVICE, GPU_COUNT)

# ─── BEST CONFIG ─────────────────────────────────────────────────────────────
BEST_ARCH   = "DeepLabV3Plus"
BEST_ENC    = "resnet34"
BEST_LR     = 0.0003
BEST_WD     = 0.001
BEST_BS     = 4          # per GPU
BEST_AUG    = "heavy"
BEST_CLAHE  = True
BEST_DW     = 0.5
BEST_EPOCHS = 150
PATIENCE    = 25

# ─── CLAHE ─────────────────────────────────────────────────────────────────────
def apply_clahe(img: np.ndarray, clip_limit=2.0, tile_grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    chs   = [clahe.apply(c) for c in cv2.split(img.astype(np.uint8))]
    return cv2.merge(chs)

# ─── AUGMENTATION ─────────────────────────────────────────────────────────────
def build_aug(name):
    t = {f"mask{i}": "mask" for i in range(NUM_CLASSES - 1)}
    if name == "heavy":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.6),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(std_range=(0.01, 0.08), p=0.3),
            A.Affine(translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                   scale=(0.9, 1.1), rotate=(-8, 8), p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
            A.OpticalDistortion(distort_limit=0.1, p=0.15),
        ], additional_targets=t)
    raise ValueError(f"Unknown aug: {name}")

# ─── DATASET ───────────────────────────────────────────────────────────────────
class SegmentationDataset4Class(Dataset):
    def __init__(self, records, image_dir, input_size=(512, 512), transform=None, apply_clahe=False):
        self.records, self.image_dir, self.H, self.W = records, Path(image_dir), *input_size
        self.transform, self.apply_clahe = transform, apply_clahe

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        img  = cv2.imread(str(self.image_dir / rec["filename"]))
        oh, ow = img.shape[:2]
        img = cv2.resize(img, (self.W, self.H))
        if self.apply_clahe: img = apply_clahe(img)
        mask = np.zeros((self.H, self.W), dtype=np.int64)
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            if cls_name not in rec.get("polygons", {}): continue
            pts = np.array(rec["polygons"][cls_name], dtype=np.float32)
            pts[:, 0] *= self.W / ow; pts[:, 1] *= self.H / oh
            canvas = np.zeros((self.H, self.W), dtype=np.uint8)
            cv2.fillPoly(canvas, [pts.astype(np.int32)], color=1)
            mask[canvas == 1] = cls_idx

        if self.transform:
            r = self.transform(image=img, mask=mask)
            img, mask = r["image"], r["mask"]

        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return img, mask

# ─── MODEL ─────────────────────────────────────────────────────────────────────
def build_model():
    base = smp.DeepLabV3Plus(
        encoder_name=BEST_ENC, encoder_weights="imagenet",
        in_channels=3, classes=NUM_CLASSES, activation=None,
    )
    if GPU_COUNT > 1:
        base = nn.DataParallel(base)
    return base.to(DEVICE)

# ─── LOSS (verified working — matches auto_research_iter1.py) ──────────────────
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.dice_weight = dice_weight
        self.smooth = 1e-6

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
    p, t = torch.argmax(pred, dim=1), target
    scores = []
    for c in range(nc):
        pcm = (p == c).cpu().float()
        tcm = (t == c).cpu().float()
        inter = (pcm * tcm).sum().item()
        union = pcm.sum().item() + tcm.sum().item() - inter
        scores.append(inter / (union + 1e-6))
    return float(np.mean(scores))

# ─── SPLIT ─────────────────────────────────────────────────────────────────────────
def patient_split(records, val_pct=0.2, seed=42):
    random.seed(seed)
    pids = list({r["patient_id"] for r in records})
    random.shuffle(pids)
    n_val = max(1, int(len(pids) * val_pct))
    val_pids = set(pids[:n_val])
    return [r for r in records if r["patient_id"] not in val_pids], [r for r in records if r["patient_id"] in val_pids]

# ─── TRAIN ─────────────────────────────────────────────────────────────────────
def train(epochs, train_r, val_r):
    model = build_model()
    opt   = torch.optim.AdamW(model.parameters(), lr=BEST_LR, weight_decay=BEST_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = CrossEntropyDiceLoss(dice_weight=BEST_DW)

    bs_total = BEST_BS * GPU_COUNT
    train_ds = SegmentationDataset4Class(train_r, RAW_DIR, (512, 512), build_aug(BEST_AUG), BEST_CLAHE)
    val_ds   = SegmentationDataset4Class(val_r,   RAW_DIR, (512, 512), None,           BEST_CLAHE)
    train_dl = DataLoader(train_ds, batch_size=bs_total, shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs_total, shuffle=False, num_workers=0, pin_memory=True)

    best_dice, patience_counter, best_iou = 0.0, 0, 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for imgs, masks in train_dl:
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = loss_fn(out, masks)
            loss.backward()
            opt.step()
            t_loss += loss.item()
        sched.step()

        # Validate
        model.eval()
        dice_list, iou_list = [], []
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs  = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                out   = model(imgs)
                dice_list.append(compute_dice(out, masks, NUM_CLASSES))
                iou_list.append(compute_iou(out, masks, NUM_CLASSES))
        dice = np.mean(dice_list)
        iou  = np.mean(iou_list)

        log.info("  Ep %d/%d | dice=%.4f | iou=%.4f | lr=%.2e",
                 ep, epochs, dice, iou, opt.param_groups[0]["lr"])

        if dice > best_dice:
            best_dice = dice
            best_iou  = iou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            log.info("  Early stop @ ep %d (patience %d)", ep, PATIENCE)
            break

    model.load_state_dict(best_state)
    return model, best_dice, best_iou

def compute_iou(pred, target, nc):
    p, t = torch.argmax(pred, dim=1), target
    scores = []
    for c in range(nc):
        pcm = (p == c).cpu().float()
        tcm = (t == c).cpu().float()
        inter = (pcm * tcm).sum().item()
        union = pcm.sum().item() + tcm.sum().item() - inter
        scores.append(inter / (union + 1e-6))
    return float(np.mean(scores))

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"finalDeepRun_deeplabv3plus_{BEST_ENC}_lr{BEST_LR}_wd{BEST_WD}_bs{BEST_BS}x{GPU_COUNT}_150ep_{ts}"
    ckpt_dir = MODEL_OUT / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(SEG_JSON.read_text())
    train_r, val_r = patient_split(records, 0.2, 42)
    log.info("Train: %d | Val: %d | Classes: %d", len(train_r), len(val_r), NUM_CLASSES)
    log.info("Config: %s(%s) lr=%.0e wd=%.0e bs=%d aug=%s clahe=%s dice_w=%.1f epochs=%d patience=%d",
             BEST_ARCH, BEST_ENC, BEST_LR, BEST_WD, BEST_BS, BEST_AUG, BEST_CLAHE, BEST_DW, epochs, PATIENCE)

    log.info("=" * 65)
    log.info("  FINAL DEEP RUN — 150 EPOCH DEEPLABV3PLUS 4-CLASS")
    log.info("  GPUs: %d | Batch: %d total", GPU_COUNT, BEST_BS * GPU_COUNT)
    log.info("=" * 65)

    model, best_dice, best_iou = train(BEST_EPOCHS, train_r, val_r)

    log.info("FINAL RESULT: Dice=%.4f | IoU=%.4f", best_dice, best_iou)
    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    cfg = {
        "experiment_index": -1,
        "run_id": run_id,
        "arch_name": BEST_ARCH,
        "encoder_name": BEST_ENC,
        "lr": BEST_LR,
        "weight_decay": BEST_WD,
        "batch_size": BEST_BS,
        "aug_name": BEST_AUG,
        "use_clahe": BEST_CLAHE,
        "loss_dice_weight": BEST_DW,
        "epochs_trained": BEST_EPOCHS,
        "patience": PATIENCE,
        "val_dice": best_dice,
        "val_iou": best_iou,
        "num_classes": NUM_CLASSES,
        "classes": ["Background"] + POLYGON_CLASSES,
        "timestamp": datetime.now().isoformat(),
    }
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    log.info("Model saved: %s", ckpt_dir / "best_model.pt")

    # ── Auto-visualize on completion ─────────────────────────────────────────
    log.info("Auto-running evaluate_and_plot.py ...")
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_and_plot.py",
         "--model-dir", str(ckpt_dir),
         "--image", "Patient01_T1.jpg"],
        capture_output=True, text=True, timeout=120,
    )
    for line in result.stdout.splitlines():
        if "Saved visualization" in line or "DONE" in line:
            log.info("  %s", line.strip())
    if result.returncode != 0:
        log.error("  evaluate_and_plot.py failed: %s", result.stderr[-500:])

    log.info("COMPLETE. Best Dice: %.4f | IoU: %.4f", best_dice, best_iou)

if __name__ == "__main__":
    epochs = BEST_EPOCHS  # used in train() closure
    main()
