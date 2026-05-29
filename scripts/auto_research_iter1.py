#!/usr/bin/env python3
"""
Infinite Auto-Research Loop for Phase 2B — Alveolar Bone Segmentation.
Iteration 2: Multi-GPU DataParallel + Background class + CLAHE.
"""
from __future__ import annotations
import argparse, json, logging, os, random, subprocess, sys, time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

# ─── CONFIG ───────────────────────────────────────────────────────────────────
POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]
NUM_CLASSES = len(POLYGON_CLASSES) + 1  # +1 for Background
CLASS_TO_IDX = {cls: i+1 for i, cls in enumerate(POLYGON_CLASSES)}

DATA_DIR   = PROJECT_ROOT / "data"
RAW_DIR    = DATA_DIR / "raw" / "images"
SEG_JSON  = DATA_DIR / "processed" / "segmentation_train.json"
MODEL_OUT  = PROJECT_ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("autoresearch")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
log.info("Device: %s | GPUs: %d", DEVICE, GPU_COUNT)

# ─── CLAHE ───────────────────────────────────────────────────────────────────────
def apply_clahe(img: np.ndarray, clip_limit=2.0, tile_grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    chs = [clahe.apply(c) for c in cv2.split(img.astype(np.uint8))]
    return cv2.merge(chs)

# ─── AUGMENTATION ──────────────────────────────────────────────────────────────
def build_aug(name):
    t = {f"mask{i}": "mask" for i in range(NUM_CLASSES - 1)}
    if name == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.3), A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3)], additional_targets=t)
    elif name == "medium":
        return A.Compose([
            A.HorizontalFlip(p=0.4), A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.4), A.GaussNoise(std_range=(0.01, 0.05), p=0.25),
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.9, 1.1), p=0.3)],
            additional_targets=t)
    elif name == "heavy":
        return A.Compose([
            A.HorizontalFlip(p=0.5), A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.6),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5), A.GaussNoise(std_range=(0.01, 0.08), p=0.3),
            A.Affine(translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)}, scale=(0.88, 1.12),
                     rotate=(-5, 5), p=0.4), A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2)],
            additional_targets=t)
    elif name == "extreme":
        return A.Compose([
            A.HorizontalFlip(p=0.5), A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.5), A.GaussNoise(std_range=(0.01, 0.10), p=0.3),
            A.Affine(translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)}, scale=(0.85, 1.15),
                     rotate=(-8, 8), shear=(-5, 5), p=0.5),
            A.GridDistortion(num_steps=6, distort_limit=0.2, p=0.25),
            A.OpticalDistortion(distort_limit=0.1, p=0.15),
            A.Perspective(scale=(0.05, 0.1), p=0.2)], additional_targets=t)

# ─── DATASET ───────────────────────────────────────────────────────────────────
class SegmentationDataset4Class(Dataset):
    def __init__(self, records, image_dir, input_size=(512, 512), transform=None, apply_clahe=False):
        self.records, self.image_dir, self.H, self.W = records, Path(image_dir), input_size[0], input_size[1]
        self.transform, self.apply_clahe = transform, apply_clahe

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(str(self.image_dir / rec["filename"]))
        if img is None: raise FileNotFoundError(rec["filename"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            r = self.transform(image=img, mask=mask); img, mask = r["image"], r["mask"]

        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_norm = (img.astype(np.float32) / 255.0 - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        return img_tensor, torch.from_numpy(mask).long()

# ─── LOSS ──────────────────────────────────────────────────────────────────────
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.dice_weight = dice_weight; self.smooth = 1e-6

    def forward(self, pred, target):
        ce = self.ce(pred, target)
        dice = 0.0
        for c in range(1, NUM_CLASSES):
            p = F.softmax(pred, dim=1)[:, c].reshape(-1)
            t = (target == c).float().reshape(-1)
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice += 1.0 - (2.0*inter + self.smooth)/(union + self.smooth)
        return (1.0 - self.dice_weight)*ce + self.dice_weight*(dice / (NUM_CLASSES - 1))

# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_dice(pred, target, nc):
    pm = F.softmax(pred, dim=1).argmax(dim=1)
    scores = []
    for c in range(nc):
        pcm = (pm == c).cpu().float()
        tcm = (target == c).cpu().float()
        inter = (pcm * tcm).sum().item()
        union = pcm.sum().item() + tcm.sum().item()
        scores.append((2.0*inter + 1e-6) / (union + 1e-6))
    return float(np.mean(scores))

def compute_iou(pred, target, nc):
    pm = F.softmax(pred, dim=1).argmax(dim=1)
    scores = []
    for c in range(nc):
        pcm = (pm == c).cpu().float()
        tcm = (target == c).cpu().float()
        inter = (pcm * tcm).sum().item()
        union = pcm.sum().item() + tcm.sum().item() - inter
        scores.append(inter / (union + 1e-6))
    return float(np.mean(scores))

# ─── MODEL ─────────────────────────────────────────────────────────────────────
def build_model(arch, enc):
    base = (smp.DeepLabV3Plus(encoder_name=enc, encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES, activation=None)
            if arch == "DeepLabV3Plus" else
            smp.Unet(encoder_name=enc, encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES, activation=None,
                     attention_mode={"attention_mode": "scse"} if arch == "AttentionUnet" else {}))
    if GPU_COUNT > 1:
        log.info("→ DataParallel across %d GPUs", GPU_COUNT)
        base = nn.DataParallel(base)
    return base.to(DEVICE)

def get_train_pid():
    """Dynamically find the training script PID."""
    try:
        result = subprocess.run(["pgrep", "-f", "auto_research_iter1.py"],
                                capture_output=True, text=True)
        pids = [p for p in result.stdout.strip().split("\n") if p and p != str(os.getpid())]
        return pids[0] if pids else None
    except:
        return None

# ─── TRAINING ────────────────────────────────────────────────────────────────
def train(arch, enc, lr, wd, bs_per_gpu, aug_name, use_clahe, dice_w, epochs, train_r, val_r):
    model      = build_model(arch, enc)
    total_bs   = bs_per_gpu * max(1, GPU_COUNT)
    log.info("  Batch: %d × %d = %d total | GPUs: %d", bs_per_gpu, max(1, GPU_COUNT), total_bs, GPU_COUNT)

    train_loader = DataLoader(SegmentationDataset4Class(train_r, RAW_DIR, (512,512), build_aug(aug_name), use_clahe),
                              batch_size=total_bs, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(SegmentationDataset4Class(val_r,   RAW_DIR, (512,512), None, use_clahe),
                              batch_size=total_bs, shuffle=False, num_workers=0)
    crit = CrossEntropyDiceLoss(dice_w)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_dice, best_iou, best_state, no_improve = 0.0, 0.0, {}, 0
    for ep in range(1, epochs+1):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(imgs), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        dice_list, iou_list = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                p = model(imgs)
                dice_list.append(compute_dice(p, masks, NUM_CLASSES))
                iou_list.append(compute_iou(p, masks, NUM_CLASSES))

        dice = float(np.mean(dice_list)); iou = float(np.mean(iou_list))
        log.info("  Ep %d/%d | dice=%.4f | iou=%.4f", ep, epochs, dice, iou)
        if dice > best_dice:
            best_dice = dice; best_iou = iou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 4: log.info("  → Early stop ep %d", ep); break

    model.load_state_dict(best_state); model.to(DEVICE)
    return model, best_dice, best_iou

# ─── GRID ─────────────────────────────────────────────────────────────────────
ARCHITECTURES = [
    ("Unet",            "resnet34"),
    ("Unet",            "efficientnet-b4"),
    ("DeepLabV3Plus",   "resnet34"),
    ("AttentionUnet",   "resnet34"),
    ("Linknet",         "resnet34"),
]

def build_grid():
    # Override: if a fast_grid.json exists, use it as the sole grid (pruned search)
    override_file = DATA_DIR / "processed" / "fast_grid.json"
    if override_file.exists():
        combos_raw = json.loads(override_file.read_text())
        combos = []
        for c in combos_raw:
            arch = c["arch_name"]
            enc  = c.get("encoder_name",
                         ("resnet34" if arch != "Unet" else "efficientnet-b4"))
            lr   = c["lr"]; wd = c["weight_decay"]
            bs   = c.get("batch_size", 4)
            aug  = c["aug_name"]; clahe = c.get("use_clahe", True)
            dw   = c.get("dice_w", 0.5)
            combos.append((arch, enc, lr, wd, bs, aug, clahe, dw))
        log.info("  Loaded %d combos from fast_grid.json", len(combos))
        return combos
    combos = []
    for arch, enc in ARCHITECTURES:
        for lr in [1e-4, 3e-4, 1e-3]:
            for wd in [1e-4, 5e-4, 1e-3]:
                for bs in [4, 8]:
                    for aug in ["light", "medium", "heavy", "extreme"]:
                        for clahe in [True, False]:
                            for dw in [0.3, 0.5, 0.7]:
                                combos.append((arch, enc, lr, wd, bs, aug, clahe, dw))
    return combos

# ─── GIT HELPERS ─────────────────────────────────────────────────────────────
WORK_BRANCH = "optimize"

def git_commit_best(model_name, dice):
    branch = f"experiment/{model_name}_gpu{GPU_COUNT}_dice{int(dice*1000):03d}"
    try:
        subprocess.run(["git", "stash"], capture_output=True)
        subprocess.run(["git", "checkout", "-b", branch], check=True, capture_output=True)
        subprocess.run(["git", "add", "-f", "scripts/auto_research_iter1.py", "config.yaml", "product.md"], check=True)
        subprocess.run(["git", "commit", "-m", f"feat(ai): [Auto-Research] {model_name} dice={dice:.4f}"], check=True, capture_output=True)
        subprocess.run(["git", "checkout", WORK_BRANCH], check=True, capture_output=True)
        result = subprocess.run(["git", "stash", "list"], text=True, capture_output=True)
        if result.stdout.strip(): subprocess.run(["git", "stash", "pop"], capture_output=True)
    except Exception as e:
        log.error("Git error: %s", e)

# ─── SPLIT ────────────────────────────────────────────────────────────────────
def patient_split(records, val_pct=0.2, seed=42):
    random.seed(seed)
    pids = list({r["patient_id"] for r in records})
    random.shuffle(pids)
    n_val = max(1, int(len(pids) * val_pct))
    val_pids = set(pids[:n_val])
    return [r for r in records if r["patient_id"] not in val_pids], [r for r in records if r["patient_id"] in val_pids]

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-per-run", type=int, default=12)
    ap.add_argument("--max-train-images", type=int, default=None)
    ap.add_argument("--max-experiments", type=int, default=0)  # 0 = unlimited
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log.info("=" * 65)
    log.info("  INFINITE AUTO-RESEARCH — 4-Class + Multi-GPU DataParallel")
    log.info("  GPUs: %d | Batch scaled × GPU_COUNT", GPU_COUNT)
    log.info("=" * 65)

    records = json.loads(SEG_JSON.read_text())
    if args.max_train_images: records = records[:args.max_train_images]
    train_r, val_r = patient_split(records, 0.2, args.seed)
    log.info("Train: %d | Val: %d | Classes: %d", len(train_r), len(val_r), NUM_CLASSES)

    grid = build_grid()
    random.seed(args.seed); random.shuffle(grid)
    log.info("Grid: %d combos | epochs: %d", len(grid), args.epochs_per_run)

    best_ever, exp_idx = 0.8401, 0  # 0.8401 is previous 3-class baseline
    exp_count = 0

    try:
        while True:
            arch, enc, lr, wd, bs, aug, clahe, dw = next(iter(grid))
            if args.max_experiments > 0 and exp_count >= args.max_experiments:
                log.info(f"Max experiments ({args.max_experiments}) reached. Exiting.")
                break
            exp_count += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"exp{exp_idx:04d}_{arch}_{enc}_gpu{GPU_COUNT}_clahe{int(clahe)}_{ts}"

            log.info("")
            log.info("═══════════════════════════════════════════════════════")
            log.info("EXPERIMENT #%d: %s(%s) lr=%.0e wd=%.0e bs=%d aug=%s clahe=%s dice_w=%.1f",
                     exp_idx, arch, enc, lr, wd, bs, aug, clahe, dw)
            log.info("═══════════════════════════════════════════════════════")

            model, dice, iou = train(arch, enc, lr, wd, bs, aug, clahe, dw,
                                   args.epochs_per_run, train_r, val_r)

            ckpt_dir = MODEL_OUT / run_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            cfg = {"experiment_index": exp_idx, "run_id": run_id, "arch_name": arch,
                   "encoder_name": enc, "lr": lr, "weight_decay": wd, "batch_size": bs,
                   "aug_name": aug, "use_clahe": clahe, "loss_dice_weight": dw,
                   "val_dice": dice, "val_iou": iou, "num_classes": NUM_CLASSES,
                   "classes": ["Background"] + POLYGON_CLASSES,
                   "elapsed_sec": 0, "timestamp": datetime.now().isoformat()}
            (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))

            log.info("  Result: Dice=%.4f | IoU=%.4f", dice, iou)
            if dice > best_ever:
                log.info("  ★★★ NEW BEST: %.4f (was %.4f)", dice, best_ever)
                git_commit_best(f"{arch}_{enc}_gpu{GPU_COUNT}", dice)
                best_ever = dice

            exp_idx += 1
            try: next(iter(grid))
            except StopIteration:
                random.shuffle(grid); log.info("Grid reshuffled")

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()