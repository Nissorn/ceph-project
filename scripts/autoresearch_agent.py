#!/usr/bin/env python3
"""
Karpathy-style AutoResearch Agent — Phase 2B Alveolar Bone Segmentation.

Unlike the grid-search script, this daemon queries an LLM after every experiment
to decide the next architecture and hyperparameters. The LLM receives the full
experiment history and suggests the next config in JSON format.

Architecture: LLM-driven closed loop
  Experiment → Metrics → LLM API Call → Suggested Config → Train again

Exit: Manual Ctrl-C / SIGINT only.

Usage:
    python scripts/autoresearch_agent.py                           # defaults
    python scripts/autoresearch_agent.py --api-key sk-...         # with key
    python scripts/autoresearch_agent.py --base-url https://...    # custom endpoint
    python scripts/autoresearch_agent.py --epochs-per-run 30       # longer runs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.phase2b.segmentation import POLYGON_CLASSES, NUM_SEG_CLASSES

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = PROJECT_ROOT / "autoresearch_agent.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("autoresearch_agent")

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
SEG_JSON       = PROJECT_ROOT / "data" / "processed" / "segmentation_train.json"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
GIT_WORK_BRANCH = "optimize"

# ── LLM API Configuration ────────────────────────────────────────────────────
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL    = "gpt-4o-mini"
DEFAULT_PROVIDER  = "openai"   # or "anthropic" — selects SDK / call style


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATASET & TRAINING INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════

class SegmentationDatasetAuto(torch.utils.data.Dataset):
    """Reads segmentation_train.json and rasterises CVAT polygons into binary masks."""

    def __init__(self, records: list[dict], image_dir: Path,
                 input_size: tuple[int, int] = (512, 512), transform=None):
        self.records  = records
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
            result    = self.transform(image=img_resized, masks=aug_masks)
            img_resized = result["image"]
            aug_masks   = result["masks"]
            masks = np.stack(aug_masks, axis=0)

        img_tensor   = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        masks_tensor = torch.from_numpy(masks)
        meta = {"image_id": rec["image_id"], "patient_id": rec["patient_id"]}
        return img_tensor, masks_tensor, meta


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: AUGMENTATION PRESETS
# ═══════════════════════════════════════════════════════════════════════════

AUG_PRESETS: dict[str, A.Compose] = {}

def _build_aug_presets():
    """Build augmentation presets once NUM_SEG_CLASSES is known."""
    targets = {f"mask{i}": "mask" for i in range(NUM_SEG_CLASSES - 1)}
    AUG_PRESETS["none"] = A.Compose([
        A.Resize(512, 512),
        ToTensorV2(),
    ], additional_targets=targets)

    AUG_PRESETS["light"] = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Resize(512, 512),
        ToTensorV2(),
    ], additional_targets=targets)

    AUG_PRESETS["medium"] = A.Compose([
        A.HorizontalFlip(p=0.4),
        A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.25),
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 scale=(0.9, 1.1), p=0.3),
        A.Resize(512, 512),
        ToTensorV2(),
    ], additional_targets=targets)

    AUG_PRESETS["heavy"] = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.08), p=0.3),
        A.Affine(translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                 scale=(0.88, 1.12), rotate=(-5, 5), p=0.4),
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        A.Resize(512, 512),
        ToTensorV2(),
    ], additional_targets=targets)

_build_aug_presets()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: LOSS & METRICS
# ═══════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """BCE + Dice loss with configurable alpha/beta weighting."""
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss  = self.bce(pred, target)
        pred_sig  = torch.nn.functional.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union        = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = (1.0 - (2 * intersection + 1e-6) / (union + 1e-6)).mean()
        return self.alpha * bce_loss + self.beta * dice_loss


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Mean Dice score across classes. pred = logits, target = binary masks."""
    pred_sig = torch.sigmoid(pred)
    dice_per_class = []
    for c in range(pred.shape[1]):
        p = pred_sig[:, c].reshape(-1).cpu().numpy()
        t = target[:, c].reshape(-1).cpu().numpy()
        intersection = (p * t).sum()
        union        = p.sum() + t.sum()
        dice_per_class.append((2.0 * intersection + smooth) / (union + smooth))
    return float(np.mean(dice_per_class))


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Mean IoU score across classes."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    iou_per_class = []
    for c in range(pred.shape[1]):
        p = pred_bin[:, c].reshape(-1).cpu().numpy()
        t = target[:, c].reshape(-1).cpu().numpy()
        intersection = np.sum(p * t)
        union        = np.sum(p) + np.sum(t) - intersection
        iou_per_class.append(intersection / (union + 1e-6))
    return float(np.mean(iou_per_class))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_one_config(
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
    patience: int = 3,
    experiment_index: int = 0,
) -> dict:
    """
    Train a single configuration. Returns result dict with metrics and history.
    """
    # ── Build model ───────────────────────────────────────────────────────────
    if arch_name == "AttentionUnet":
        model = smp.Unet(
            encoder_name=encoder_name, encoder_weights="imagenet",
            in_channels=3, classes=NUM_SEG_CLASSES, activation=None,
        )
    elif arch_name == "UnetPlusPlus" or arch_name == "MAnet":
        model = smp.MAnet(
            encoder_name=encoder_name, encoder_weights="imagenet",
            in_channels=3, classes=NUM_SEG_CLASSES, activation=None,
        )
    elif arch_name == "FPN":
        model = smp.FPN(
            encoder_name=encoder_name, encoder_weights="imagenet",
            in_channels=3, classes=NUM_SEG_CLASSES, activation=None,
        )
    elif arch_name == "PSPNet":
        model = smp.PSPNet(
            encoder_name=encoder_name, encoder_weights="imagenet",
            in_channels=3, classes=NUM_SEG_CLASSES, activation=None,
        )
    else:
        # Try as direct SMP factory
        arch_map = {
            "Unet": smp.Unet, "DeepLabV3Plus": smp.DeepLabV3Plus,
            "Linknet": smp.Linknet, "PAN": smp.PAN,
        }
        factory = arch_map.get(arch_name, smp.Unet)
        model = factory(
            encoder_name=encoder_name, encoder_weights="imagenet",
            in_channels=3, classes=NUM_SEG_CLASSES, activation=None,
        )
    model = model.to(DEVICE)

    # ── Datasets & Loaders ─────────────────────────────────────────────────────
    aug = AUG_PRESETS.get(aug_name, AUG_PRESETS["none"])
    train_ds = SegmentationDatasetAuto(train_records, image_dir, (512, 512), transform=aug)
    val_ds   = SegmentationDatasetAuto(val_records,   image_dir, (512, 512), transform=AUG_PRESETS["none"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Optimiser & Scheduler ───────────────────────────────────────────────────
    criterion   = CombinedLoss(alpha=loss_alpha, beta=loss_beta)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training ───────────────────────────────────────────────────────────────
    best_dice = 0.0
    best_iou  = 0.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    epoch_history = []
    start_time = time.time()

    config_str = f"{arch_name}({encoder_name}) lr={lr} wd={weight_decay} bs={batch_size} aug={aug_name}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"auto{experiment_index:04d}_{arch_name}_{encoder_name}_{ts}"

    log.info("")
    log.info("═══════════════════════════════════════════════════════")
    log.info("EXPERIMENT #%d: %s", experiment_index, config_str)
    log.info("═══════════════════════════════════════════════════════")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, masks, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)
            optimizer.zero_grad()
            pred   = model(images)
            loss   = criterion(pred, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_dice_list, val_iou_list = [], []
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(DEVICE)
                masks  = masks.to(DEVICE)
                pred   = model(images)
                val_dice_list.append(compute_dice(pred, masks))
                val_iou_list.append(compute_iou(pred, masks))

        val_dice = float(np.mean(val_dice_list))
        val_iou  = float(np.mean(val_iou_list))
        lr_now   = optimizer.param_groups[0]["lr"]

        log.info(
            "  Epoch %d/%d | train_loss=%.4f | val_dice=%.4f | val_iou=%.4f | lr=%.6f",
            epoch, epochs, avg_loss, val_dice, val_iou, lr_now,
        )

        # Track per-epoch history
        epoch_history.append({
            "epoch": epoch, "train_loss": round(avg_loss, 4),
            "val_dice": round(val_dice, 4), "val_iou": round(val_iou, 4),
            "lr": round(lr_now, 6),
        })

        if val_dice > best_dice:
            best_dice  = val_dice
            best_iou   = val_iou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            log.info("  ★ New best dice: %.4f", best_dice)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  → Early stopping at epoch %d", epoch)
                break

    elapsed = time.time() - start_time
    model.load_state_dict(best_state)
    model.to(DEVICE)

    log.info("  Result: Dice=%.4f | IoU=%.4f | Time=%.0fs", best_dice, best_iou, elapsed)

    # ── Save checkpoint ────────────────────────────────────────────────────────
    ckpt_dir = MODEL_OUTPUT_DIR / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    config_json = {
        "run_id": run_id, "experiment_index": experiment_index,
        "arch_name": arch_name, "encoder_name": encoder_name,
        "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        "aug_name": aug_name, "loss_alpha": loss_alpha, "loss_beta": loss_beta,
        "epochs_trained": epochs, "val_dice": round(best_dice, 4),
        "val_iou": round(best_iou, 4), "elapsed_sec": round(elapsed, 1),
        "epoch_history": epoch_history,
        "timestamp": datetime.now().isoformat(),
    }
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    return {
        "run_id": run_id, "experiment_index": experiment_index,
        "config": config_json, "dice": best_dice, "iou": best_iou,
        "elapsed": elapsed, "epoch_history": epoch_history,
        "ckpt_dir": ckpt_dir,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: LLM ADVISOR — Karpathy Autoresearch Core
# ═══════════════════════════════════════════════════════════════════════════

class LLMAdvisor:
    """
    Queries an LLM after each experiment to suggest the next configuration.

    Prompt engineering:
    - Feeds full experiment history (arch, LR, WD, BS, aug, loss weights → Dice, IoU)
    - Asks the LLM to analyse overfitting, undertrained models, and suggest next config
    - Parses JSON response for the next hyperparameters

    Supports OpenAI (chat.completions) and Anthropic (messages) APIs.
    """

    SYSTEM_PROMPT = """You are an autonomous ML researcher specialising in medical image segmentation.
You are advising a training loop that runs segmentation experiments on cephalometric X-ray images.
You receive the history of past experiments with their hyperparameters and validation Dice scores.
Your task is to analyse the trends and suggest the next experiment configuration.

Key domain knowledge:
- Segmentation targets alveolar bone boundary polygons in dental X-rays
- Background class dominates (>95% of pixels), making Dice harder to optimise than IoU
- Small training set (~274 images), so augmentation and regularisation are critical
- ImageNet-pretrained encoders: resnet34, efficientnet-b4, se_resnet50, resnet50
- SCSE attention (in AttentionUnet) consistently outperforms vanilla U-Net
- BCE weight alpha=0.7 helps when the model is underfit early in training

Respond ONLY with valid JSON in this exact format, no markdown, no explanation:
{
  "reasoning": "2-3 sentences explaining your choice based on the history",
  "arch_name": "Unet|DeepLabV3Plus|AttentionUnet|UnetPlusPlus|MAnet|FPN|PSPNet|PAN",
  "encoder_name": "resnet34|efficientnet-b4|se_resnet50|resnet50",
  "lr": 0.0001 or 0.0003 or 0.001,
  "weight_decay": 0.0001 or 0.0005 or 0.001,
  "batch_size": 4 or 8,
  "aug_name": "none|light|medium|heavy",
  "loss_alpha": 0.5 or 0.7 or 1.0,
  "loss_beta": 0.0 or 0.3 or 0.5
}"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        max_history: int = 20,
    ):
        self.api_key   = api_key
        self.base_url  = base_url.rstrip("/")
        self.model     = model
        self.provider  = provider
        self.max_history = max_history
        self._client    = None

    def _build_client(self):
        """Lazy-import and configure the HTTP client."""
        if self.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_user_prompt(self, history: list[dict]) -> str:
        """Build the user prompt with experiment history."""
        if not history:
            return (
                "No experiments run yet. We have a small dataset (~274 images) of "
                "cephalometric X-rays for alveolar bone segmentation. "
                "Start with a strong baseline: AttentionUnet with resnet34 encoder, "
                "lr=0.001, weight_decay=0.001, batch_size=4, aug=light, "
                "loss_alpha=0.5, loss_beta=0.5.\n"
                f"Suggest the first experiment configuration in JSON format."
            )

        # Format recent history
        lines = ["Experiment history (most recent last):", ""]
        for h in history[-self.max_history:]:
            cfg = h["config"]
            lines.append(
                f"  #{h['experiment_index']}: {cfg['arch_name']}({cfg['encoder_name']}) "
                f"lr={cfg['lr']} wd={cfg['weight_decay']} bs={cfg['batch_size']} "
                f"aug={cfg['aug_name']} loss=α{cfg['loss_alpha']}_β{cfg['loss_beta']} "
                f"→ Dice={h['dice']:.4f} IoU={h['iou']:.4f}"
            )

        lines.extend([
            "",
            "Analyse the above history and suggest the next configuration.",
            "Prioritise: (1) beating best Dice, (2) exploring underrepresented configs,",
            "(3) avoiding repeating failed combos.",
        ])
        return "\n".join(lines)

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response, stripping any markdown/code fences."""
        text = text.strip()
        # Remove ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        text = text.strip("`").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("JSON parse error: %s — raw response: %s", exc, text[:200])
            return None

    def _call_openai(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, messages: list[dict]) -> str:
        # Convert messages format
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        user_content = "\n\n".join(user_msgs)
        response = self._client.messages.create(
            model=self.model,
            system=sys_msg,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=512,
        )
        return response.content[0].text

    def suggest(self, experiment_history: list[dict]) -> Optional[dict]:
        """
        Query the LLM and return the next configuration as a dict.
        Returns None on failure; the caller should use a fallback config.
        """
        if self._client is None:
            self._build_client()

        system_msg = {"role": "system", "content": self.SYSTEM_PROMPT}
        user_prompt = self._build_user_prompt(experiment_history)
        user_msg = {"role": "user", "content": user_prompt}

        try:
            if self.provider == "openai":
                raw = self._call_openai([system_msg, user_msg])
            elif self.provider == "anthropic":
                raw = self._call_anthropic([system_msg, user_msg])
            else:
                return None

            log.info("LLM response received (%d chars)", len(raw))
            result = self._parse_json_response(raw)
            if result is None:
                return None

            # Validate required keys
            required = ["arch_name", "encoder_name", "lr", "weight_decay",
                        "batch_size", "aug_name", "loss_alpha", "loss_beta"]
            for k in required:
                if k not in result:
                    log.warning("LLM response missing key '%s': %s", k, result)
                    return None

            log.info(
                "LLM suggested: %s(%s) lr=%.4f wd=%.4f bs=%d aug=%s loss=α%.1f_β%.1f",
                result["arch_name"], result["encoder_name"],
                result["lr"], result["weight_decay"], result["batch_size"],
                result["aug_name"], result["loss_alpha"], result["loss_beta"],
            )
            return result

        except Exception as exc:
            log.error("LLM API call failed: %s", exc)
            return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: FALLBACK CANDIDATE POOL
# ═══════════════════════════════════════════════════════════════════════════

FALLBACK_GRID: list[dict] = [
    # Arch, encoder, lr, wd, bs, aug, alpha, beta
    ("AttentionUnet", "resnet34",       0.001, 0.001,  4, "light",  0.5, 0.5),
    ("AttentionUnet", "resnet34",       0.001, 0.0005, 4, "heavy",  0.5, 0.5),
    ("AttentionUnet", "se_resnet50",    0.001, 0.001,  4, "light",  0.7, 0.3),
    ("DeepLabV3Plus", "resnet34",       0.0003,0.001,  4, "medium", 0.5, 0.5),
    ("Unet",          "efficientnet-b4",0.0003,0.001,  4, "light",  0.7, 0.3),
    ("Linknet",       "resnet34",       0.001, 0.0005, 4, "light",  0.5, 0.5),
    ("UnetPlusPlus",  "resnet34",       0.0003,0.001,  4, "medium", 0.5, 0.5),
    ("MAnet",         "resnet34",       0.001, 0.0005, 4, "heavy",  0.7, 0.3),
    ("FPN",           "resnet34",       0.0003,0.001,  4, "light",  0.5, 0.5),
    ("PSPNet",        "resnet34",       0.0003,0.001,  8, "medium", 0.5, 0.5),
]

_fallback_iter = None

def get_fallback_config() -> dict:
    """Return next fallback config from the candidate pool (round-robin)."""
    global _fallback_iter
    if _fallback_iter is None:
        _fallback_iter = iter(FALLBACK_GRID)
    try:
        cfg = next(_fallback_iter)
    except StopIteration:
        _fallback_iter = iter(FALLBACK_GRID)
        cfg = next(_fallback_iter)
    return dict(
        arch_name=cfg[0], encoder_name=cfg[1], lr=cfg[2],
        weight_decay=cfg[3], batch_size=cfg[4], aug_name=cfg[5],
        loss_alpha=cfg[6], loss_beta=cfg[7],
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: GIT AUTO-COMMIT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def git_current_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()


def git_checkout(branch: str) -> None:
    subprocess.run(["git", "checkout", branch], check=True, capture_output=True)


def git_create_branch(branch: str) -> None:
    subprocess.run(["git", "checkout", "-b", branch], check=True, capture_output=True)


def git_add_commit(message: str, files: list[str]) -> None:
    for f in files:
        subprocess.run(["git", "add", str(f)], check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)


def stash_and_checkout(target: str) -> bool:
    """Stash dirty state, checkout target branch. Returns True if clean."""
    result = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True)
    if result.stdout.strip():
        subprocess.run(["git", "stash"], capture_output=True)
    try:
        git_checkout(target)
        return True
    except subprocess.CalledProcessError:
        return False


def auto_commit_best(result: dict, best_dice: float, current_best: float) -> float:
    """
    If result['dice'] > current_best: create experiment branch, commit config, return new best.
    """
    dice = result["dice"]
    if dice <= current_best:
        return current_best

    cfg  = result["config"]
    run_id = result["run_id"]
    branch = f"llm/{cfg['arch_name']}-{cfg['encoder_name']}-dice{int(dice*1000):04d}"
    log.info("★★★ NEW BEST: %.4f (was %.4f) — branching to %s ★★★", dice, current_best, branch)

    # Stash any WIP, create branch, commit, return to work branch
    stash_and_checkout(GIT_WORK_BRANCH)
    try:
        git_create_branch(branch)
        git_add_commit(
            f"perf(llm): achieve new best dice {dice:.4f} with {cfg['arch_name']}({cfg['encoder_name']}) "
            f"lr={cfg['lr']} wd={cfg['weight_decay']} bs={cfg['batch_size']} "
            f"aug={cfg['aug_name']} loss=α{cfg['loss_alpha']}_β{cfg['loss_beta']}",
            files=[
                MODEL_OUTPUT_DIR / run_id / "config.json",
                PROJECT_ROOT / "config.yaml",
            ],
        )
        log.info("Branch '%s' committed.", branch)
    except subprocess.CalledProcessError as exc:
        log.warning("Git pipeline failed (non-critical): %s", exc)

    stash_and_checkout(GIT_WORK_BRANCH)
    return dice


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: PATIENT SPLIT HELPER
# ═══════════════════════════════════════════════════════════════════════════

def patient_split(records: list[dict], val_pct: float = 0.2, seed: int = 42):
    """Split by patient_id so T1+T2 of same patient stay in same split."""
    random.seed(seed)
    patient_ids = list({r["patient_id"] for r in records})
    random.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * val_pct))
    val_patients = set(patient_ids[:n_val])
    train = [r for r in records if r["patient_id"] not in val_patients]
    val   = [r for r in records if r["patient_id"] in val_patients]
    return train, val


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: MAIN AUTORESEARCH DAEMON
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLM-Driven Autoresearch Agent for Phase 2B")
    parser.add_argument("--api-key",      default=os.environ.get("OPENAI_API_KEY", ""),
                        help="API key for LLM provider")
    parser.add_argument("--base-url",     default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
                        help="API base URL (OpenAI-compatible)")
    parser.add_argument("--model",        default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
                        help="Model name (e.g. gpt-4o-mini)")
    parser.add_argument("--provider",     default="openai",
                        choices=["openai", "anthropic"],
                        help="API provider")
    parser.add_argument("--epochs-per-run", type=int, default=30,
                        help="Epochs per training run")
    parser.add_argument("--patience",     type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--max-history",  type=int, default=20,
                        help="Max experiment history sent to LLM")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip-llm",    action="store_true",
                        help="Skip LLM calls — use fallback grid only (for debugging)")
    args = parser.parse_args()

    if not args.api_key:
        log.warning("No API key provided. Set OPENAI_API_KEY env var or --api-key. "
                    "Running in FALLBACK mode (no LLM).")
        args.skip_llm = True

    print("=" * 65)
    print("  AUTORESEARCH AGENT — Phase 2B Alveolar Bone Segmentation")
    print("  LLM-driven hyperparameter optimisation")
    print("  Press Ctrl-C to stop at any time")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    if not SEG_JSON.exists():
        log.error("segmentation_train.json not found. Run scripts/merge_cvat_data.py first.")
        sys.exit(1)

    with open(SEG_JSON) as f:
        all_records = json.load(f)
    log.info("Loaded %d segmentation records", len(all_records))

    train_records, val_records = patient_split(all_records, val_pct=0.2, seed=args.seed)
    log.info("Train: %d | Val: %d (patient-level split, seed=%d)",
             len(train_records), len(val_records), args.seed)

    if not RAW_IMAGE_DIR.exists():
        log.error("Image directory not found: %s", RAW_IMAGE_DIR)
        sys.exit(1)

    # ── Initialise LLM Advisor ────────────────────────────────────────────────
    advisor = None
    if not args.skip_llm:
        advisor = LLMAdvisor(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            provider=args.provider,
            max_history=args.max_history,
        )
        log.info("LLM Advisor configured: %s / %s", args.provider, args.model)
    else:
        log.info("Running in FALLBACK mode (no LLM) — cycling through candidate grid.")

    # ── Training loop ──────────────────────────────────────────────────────────
    experiment_history: list[dict] = []
    experiment_index   = 0
    best_dice_ever     = 0.0
    fallback_fresh     = True  # ensure we get a real config first

    log.info("")
    log.info("Starting LLM-driven autoresearch loop.")
    log.info("Best Dice tracker initialised at 0.0")
    log.info("=" * 65)

    try:
        while True:
            # ── Get next config ───────────────────────────────────────────────────
            if args.skip_llm or advisor is None:
                cfg = get_fallback_config()
                log.info("[FALLBACK] Using candidate config: %s(%s)",
                         cfg["arch_name"], cfg["encoder_name"])
            else:
                suggestion = advisor.suggest(experiment_history)
                if suggestion is None:
                    log.warning("LLM returned no valid config — using fallback.")
                    cfg = get_fallback_config()
                else:
                    cfg = suggestion
                    log.info("[LLM ADVISOR] Suggested: %s(%s) lr=%.4f wd=%.4f bs=%d aug=%s",
                             cfg["arch_name"], cfg["encoder_name"],
                             cfg["lr"], cfg["weight_decay"],
                             cfg["batch_size"], cfg["aug_name"])

            # ── Run experiment ─────────────────────────────────────────────────────
            result = train_one_config(
                arch_name      = cfg["arch_name"],
                encoder_name   = cfg["encoder_name"],
                lr             = cfg["lr"],
                weight_decay   = cfg["weight_decay"],
                batch_size     = cfg["batch_size"],
                aug_name       = cfg["aug_name"],
                loss_alpha     = cfg["loss_alpha"],
                loss_beta      = cfg["loss_beta"],
                epochs         = args.epochs_per_run,
                patience       = args.patience,
                train_records  = train_records,
                val_records    = val_records,
                image_dir      = RAW_IMAGE_DIR,
                experiment_index = experiment_index,
            )

            dice = result["dice"]
            iou  = result["iou"]
            log.info(
                "  ★ Experiment #%d complete: Dice=%.4f | IoU=%.4f | Time=%.0fs",
                experiment_index, dice, iou, result["elapsed"],
            )

            # ── Update history ────────────────────────────────────────────────────
            experiment_history.append(result)

            # ── Auto-Git pipeline ─────────────────────────────────────────────────
            best_dice_ever = auto_commit_best(result, best_dice_ever, best_dice_ever)

            # ── Early exit on major milestone ─────────────────────────────────────
            if dice >= 0.90:
                log.info("★★★ MAJOR MILESTONE: Dice >= 0.90 achieved! ★★★")
                # Continue running but log prominently

            experiment_index += 1
            log.info("")
            log.info("Loop continuing... (exp #%d done, best Dice=%.4f)",
                     experiment_index - 1, best_dice_ever)
            log.info("")

    except KeyboardInterrupt:
        log.info("")
        log.info("═══════════════════════════════════════════════════════")
        log.info("  KeyboardInterrupt received.")
        log.info("  Auto-research agent stopped.")
        log.info("  Experiments run: %d | Best Dice ever: %.4f",
                 experiment_index, best_dice_ever)
        log.info("═══════════════════════════════════════════════════════")
        return


if __name__ == "__main__":
    main()