#!/usr/bin/env python3
"""
Run script for Phase 2b — Bone Segmentation Pipeline.

Usage:
    python scripts/run_phase2b_segmentation.py                      # live mode
    python scripts/run_phase2b_segmentation.py --mock              # dry-run / mock mode
    python scripts/run_phase2b_segmentation.py --mock --limit 10   # mock with cap

Install dependencies:
    pip install segmentation-models-pytorch albumentations opencv-python

Expected data layout:
    data/processed/
        landmarks_clean.json     # {image_id, patient_id, filename, keypoints, polygons}
        images/                  # raw .jpg / .png scans

The pipeline:
  1. SegmentationDataset  — rasterises CVAT polygons → binary masks via cv2.fillPoly
  2. Albumentations dual-target — rotates/flips/elastic-deforms image + masks together
  3. UNet (ResNet-34 encoder) — pretrained ImageNet backbone
  4. SegmentationLoss     — Dice + BCE combined

NOTE: 0/104 images currently have polygon annotations.
This scaffold is ready to train the moment Dr. exports polygon labels from CVAT.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local imports
from src.phase2b.segmentation import POLYGON_CLASSES, build_segmentation_model, SegmentationLoss
from src.phase2b.segmentation_dataset import SegmentationDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGE_DIR = PROCESSED_DIR / "images"
LANDMARKS_JSON = PROCESSED_DIR / "landmarks_clean.json"


def load_records(path: Path) -> list[dict]:
    """Load the cleaned landmark records (one dict per image)."""
    if not path.exists():
        print(f"[ERROR] landmarks_clean.json not found at {path}")
        print("  → Run scripts/run_phase1_calibration.py first to generate it.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_mock_batch():
    """
    Generate a deterministic mock batch to validate the training loop
    end-to-end without needing real images or polygons.
    """
    B, C, H, W = 2, len(POLYGON_CLASSES), 512, 512
    images = torch.rand(B, 3, H, W)
    masks = torch.zeros((B, C, H, W), dtype=torch.float32)
    # Place a few filled polygons in channel 0 (Upper_incisor) so dice loss is meaningful
    masks[:, 0, 100:200, 100:200] = 1.0
    return images, masks


def run_mock_epoch(model: torch.nn.Module, device: torch.device):
    """
    Validate the forward + backward pass using synthetic data.
    Prints gradient norm as a health check.
    """
    model.train()
    criterion = SegmentationLoss()
    images, masks = build_mock_batch()
    images = images.to(device)
    masks = masks.to(device)

    pred = model(images)          # [B, C, H, W] logits
    loss = criterion(pred, masks)

    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()

    print(f"  ✓ Mock forward+backward OK | loss={loss.item():.4f} | grad_norm={grad_norm:.4f}")
    return loss.item()


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch: int):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, masks, _) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | loss={loss.item():.4f}")

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Phase 2b Segmentation Pipeline")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in dry-run mock mode — uses synthetic data, no real images needed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of records loaded (useful with --mock or for quick sanity check).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs in live mode. Default: 5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size. Default: 4.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Default: 1e-4.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        help="smp encoder name. Default: resnet34 (lightweight, accurate).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2b — Bone Segmentation Pipeline")
    print("=" * 60)

    if args.mock:
        print("[MOCK MODE] — Using synthetic data, no real images required.")
        print("  This validates the training loop end-to-end.")
    else:
        print(f"[LIVE MODE] — Loading from {PROCESSED_DIR}")
        records = load_records(LANDMARKS_JSON)
        if args.limit:
            records = records[: args.limit]
            print(f"  Capped to {args.limit} records.")

        polygon_records = [r for r in records if r.get("polygons")]
        print(f"  Total records : {len(records)}")
        print(f"  With polygons: {len(polygon_records)} ({len(polygon_records)/max(len(records),1)*100:.1f}%)")
        if len(polygon_records) == 0:
            print("\n[WARNING] No records have polygon annotations.")
            print("  Annotate polygons in CVAT and re-run run_phase1_calibration.py to regenerate landmarks_clean.json.")
            print("  Falling back to mock mode for scaffold validation.")
            args.mock = True

    # ── Device ──────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")

    # ── Mock mode: validate model scaffold with synthetic batch ─────────────────
    if args.mock:
        print("\n[Step 1] Building model...")
        model = build_segmentation_model(num_classes=len(POLYGON_CLASSES), encoder_name=args.encoder, pretrained=True)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: UNet ({args.encoder}) | {total_params:,} parameters")

        print("\n[Step 2] Running mock epoch validation...")
        losses = []
        for i in range(1, 4):
            loss = run_mock_epoch(model, device)
            losses.append(loss)
        print(f"\n  Mock validation complete | avg loss={np.mean(losses):.4f}")
        print("\n✓ Phase 2b scaffold is healthy. Ready for real data.")
        return

    # ── Live mode ───────────────────────────────────────────────────────────────
    print("\n[Step 1] Building model...")
    model = build_segmentation_model(num_classes=len(POLYGON_CLASSES), encoder_name=args.encoder, pretrained=True)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: UNet ({args.encoder}) | {total_params:,} parameters")

    print("\n[Step 2] Preparing dataset...")
    # Import albumentations lazily (heavy dependency)
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        print("[ERROR] albumentations not installed.")
        print("  → pip install albumentations")
        sys.exit(1)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        # dual-target: tell albumentations about the 'mask' key so it processes it alongside 'image'
        # additional_targets={'mask': 'mask'} is already set in segmentation_dataset.py transform wiring
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    dataset = SegmentationDataset(
        records=polygon_records,
        image_dir=str(IMAGE_DIR),
        input_size=(512, 512),
        transform=train_transform,
        require_polygons=True,
    )
    print(f"  Dataset: {len(dataset)} images with polygon annotations")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,   # MPS restriction on Mac
        pin_memory=False,
    )

    criterion = SegmentationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n[Step 3] Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device, epoch)
        print(f"  ✓ Epoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f}")

    checkpoint_path = PROJECT_ROOT / "models" / "phase2b_unet.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✓ Training complete. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()