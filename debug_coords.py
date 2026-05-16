#!/usr/bin/env python3
"""Debug script: inspect raw predicted coordinates from validation set."""
import json, sys, torch, numpy as np, pandas as pd, glob, os
sys.path.insert(0, '/home/iddi/ceph-v2')
from src.phase2.dataset import CephalometricDataset, get_kfold_splits
from src.phase2.model import HeatmapHead, build_hrnet
from src.phase2.heatmap import decode_heatmaps
from src.phase2.augmentation import build_val_transform

import yaml
with open('/home/iddi/ceph-v2/config.yaml') as f:
    cfg = yaml.safe_load(f)

input_size = tuple(cfg['model']['input_size'])
heatmap_size = tuple(cfg['model']['heatmap_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DIR = '/home/iddi/ceph-v2/data/raw/images'

# Load data
with open('/home/iddi/ceph-v2/data/processed/landmarks_clean.json') as f:
    data = json.load(f)['images']
records = [r for r in data if r.get('has_landmarks')]
splits = get_kfold_splits(records, n_folds=5)
train_ids, val_ids = splits[0]

val_records = {r['image_id']: r for r in records}
val_ds = CephalometricDataset(
    [val_records[i] for i in val_ids],
    image_dir=IMAGE_DIR,
    input_size=input_size,
    transform=build_val_transform(),
)

# Build model
backbone = build_hrnet(num_keypoints=10, pretrained=True)
model = torch.nn.Sequential(backbone, HeatmapHead(in_channels=2048, num_keypoints=10))
model = model.to(device)
model.eval()

# Load best model from Fold 1 if it exists
models = sorted(glob.glob('/home/iddi/ceph-v2/outputs/fold_1_best*.pth'))
if models:
    print(f"Loading: {models[-1]}")
    state = torch.load(models[-1], map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
else:
    print("No saved model — using random init (to see what center predicts)")

# Calibration
calib_df = pd.read_csv('/home/iddi/ceph-v2/data/processed/calibration.csv')
calibration_lookup = dict(zip(calib_df['image_id'], calib_df['mm_per_pixel']))

print(f"\n{'='*70}")
print(f"DEBUG: Raw coordinate inspection — Fold 1 val ({len(val_ds)} images)")
print(f"{'='*70}")
print(f"Heatmap: {heatmap_size} | Input: {input_size} | Device: {device}")
print()

from torch.utils.data import DataLoader
loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

center_x = input_size[1] / 2  # 256.0
center_y = input_size[0] / 2  # 256.0

with torch.no_grad():
    for batch_idx, (imgs, keypoints_gt, valid_mask, metas) in enumerate(loader):
        if batch_idx >= 4:  # 4 batches × 4 = 16 samples
            break

        imgs = imgs.to(device)
        pred_heatmaps = model(imgs)

        if pred_heatmaps.shape[-2:] != (heatmap_size[0], heatmap_size[1]):
            pred_heatmaps = torch.nn.functional.interpolate(
                pred_heatmaps, size=heatmap_size, mode="bilinear", align_corners=False
            )

        # --- SOFT-ARGMAX COORDS ---
        coords_soft, confidence = decode_heatmaps(pred_heatmaps.cpu(), input_size)

        # --- HARD ARGMAX COORDS ---
        B, N, H, W = pred_heatmaps.shape
        conf_sigmoid = torch.sigmoid(pred_heatmaps)
        flat_conf = conf_sigmoid.view(B * N, H * W)
        max_vals, flat_idx = flat_conf.max(dim=-1)
        argmax_x = (flat_idx % W).float()
        argmax_y = (flat_idx // W).float()
        # scale to input_size coords
        argmax_x = argmax_x / (W - 1) * input_size[1]
        argmax_y = argmax_y / (H - 1) * input_size[0]
        coords_argmax = torch.stack([argmax_x, argmax_y], dim=-1).view(B, N, 2)

        for b in range(imgs.shape[0]):
            image_id = metas['image_id'][b]
            mm_per_px = calibration_lookup.get(image_id, 0.0984)
            valid = valid_mask[b].numpy()
            gt = keypoints_gt[b].numpy()[valid]
            coord_soft = coords_soft[b].numpy()[valid]
            coord_argmax = coords_argmax[b].cpu().numpy()[valid]

            dist_from_center = np.sqrt(
                (coord_soft[:, 0] - center_x)**2 + (coord_soft[:, 1] - center_y)**2
            ).mean()

            mre_soft = np.sqrt(((coord_soft - gt)**2).sum(axis=1)).mean() * mm_per_px
            mre_argmax = np.sqrt(((coord_argmax - gt)**2).sum(axis=1)).mean() * mm_per_px
            mre_center = np.sqrt(
                ((gt[:, 0] - center_x)**2 + (gt[:, 1] - center_y)**2)
            ).mean() * mm_per_px

            all_same = all(np.allclose(coord_soft[i], coord_soft[0]) for i in range(len(coord_soft)))

            print(f"Image: {image_id}  mm/px={mm_per_px:.4f}")
            print(f"  Soft coords (x,y):    {[f'{c[0]:6.1f},{c[1]:6.1f}' for c in coord_soft]}")
            print(f"  Hard coords (x,y):    {[f'{c[0]:6.1f},{c[1]:6.1f}' for c in coord_argmax]}")
            print(f"  GT    coords (x,y):    {[f'{c[0]:6.1f},{c[1]:6.1f}' for c in gt]}")
            print(f"  All soft identical? {all_same}")
            print(f"  Dist from center: {dist_from_center:.1f}px")
            print(f"  MRE soft={mre_soft:.2f}mm  argmax={mre_argmax:.2f}mm  center={mre_center:.2f}mm")
            print()

print("Done.")