#!/usr/bin/env python3
"""Test if temperature=10.0 gives varied non-center predictions vs temperature=0.1."""
import json, sys, torch, numpy as np, pandas as pd, glob
sys.path.insert(0, '/home/iddi/ceph-v2')
from src.phase2.dataset import CephalometricDataset, get_kfold_splits
from src.phase2.model import HeatmapHead, build_hrnet
from src.phase2.augmentation import build_val_transform
from src.phase2.heatmap import SoftArgmax2D
import yaml

with open('/home/iddi/ceph-v2/config.yaml') as f:
    cfg = yaml.safe_load(f)

input_size = tuple(cfg['model']['input_size'])
heatmap_size = tuple(cfg['model']['heatmap_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DIR = '/home/iddi/ceph-v2/data/raw/images'

with open('/home/iddi/ceph-v2/data/processed/landmarks_clean.json') as f:
    records = [r for r in json.load(f)['images'] if r.get('has_landmarks')]
splits = get_kfold_splits(records, n_folds=5)
_, val_ids = splits[0]

val_records = {r['image_id']: r for r in records}
val_ds = CephalometricDataset(
    [val_records[i] for i in val_ids],
    image_dir=IMAGE_DIR,
    input_size=input_size,
    transform=build_val_transform(),
)

backbone = build_hrnet(num_keypoints=10, pretrained=True)
model = torch.nn.Sequential(backbone, HeatmapHead(in_channels=2048, num_keypoints=10)).to(device)
model.eval()

models = sorted(glob.glob('/home/iddi/ceph-v2/outputs/fold_1_best*.pth'))
if models:
    print(f"Loading trained model: {models[-1]}")
    state = torch.load(models[-1], map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
else:
    print("No trained model — using random init")

calib_df = pd.read_csv('/home/iddi/ceph-v2/data/processed/calibration.csv')
calib_lookup = dict(zip(calib_df['image_id'], calib_df['mm_per_pixel']))

print(f"\n{'='*60}")
print(f"Comparing soft-argmax at temp=0.1 vs temp=10.0")
print(f"{'='*60}\n")

from torch.utils.data import DataLoader
loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

# Create two SoftArgmax extractors
soft_low = SoftArgmax2D(temperature=0.1)
soft_high = SoftArgmax2D(temperature=10.0)

with torch.no_grad():
    for batch_idx, (imgs, keypoints_gt, valid_mask, metas) in enumerate(loader):
        if batch_idx >= 2:
            break

        imgs = imgs.to(device)
        pred = model(imgs)

        if pred.shape[-2:] != (heatmap_size[0], heatmap_size[1]):
            pred = torch.nn.functional.interpolate(pred, size=heatmap_size, mode="bilinear", align_corners=False)

        # Decode with both temperatures directly
        coords_low, _ = soft_low(pred.cpu(), input_size)
        coords_high, _ = soft_high(pred.cpu(), input_size)

        for b in range(imgs.shape[0]):
            image_id = metas['image_id'][b]
            mm_per_px = calib_lookup.get(image_id, 0.0984)
            valid = valid_mask[b].numpy()
            gt = keypoints_gt[b].numpy()[valid]

            cl = coords_low[b].numpy()[valid]
            ch = coords_high[b].numpy()[valid]

            all_same_low = all(np.allclose(cl[i], cl[0], atol=0.1) for i in range(len(cl)))
            all_same_high = all(np.allclose(ch[i], ch[0], atol=0.1) for i in range(len(ch)))

            dist_from_center = np.sqrt((cl[:, 0] - 256)**2 + (cl[:, 1] - 256)**2).mean()
            dist_high = np.sqrt((ch[:, 0] - 256)**2 + (ch[:, 1] - 256)**2).mean()

            mre_low = np.sqrt(((cl - gt)**2).sum(axis=1)).mean() * mm_per_px
            mre_high = np.sqrt(((ch - gt)**2).sum(axis=1)).mean() * mm_per_px

            print(f"{image_id} (mm/px={mm_per_px:.4f}):")
            print(f"  temp=0.1: {[f'({c[0]:.1f},{c[1]:.1f})' for c in cl]}")
            print(f"            all_same={all_same_low}, dist_from_center={dist_from_center:.1f}px, MRE={mre_low:.2f}mm")
            print(f"  temp=10:  {[f'({c[0]:.1f},{c[1]:.1f})' for c in ch]}")
            print(f"            all_same={all_same_high}, dist_from_center={dist_high:.1f}px, MRE={mre_high:.2f}mm")
            print()

print("Done.")