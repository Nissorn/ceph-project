import json, sys, torch, numpy as np
sys.path.insert(0, '/home/iddi/ceph-v2')
from src.phase2.dataset import CephalometricDataset, get_kfold_splits
from src.phase2.model import HeatmapHead, build_hrnet
from src.phase2.augmentation import build_val_transform
from src.phase2.heatmap import SoftArgmax2D
import yaml, pandas as pd

with open('/home/iddi/ceph-v2/config.yaml') as f:
    cfg = yaml.safe_load(f)
input_size = tuple(cfg['model']['input_size'])
heatmap_size = tuple(cfg['model']['heatmap_size'])
device = torch.device('cuda')
IMAGE_DIR = '/home/iddi/ceph-v2/data/raw/images'

with open('/home/iddi/ceph-v2/data/processed/landmarks_clean.json') as f:
    records = [r for r in json.load(f)['images'] if r.get('has_landmarks')]
splits = get_kfold_splits(records, n_folds=5)
_, val_ids = splits[0]
val_records = {r['image_id']: r for r in records}
val_ds = CephalometricDataset([val_records[i] for i in val_ids], image_dir=IMAGE_DIR,
    input_size=input_size, transform=build_val_transform())

backbone = build_hrnet(num_keypoints=10, pretrained=False)
model = torch.nn.Sequential(backbone, HeatmapHead(in_channels=2048, num_keypoints=10)).to(device)
model.eval()

# Load a trained model
import glob
models = sorted(glob.glob('/home/iddi/ceph-v2/outputs/fold_1_best*.pth'))
if models:
    print(f'Loading {models[-1]}')
    state = torch.load(models[-1], map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)

# Take one image
from torch.utils.data import DataLoader
loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
imgs, _, _, metas = next(iter(loader))
imgs = imgs.to(device)

with torch.no_grad():
    logits = model(imgs)  # [1, 10, 256, 256]
    sig = torch.sigmoid(logits)
    print(f'Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}')
    print(f'Sigmoid: min={sig.min():.4f}, max={sig.max():.4f}, mean={sig.mean():.4f}')
    print()
    # Stats per keypoint
    for k in range(10):
        kp_sig = sig[0, k]
        print(f'  KP{k}: min={kp_sig.min():.4f}, max={kp_sig.max():.4f}, '
              f'std={kp_sig.std():.4f}, mean={kp_sig.mean():.4f}, '
              f'range={kp_sig.max()-kp_sig.min():.4f}')
        # How many pixels > mean + 0.01?
        threshold = kp_sig.mean() + 0.01
        above = (kp_sig > threshold).sum().item()
        print(f'         pixels above mean+0.01: {above}/{kp_sig.numel()} = {above/kp_sig.numel()*100:.1f}%')

    # Now decode
    soft10 = SoftArgmax2D(temperature=10.0)
    coords, _ = soft10(logits.cpu(), input_size)
    print(f'\nSoft-argmax coords (temp=10): {coords[0]}')
    
    # What about with high temperature?
    for t in [0.1, 1.0, 5.0, 10.0, 100.0]:
        soft = SoftArgmax2D(temperature=t)
        c, _ = soft(logits.cpu(), input_size)
        print(f'temp={t}: coords = {c[0, 0].tolist()}')
