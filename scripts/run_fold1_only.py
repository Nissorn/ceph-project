
import sys, json, yaml, torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(".").resolve()))

from src.phase2.model import CephalometricModel
from src.phase2.dataset import CephalometricDataset
from src.phase2.augmentation import build_train_transform, build_val_transform
from src.phase2.heatmap import encode_heatmaps
from src.phase2.loss import AdaptiveWingLoss
from src.phase2.train import compute_mean_mre

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

with open(cfg["data"]["landmarks_json"]) as f:
    landmarks_data = json.load(f)
records_raw = landmarks_data["images"] if isinstance(landmarks_data, dict) else landmarks_data
records = [r for r in records_raw if r.get("has_landmarks")]

cal_df = pd.read_csv(cfg["data"]["calibration_csv"]).set_index("image_id")
calibration_lookup = cal_df["mm_per_pixel"].to_dict()

device = torch.device(cfg["training"]["device"])
input_size = tuple(cfg["model"]["input_size"])
heatmap_size = tuple(cfg["model"]["heatmap_size"])
sigma = cfg["model"]["sigma"]
epochs = 50
lr = cfg["training"]["lr"]
batch_size = cfg["training"]["batch_size"]

train_transform = build_train_transform(
    rotation_limit=cfg["augmentation"]["rotation_limit"],
    zoom_limit=cfg["augmentation"]["zoom_limit"],
    brightness_limit=cfg["augmentation"]["brightness_limit"],
    contrast_limit=cfg["augmentation"]["contrast_limit"],
    clahe=cfg["augmentation"]["clahe"],
    horizontal_flip=False,
)
val_transform = build_val_transform()

image_ids = [r["image_id"] for r in records]
groups = [r["patient_id"] for r in records]
gkf = GroupKFold(n_splits=5)
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(image_ids, groups=groups)):
    if fold_idx == 0:
        train_ids = [image_ids[i] for i in train_idx]
        val_ids = [image_ids[i] for i in val_idx]
        break

print(f"Fold 1: train={len(train_ids)}, val={len(val_ids)}")

train_ds = CephalometricDataset(records, cfg["data"]["image_dir"], input_size, train_transform, image_ids=train_ids)
val_ds = CephalometricDataset(records, cfg["data"]["image_dir"], input_size, val_transform, image_ids=val_ids)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

model = CephalometricModel(num_keypoints=cfg["keypoints"]["num_keypoints"], pretrained=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.001)
criterion = AdaptiveWingLoss()

best_mre = float("inf")
best_state = None
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for imgs, keypoints, valid_mask, metas in train_loader:
        imgs, keypoints, valid_mask = imgs.to(device), keypoints.to(device), valid_mask.to(device)
        gt_heatmaps = []
        for b in range(len(keypoints)):
            hm = encode_heatmaps(keypoints[b].cpu().numpy(), valid_mask[b].cpu().numpy(), heatmap_size, sigma, input_size)
            gt_heatmaps.append(hm)
        gt_tensor = torch.from_numpy(np.stack(gt_heatmaps)).to(device)
        pred = model(imgs)
        if pred.shape[-2:] != gt_tensor.shape[-2:]:
            pred = torch.nn.functional.interpolate(pred, size=heatmap_size, mode="bilinear", align_corners=False)
        mask = valid_mask.bool().to(device)
        loss = criterion(pred, gt_tensor, mask)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch+1) % 5 == 0 or epoch == epochs-1:
        model.eval()
        soft_errors, argmax_errors = [], []
        with torch.no_grad():
            for imgs, keypoints, valid_mask, metas in val_loader:
                imgs = imgs.to(device)
                pred = model(imgs)
                if pred.shape[-2:] != heatmap_size:
                    pred = torch.nn.functional.interpolate(pred, size=heatmap_size, mode="bilinear", align_corners=False)
                B, N, H, W = pred.shape
                conf = torch.sigmoid(pred.cpu())
                flat = conf.view(B*N, -1)
                _, flat_idx = flat.max(dim=-1)
                x_a = (flat_idx % W).float() / W * input_size[1]
                y_a = (flat_idx // W).float() / H * input_size[0]
                coords_a = torch.stack([x_a, y_a], dim=-1).view(B, N, 2)
                for b in range(B):
                    iid = metas["image_id"][b]
                    mm_per_px = calibration_lookup.get(iid, 1.0)
                    v = valid_mask[b].numpy()
                    ae = np.sqrt(np.sum((coords_a[b].numpy()[v] - keypoints[b].numpy()[v])**2, axis=1)) * mm_per_px
                    argmax_errors.append(ae)
        mre_a = compute_mean_mre(argmax_errors)
        marker = ""
        if mre_a < best_mre:
            best_mre = mre_a
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *BEST*"
        else:
            patience_counter += 5
        print(f"  [Fold 1] Epoch {epoch+1}/{epochs} — MRE_argmax: {mre_a:.3f}mm, best: {best_mre:.3f}mm{marker}")
        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

ckpt_dir = Path(cfg["data"]["calibration_csv"]).parent / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": best_state,
    "fold_mre_argmax": best_mre,
    "fold_idx": 1,
    "calibration_lookup": calibration_lookup,
}, ckpt_dir / "fold1_best.pth")
print(f"Saved fold1_best.pth with MRE={best_mre:.3f}mm")
