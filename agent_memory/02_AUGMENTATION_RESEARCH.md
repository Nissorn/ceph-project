# 02_AUGMENTATION_RESEARCH.md — Data Augmentation Strategy

**Research Completed:** 2026-05-05  
**Status:** Ready for implementation  
**Context:** 104 initial images → 300+ incoming; preventing overfitting is CRITICAL

---

## Problem Statement

Paper findings (arxiv:2505.06055):
- **HRNet-W32 without augmentation on 104 images:** 63.8% SDR@2.5mm (severe overfitting)
- **With heavy augmentation:** 75–80% SDR@2.5mm
- **With augmentation + diffusion synthesis:** 88–91% SDR@2.5mm (state-of-art)

**Key insight:** Augmentation effort matters MORE than model switching for low-data regimes

---

## Safe Augmentation Ranges for Lateral Cephalograms

| Technique | Safe Range | Rationale | Implementation |
|-----------|-----------|-----------|-----------------|
| **Rotation** | ±10–15° | Beyond breaks anatomical head positioning | `A.Rotate(limit=15, ...)` |
| **Brightness/Contrast** | ±20–40% | X-rays have specific grayscale; preserve landmarks | `A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.4)` |
| **Elastic Deformation** | α: 30–100, σ: 3–5 | Simulates natural imaging variation | `A.ElasticTransform(alpha=(30,80), sigma=4)` |
| **Shift (Crop/Pan)** | ±5–10% dims | Keep landmarks visible in frame | `A.Shift(limit=0.08)` |
| **Gaussian Noise** | Std dev 0.01–0.05 | Simulate X-ray sensor noise | `A.GaussNoise(var_limit=(10,40))` |
| **Grid Distortion** | num_steps: 5–7, distort: 0.1–0.3 | Controlled spatial warping | `A.GridDistortion(num_steps=5, distort_limit=0.2)` |
| **Perspective** | scale: 0.05–0.1 | Subtle 3D camera angle effect | `A.Perspective(scale=(0.05,0.1))` |

**NEVER USE:** Horizontal flip (breaks lateral anatomy specificity)

---

## ✅ Recommended Implementation: Albumentations Pipeline

### File to Create: `src/phase2/augmentation.py`

```python
import albumentations as A
import cv2
from typing import Tuple

def create_cephalogram_augmentation(p_enable: float = 0.8) -> A.Compose:
    """
    Heavy augmentation for cephalometric X-rays.
    
    Input: 104 images
    Output: ~500+ variants per epoch via augmentation
    Expected SDR@2.5mm: 75–80% (vs. 63.8% baseline without augmentation)
    
    Args:
        p_enable: Probability to apply each augmentation (0.8 = 80%)
    
    Returns:
        Albumentations Compose pipeline with keypoint support
    """
    return A.Compose([
        # Core transforms
        A.Rotate(
            limit=15,  # ±15 degrees
            border_mode=cv2.BORDER_REFLECT,
            p=p_enable
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # ±20%
            contrast_limit=0.4,    # ±40%
            p=p_enable
        ),
        A.ElasticTransform(
            alpha=(30, 80),  # Deformation intensity
            sigma=4,         # Smoothness
            p=0.5
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,  # ±20% distortion
            p=0.3
        ),
        A.GaussNoise(
            var_limit=(10, 40),  # X-ray sensor noise
            p=0.3
        ),
        A.Shift(
            limit=0.08,  # ±8% shift
            p=0.5
        ),
        A.Perspective(
            scale=(0.05, 0.1),  # Subtle 3D effect
            p=0.2
        ),
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False  # Keep all landmarks even if off-frame
    ))


def create_light_augmentation() -> A.Compose:
    """
    Light augmentation for validation/test (minimal changes).
    Use this for validation set to avoid artificially inflating metrics.
    """
    return A.Compose([
        A.NoOp(),  # Placeholder; add minimal transforms if needed
    ], keypoint_params=A.KeypointParams(
        format='xy',
        remove_invisible=False
    ))
```

### Integration with DataLoader

```python
# In src/phase2/dataset.py or wherever you load training data

from augmentation import create_cephalogram_augmentation

class CephalogramDataset(torch.utils.data.Dataset):
    def __init__(self, images, landmarks, augment=True):
        self.images = images
        self.landmarks = landmarks
        self.augment = augment
        
        if augment:
            self.transform = create_cephalogram_augmentation()
        else:
            self.transform = None
    
    def __getitem__(self, idx):
        image = self.images[idx]  # numpy array, shape (H, W) or (H, W, 1)
        landmarks = self.landmarks[idx]  # list of (x, y) tuples, 10 keypoints
        
        if self.transform:
            augmented = self.transform(
                image=image,
                keypoints=landmarks
            )
            image = augmented['image']
            landmarks = augmented['keypoints']
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        landmarks = torch.tensor(landmarks).float()
        
        return image, landmarks
```

### Usage in Training Loop

```python
# Training setup
train_dataset = CephalogramDataset(
    images=train_images,
    landmarks=train_landmarks,
    augment=True  # Enable heavy augmentation
)
val_dataset = CephalogramDataset(
    images=val_images,
    landmarks=val_landmarks,
    augment=False  # NO augmentation for validation
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0  # IMPORTANT: MPS restriction on Mac M4
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    num_workers=0
)

# Training loop
model = HRNet(num_keypoints=10)  # Your model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()  # Or your landmark loss

for epoch in range(100):
    model.train()
    for images, landmarks in train_loader:
        # Augmentation already applied in DataLoader
        pred_landmarks = model(images)
        loss = criterion(pred_landmarks, landmarks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation (no augmentation)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, landmarks in val_loader:
            pred_landmarks = model(images)
            loss = criterion(pred_landmarks, landmarks)
            val_loss += loss.item()
    print(f"Epoch {epoch} | Val Loss: {val_loss/len(val_loader):.4f}")
```

---

## 📊 Expected Performance

| Configuration | SDR@2mm | SDR@2.5mm | MRE (mm) | Notes |
|---------------|---------|-----------|----------|-------|
| Baseline (no augment) | 58% | 63.8% | 2.8 | Severe overfitting |
| **Heavy Augmentation (Albumentations)** | **68%** | **75–80%** | **2.2** | ✅ RECOMMENDED (2–3 days effort) |
| + Diffusion Synthesis (300 synthetic) | 82% | 88% | 1.4 | Optional Phase 2 (7–10 days) |

---

## 🤖 Phase 2: Diffusion-Based Synthetic Generation (Optional)

### When to activate:
- After traditional augmentation plateaus
- When 20+ annotated images available
- Before 300+ real images arrive

### DDPM Training Code (Hugging Face Diffusers)

```python
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Initialize U-Net for cephalograms (grayscale 256px images)
unet = UNet2DModel(
    sample_size=256,
    in_channels=1,    # Grayscale X-ray
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 1024),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
    attention_head_dim=8,
)

scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
device = torch.device("mps")  # Apple Silicon

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    unet.train()
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        
        # Sample random timesteps
        t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],)).to(device)
        
        # Add noise to images
        noise = torch.randn_like(images)
        noisy_images = scheduler.add_noise(images, noise, t)
        
        # Predict noise
        noise_pred = unet(noisy_images, t).sample
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f}")

# Save trained model
torch.save(unet.state_dict(), "ceph_ddpm_unet.pth")

# Generation: Create 300 synthetic cephalograms
pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
pipeline = pipeline.to(device)

num_synthetic = 300
for i in range(num_synthetic):
    with torch.no_grad():
        synthetic_image = pipeline(
            num_inference_steps=1000,  # Full denoising schedule
            output_type="numpy"
        ).images[0]
    
    # Save synthetic image
    import numpy as np
    from PIL import Image
    synthetic_uint8 = (synthetic_image * 255).astype(np.uint8)
    Image.fromarray(synthetic_uint8).save(f"synthetic_ceph_{i:04d}.png")
    
    if (i + 1) % 50 == 0:
        print(f"Generated {i+1}/{num_synthetic} synthetic images")
```

### Synthesis Strategy:
- Generate **2–5 synthetic images per real image**
- For 104 real images: **200–400 synthetic cephalograms**
- Mix in training set: 104 real + 300 synthetic = 404 total
- Expected accuracy: **82–88% SDR@2.5mm**
- Training effort: 7–10 days + GPU cost (~$20–50)

---

## 3-Phase Roadmap

| Phase | Timeline | Action | Expected Accuracy | Cost | Effort |
|-------|----------|--------|-------------------|------|--------|
| **1: Traditional Augmentation** | Weeks 1–2 | Implement Albumentations | 75–80% SDR@2.5mm | Free | 2–3 days |
| **2: DDPM Synthesis** | Weeks 3–4 | Train diffusion model, generate 300 synthetic | 82–88% SDR@2.5mm | $20–50 GPU | 7–10 days |
| **3: Scale to 300+ Real** | Weeks 5+ | Integrate Dr.'s 300+ images | 90%+ SDR@2.5mm | Minimal | 3–5 days |

---

## Installation & Dependencies

```bash
# Install Albumentations
pip install albumentations

# (Optional) For Phase 2 diffusion synthesis
pip install diffusers transformers

# Verify installation
python -c "import albumentations; print(f'Albumentations {albumentations.__version__}')"
python -c "from diffusers import DDPMPipeline; print('Diffusers ready')"
```

---

## Implementation Checklist

- [ ] Create `src/phase2/augmentation.py` with above code
- [ ] Update `src/phase2/dataset.py` to integrate augmentation
- [ ] Run baseline test: train without augmentation, record SDR metrics
- [ ] Run with augmentation, compare metrics (expect ≥75% SDR@2.5mm)
- [ ] Document results in `notebooks/` exploratory notebook
- [ ] (Optional) Prepare DDPM training code for Phase 2
- [ ] Validate landmarks still visible after extreme augmentations
- [ ] Confirm no augmentation on validation/test sets

---

## Validation Checklist

Before deploying augmentation to full training:

1. **Visual inspection:** Apply augmentation to 5–10 images, verify landmarks still visible
2. **Landmark consistency:** Check that augmented landmark positions make anatomical sense
3. **Loss convergence:** Verify training loss decreases smoothly (not erratic)
4. **No overfitting to augmentations:** Val loss shouldn't increase despite train loss decreasing
5. **Baseline comparison:** Measure exact improvement with/without augmentation on LOPO-CV

---

## Key References

1. **arxiv:2505.06055** — Guo et al., "Towards Better Cephalometric Landmark Detection with Limited Annotated Data"
   - Proves HRNet-W32 overfitting on <300 images without augmentation
   - Recommends heavy augmentation as primary mitigation

2. **arxiv:2407.18125** — Di Via et al., "Denoising Diffusion Probabilistic Models for X-ray Synthesis"
   - Outperformed ImageNet pre-training on X-ray landmark tasks
   - Optimal synthetic:real ratio = 2:1 to 5:1

3. **Albumentations Library** — https://github.com/albumentations-team/albumentations
   - CPU-fast, keypoint-aware, medical imaging support

4. **Hugging Face Diffusers** — https://github.com/huggingface/diffusers
   - Easy DDPM implementation for medical image generation

---

**Status:** Ready for implementation. Recommend starting with Albumentations pipeline (Action 1, immediate priority).
