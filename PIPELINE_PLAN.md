# Ceph-AI Pipeline Plan
_Created: 2026-05-06 | Status: PLANNING — No implementation started_

---

## Current Reality Check

| Component | File | Status |
|-----------|------|--------|
| Calibration parser | `src/data/cvat_parser.py` | ✅ Done (10 keypoints) |
| Calibration output | `data/processed/calibration.csv` | ✅ 104/104 pass QC |
| Augmentation | `src/phase2/augmentation.py` | ⚠️ Exists but underpowered (±5° only) |
| Dataset loader | `src/phase2/dataset.py` | ❌ BUG: only 8 keypoints, missing LB+PB |
| HRNet model | `src/phase2/model.py` | ⚠️ Default=8, comment says 8 (works because train.py passes cfg=10) |
| Heatmap encode/decode | `src/phase2/heatmap.py` | ✅ Done |
| Metrics (MRE/SDR) | `src/phase2/metrics.py` | ✅ Done |
| Training loop (LOPO) | `src/phase2/train.py` | ✅ Done |
| Superimposition | `src/phase3/superimposition.py` | ✅ Done |
| Treatment heuristics | `src/phase3/heuristics.py` | ✅ Done (thresholds pending Dr.) |
| Segmentation (U-Net) | ❌ Does not exist yet | — |
| Classification (CNN) | ❌ Does not exist yet | — |
| Split strategy | ❌ Does not exist yet | — |
| landmarks_clean.json | ❌ Not generated yet | — |

**Blocker:** Dr. skeleton annotations (~2/104 done). Training needs 20+ annotated images minimum.

---

## Research Findings (NotebookLM Verified)

### Loss Function
- **MSE (L2)** — standard, but bad for heatmaps: 99% of pixels are background zeros, model learns to ignore the keypoint region
- **Adaptive Wing Loss** — designed specifically for heatmap regression; penalizes errors on foreground pixels (actual keypoints) much more than background; used in CL-Detection 2023 top teams
- **Recommendation: Replace MSE with Adaptive Wing Loss in `train.py`**

### Split Strategy
- Stratified K-fold cross-validation (k=5) is the recommended approach for small, imbalanced medical datasets
- Must isolate a holdout test set FIRST (stratified, ~15%), then do 5-fold CV on the rest
- For the 104 images: ~15 images holdout, 89 images for CV folds (~18/fold)
- Patient-awareness must be maintained: T1+T2 of same patient always in same fold
- LOPO (Leave-One-Patient-Out) is the extreme version but wastes too much training data at this scale

### Augmentation
- Research (arxiv:2505.06055) proves ±5° rotation is too conservative — use ±10–15°
- Without augmentation on 104 images: SDR@2.5mm = 63.8% (severe overfit)
- With heavy augmentation: 75–80% SDR@2.5mm
- Safe additions: ElasticTransform, GaussNoise, GridDistortion, ShiftScaleRotate

---

## Full Pipeline: Input → Output Per Phase

```
CVAT XML
   │
   ▼
[Phase 0: Parse & Validate]
   Input:  data/annotations.xml
   Output: data/processed/landmarks_clean.json
           (all 104 images, calibration, skeleton where available, polygons, tags)
   │
   ▼
[Phase 0b: Split Strategy]
   Input:  landmarks_clean.json + patient_ids + treatment tags
   Output: data/processed/splits.json
           (5-fold train/val + held-out test, patient-aware + stratified)
   │
   ▼
[Phase 1: Data Augmentation]  ← applies PER BATCH during training
   Input:  Raw image (H×W, grayscale or 3ch) + 10 keypoints [(x,y)] + polygons
   Output: Augmented image (512×512) + transformed keypoints + transformed masks
   Transforms: Rotate(±10°), BrightnessContrast, ElasticTransform, GaussNoise,
               GridDistortion, ShiftScaleRotate(±8%), Perspective(0.05–0.1),
               CLAHE — NO horizontal flip
   │
   ▼
[Phase 2a: Landmark Detection — HRNet-W32]
   Input:  Image tensor [B, 3, 512, 512] normalized to [0,1]
   Output: Heatmaps [B, 10, 128, 128] → decoded (x,y) per landmark + confidence [B, 10]
   Loss:   Adaptive Wing Loss (replaces MSE)
   Metric: MRE (mm), SDR@2mm / 2.5mm / 3mm / 4mm — per landmark + overall
   Note:   mm_per_pixel from calibration.csv (per-image, never global)
   │
   ▼
[Phase 2b: Segmentation — U-Net]
   Input:  Image tensor [B, 3, 512, 512]
   Output: Binary masks [B, 3, 512, 512]
           Channel 0 = Upper_incisor, 1 = Labial_bone, 2 = Palatal_bone
   Loss:   Dice Loss + BCE (combined)
   Metric: IoU per polygon class, mean Dice score
   Note:   Polygons → masks via cv2.fillPoly at load time
   │
   ▼
[Phase 2c: Treatment Classification — EfficientNet-B3]
   Input:  Image tensor [B, 3, 512, 512]
   Output: Multi-label logits [B, 6] → sigmoid → probabilities
           Classes: Uncontrolled_tipping, Controlled_tipping, Translation,
                    Root_torque, Extrusion, Intrusion
   Loss:   Binary Cross-Entropy with class weights (handle imbalance)
   Metric: Per-class AUC, macro-F1, confusion matrix
   Note:   T1-only images; reject "Quality_Reject" and "Low_Visibility" from training
   │
   ▼
[Phase 3: Superimposition Engine — Algorithmic]
   Input:  T1 keypoints [10, 2] + T2 keypoints [10, 2] + mm_per_pixel (T1 image)
           Only uses: ANS (idx 6), PNS (idx 7), Upper_tip (idx 0), Upper_apex (idx 1)
   Output: {
             angle_change_deg: float,
             delta_tip_mm: [dx, dy],
             delta_apex_mm: [dx, dy],
             treatment_class: str  (pending threshold until Dr. confirms params)
           }
   Method: ANS-PNS rigid registration → Δtip/Δapex vectors × mm_per_pixel
   Note:   Only valid on 52 paired T1+T2 images — NEVER train on this
   │
   ▼
[Phase 4: Clinical Output]
   Input:  All above outputs per image
   Output: JSON report per patient:
           {
             patient_id, image_t1, image_t2,
             landmarks: { name: {x,y,confidence} },
             segmentation: { polygon: iou_score },
             superimposition: { delta_tip_mm, delta_apex_mm, treatment_class },
             quality: { reject, low_visibility }
           }
```

---

## What To Do While Waiting for Dr. Annotations

### Priority 1 — Fix Bugs (do this first, ~1 hour)
**Target files:** `src/phase2/dataset.py`, `src/phase2/model.py`

**Bug 1 — dataset.py KEYPOINT_NAMES is wrong (8 → 10):**
```python
# Current (WRONG):
KEYPOINT_NAMES = ["Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
                  "Palatal_midroot", "Palatal_crest", "ANS", "PNS"]
NUM_KEYPOINTS = 8

# Fix to:
KEYPOINT_NAMES = ["Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
                  "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB"]
NUM_KEYPOINTS = 10
```

**Bug 2 — model.py docstring and default says 8:**
```python
# Fix: Update comment from "8-keypoint" → "10-keypoint"
# Fix: Change NUM_KEYPOINTS = 8 → NUM_KEYPOINTS = 10
```

---

### Priority 2 — Generate landmarks_clean.json (~30 min)
**New script:** `scripts/parse_annotations.py`

Run `cvat_parser.parse_cvat_xml()` on `data/annotations.xml` and save result to `data/processed/landmarks_clean.json`. This is what `train.py` already expects to load. Currently missing — without it, training cannot start even with annotated data.

Input: `data/annotations.xml`  
Output: `data/processed/landmarks_clean.json`

---

### Priority 3 — Implement Proper Split Strategy (~3 hours)
**New file:** `src/data/splits.py`

```
Function: build_splits(records, n_folds=5, holdout_ratio=0.15, random_state=42)

Steps:
1. Group records by patient_id (so T1+T2 always stay together)
2. Extract multi-label treatment vector per patient (for stratification)
3. StratifiedShuffleSplit → isolate ~15% holdout patients
4. StratifiedKFold(n_splits=5) → 5-fold CV on remaining 85% patients
5. Save to data/processed/splits.json:
   {
     "holdout": [patient_ids],
     "folds": [
       {"train": [patient_ids], "val": [patient_ids]},
       ...  × 5
     ]
   }

Libraries: sklearn.model_selection.StratifiedShuffleSplit + StratifiedKFold
Constraint: Multi-label stratification is tricky → use iterative-stratification
            (pip install iterative-stratification) or stratify on most imbalanced class
```

---

### Priority 4 — Upgrade Augmentation Pipeline (~2 hours)
**File:** `src/phase2/augmentation.py` (update existing)

```
Current: Rotate(±5°), RandomScale, RandomBrightnessContrast, CLAHE
Target:  Rotate(±10°), ShiftScaleRotate(shift=0.08), BrightnessContrast,
         ElasticTransform(alpha=30–80, sigma=4), GaussNoise(var=10–40),
         GridDistortion(distort=0.2), Perspective(0.05–0.1), CLAHE
         
Update config.yaml: rotation_limit: 5 → 10 (not 15, be conservative first)
```

---

### Priority 5 — Add Adaptive Wing Loss (~2 hours)
**New file:** `src/phase2/loss.py`

```python
class AdaptiveWingLoss(nn.Module):
    """
    AdaptiveWingLoss for heatmap regression.
    Source: Wang et al., "Adaptive Wing Loss for Robust Face Alignment 
            via Heatmap Regression" (ICCV 2019)
    
    Key idea: background pixels (GT near 0) → small penalty
              foreground pixels (GT near 1) → large penalty
    
    Parameters:
        omega=14, theta=0.5, epsilon=1, alpha=2.1
    """

Update train.py: replace nn.MSELoss() with AdaptiveWingLoss()
```

---

### Priority 6 — Build Augmentation Preview Notebook (~1 hour)
**File:** `notebooks/04_augmentation_preview.ipynb`

```
Purpose: Visually verify augmentation doesn't break landmark anatomical positions
Steps:
1. Load 1 annotated image (one of the 2 that exists)
2. Apply augmentation 12 times
3. Plot 3×4 grid with landmarks overlaid as colored dots
4. Eyeball: Are ANS, PNS, Upper_tip still in correct relative positions?
5. Verify no landmark goes off-frame (remove_invisible=False must hold)
```

---

### Priority 7 — Scaffold U-Net Segmentation (~4 hours)
**New file:** `src/phase2b/segmentation.py`
**New file:** `src/phase2b/segmentation_dataset.py`

```
Model: segmentation_models_pytorch (smp) — UNet with ResNet-34 encoder
       Pretrained on ImageNet → much better than training from scratch
       pip install segmentation-models-pytorch

SegmentationDataset.__getitem__():
    Input:  Image path + polygons from record["polygons"]
    Step 1: Load image (512×512)
    Step 2: For each polygon (Upper_incisor, Labial_bone, Palatal_bone):
                cv2.fillPoly() → binary mask (512×512)
    Step 3: Stack → masks [3, 512, 512]
    Output: (image_tensor [3,512,512], masks [3,512,512])

Loss: DiceLoss + BCEWithLogitsLoss (weighted 0.5 each)
Metric: IoU per class, mean Dice
Note: Only train on images that have polygon annotations
      Currently 0 images have polygons — scaffold now, train when Dr. finishes
```

---

### Priority 8 — Scaffold Classification (~2 hours)
**New file:** `src/phase2c/classifier.py`
**New file:** `src/phase2c/classifier_dataset.py`

```
Model: torchvision EfficientNet-B3 (pretrained ImageNet)
       Replace final FC layer: fc → Linear(num_features, 6)
       
ClassificationDataset.__getitem__():
    Input:  Image path + record["treatment"] list
    Output: (image_tensor [3,512,512], label_vector [6] float)
    label_vector[i] = 1.0 if class_i in record["treatment"] else 0.0
    Filter: Skip records where "Quality_Reject" in quality_flags
            Skip T2 images (no treatment tag on T2)

Loss: BCEWithLogitsLoss with pos_weight for class imbalance
      pos_weight[i] = (n_negative[i] / n_positive[i]) per class

Metric: Per-class AUC, macro-F1
Note: EfficientNet-B3 better than ResNet for small datasets
      Lighter params, ImageNet pretrain transfers well to X-rays
```

---

### Priority 9 — Dry-Run Training Test (~1 hour)
Once Priorities 1–3 are done, run a full dry-run with 2 annotated images:
```bash
python scripts/run_phase2_train.py --debug --max-images 2
```
Goal: Verify the full data → dataloader → model → loss → metrics pipeline works end-to-end without crashing. Not for real results — just pipeline validation.

---

## Phase Dependency Graph

```
[NOW — No annotations needed]
Priority 1: Fix bugs in dataset.py + model.py
Priority 2: Generate landmarks_clean.json
Priority 3: Implement split strategy
Priority 4: Upgrade augmentation
Priority 5: Add Adaptive Wing Loss
Priority 6: Augmentation preview notebook
Priority 7: U-Net scaffold
Priority 8: EfficientNet classifier scaffold
Priority 9: Dry-run training test

[WAITING FOR 20+ ANNOTATIONS]
Start 5-fold CV training on HRNet-W32
Evaluate MRE/SDR on val set
Compare MSE vs Adaptive Wing Loss empirically

[WAITING FOR FULL ANNOTATIONS (all 104)]
Full segmentation training (U-Net)
Full classification training (EfficientNet)
LOPO cross-validation (legacy, use 5-fold as primary)

[WAITING FOR 52 PAIRED PATIENT ANNOTATIONS]
Superimposition validation
Treatment classification from geometry (Phase 3)

[WAITING FOR DR. THRESHOLD CONFIRMATION]
Enable heuristic treatment_class output (currently returns "pending_threshold")
```

---

## Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Loss function | Adaptive Wing Loss | Proven better than MSE for heatmap regression in medical tasks |
| Split strategy | 5-fold Stratified CV + holdout | NotebookLM verified; LOPO wastes too much data at 104 images |
| Augmentation strength | ±10° rotation (not ±15° yet) | Start conservative, tune up if model still overfit |
| Segmentation library | segmentation_models_pytorch | Fast, well-tested, pretrained encoders |
| Classification backbone | EfficientNet-B3 | Better than ResNet for small datasets, lighter |
| Segmentation loss | Dice + BCE combined | Dice handles class imbalance in masks; BCE stabilizes training |

---

## Files To Create / Modify Summary

| Action | File | Priority |
|--------|------|----------|
| FIX | `src/phase2/dataset.py` — KEYPOINT_NAMES 8→10 | P1 |
| FIX | `src/phase2/model.py` — default NUM_KEYPOINTS 8→10 | P1 |
| CREATE | `scripts/parse_annotations.py` | P2 |
| CREATE | `src/data/splits.py` | P3 |
| UPDATE | `src/phase2/augmentation.py` — add 4 new transforms | P4 |
| UPDATE | `config.yaml` — rotation_limit 5→10 | P4 |
| CREATE | `src/phase2/loss.py` — Adaptive Wing Loss | P5 |
| UPDATE | `src/phase2/train.py` — use AdaptiveWingLoss | P5 |
| CREATE | `notebooks/04_augmentation_preview.ipynb` | P6 |
| CREATE | `src/phase2b/segmentation.py` | P7 |
| CREATE | `src/phase2b/segmentation_dataset.py` | P7 |
| CREATE | `src/phase2c/classifier.py` | P8 |
| CREATE | `src/phase2c/classifier_dataset.py` | P8 |

---

_Say "start implementation" to begin. Start with Priority 1 (bugs) → Priority 2 (parse) → Priority 3 (splits)._
