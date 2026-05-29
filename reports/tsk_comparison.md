# Phase 2D Results — TSK-01 to TSK-04 Comparison

*Generated: 2026-05-29*

## Quick Summary

| Task | Owner | Status | Dice (Seg) | MRE (Landmark) | SDR@2mm | Notes |
|------|-------|--------|------------|----------------|---------|-------|
| TSK-01 | trainer | ✅ done | — | **0.495 mm** | **98.15%** | Sliding Window (Pipeline B) |
| TSK-02 | data-engineer | ✅ done | — | — | — | 240 boundary crops generated |
| TSK-03 | trainer | ✅ done | 0.1431 (loss) | — | — | Stage 2 refiner, 4.38M params |
| TSK-04 | trainer | ✅ done | **0.8827** | — | — | Tversky+BoundaryDice fine-tune |


---
## TSK-01 — Sliding Window Inference (Pipeline B)

**Implementation:** `backend/app/services/analysis_service.py`
- Zero retraining — uses existing 512px DeepLabV3+ model
- 512×512 patches, stride=256 (50% overlap), Gaussian-weighted averaging
- Sigma = patch_size/4 = 128 → weight falls to ~0.01 at patch edges
- Activates for images >512px in any axis

**Landmark Detection Quality (5-Fold LOPO-CV):**
- **MRE: 0.4947 ± 0.7197 mm** ← below 1mm
- **SDR@2mm: 98.15%**
- **SDR@2.5mm: 98.26%**
- **SDR@3mm: 99.02%**
- **SDR@4mm: 99.57%**

**Per-landmark MRE:**
| Landmark | MRE (mm) |
|----------|----------|
| Upper_tip | 0.515 |
| Upper_apex | 0.472 |
| Labial_midroot | 0.316 |
| Labial_crest | 0.311 |
| Palatal_midroot | 0.320 |
| Palatal_crest | 0.360 |
| ANS | 0.599 |
| PNS | 0.629 |
| LB | 0.662 |
| PB | 0.763 |

**Per-fold MRE:** Fold1=0.416, Fold2=0.504, Fold3=0.564, Fold4=0.553, Fold5=0.441

**Comparison — before adaptive sigma:**
- Old (no sigma): MRE=9.426 mm, SDR@2mm=0%
- New (adaptive sigma): MRE=0.495 mm, SDR@2mm=98.15%
- **Improvement: 19× reduction in MRE**


---
## TSK-02 — Generate Refiner Crops (Pipeline A)

**Implementation:** Boundary crop extraction from 512px baseline predictions
- 240 boundary crops extracted (80 per class: Upper_incisor, Labial_bone, Palatal_bone)
- Crop size: 384×128px (aligned to bone boundary orientation)
- Used as training data for TSK-03 Stage 2 refiner
- Patient-level split maintained (no leakage)


---
## TSK-03 — Train Stage 2 Lightweight Refiner

**Architecture:** DeepLabV3Plus encoder (frozen) + MobileNetV2 decoder
- Total params: 4.38M (lightweight — designed to run fast)
- Input: 256×256 (resized boundary crops)
- Loss: 0.6×Dice + 0.4×Focal
- Training: 189 train / 51 val crops, patient-level split

**Result:** best val_loss=0.1431 at epoch 40/60 (early stopping)

**Note:** This refiner was trained but has not yet been integrated into the main inference pipeline (TSK-01 already uses sliding window with the 512px baseline). The refiner would provide additional boundary refinement if deployed.


---
## TSK-04 — Tversky + BoundaryDice Fine-Tuning

**Architecture:** DeepLabV3Plus + resnet34 (fine-tuned from 512px baseline)
- Loss: 0.6×Tversky(α=0.7,β=0.3) + 0.4×BoundaryDice
- Training: 291 train / 71 val records (patient-level split)
- Epochs: 50 (early stopping, patience=10)
- Model size: 22.5M params

**Result:**
- **Segmentation Dice: 0.8827** ← NEW PROJECT BEST
- Previous champion (baseline): 0.8588
- **Improvement: +0.0238 (+2.8%)**

**Output:** `models/tversky_deepLabV3plus_resnet34_20250529_20260529_094221/best_model.pt`

**Note:** This is a segmentation model (Dice), not directly comparable to the landmark MRE from TSK-01. To compare:
- TSK-04 improved segmentation quality (boundary detection)
- TSK-01 improved landmark localization accuracy (MRE in mm)
- These address different aspects of the pipeline


---
## Cross-Task Comparison

**Key insight:** TSK-01 and TSK-04 address different failure modes:

| Metric | TSK-01 (Sliding Window) | TSK-04 (Tversky Fine-tune) |
|--------|------------------------|---------------------------|
| Task | Landmark localization | Segmentation boundary |
| Measure | MRE in mm | Dice coefficient |
| Result | **0.495 mm MRE** | **0.8827 Dice** |
| Improvement | 19× over old baseline | +2.8% over baseline |
| Stage | Inference (zero retrain) | Training |

**Recommended next step:** Fine-tune TSK-04 model WITH sliding window inference.
The TSK-04 champion (Dice=0.8827) has not yet been tested with the sliding window
pipeline — combining the best segmentation model with the best inference strategy
should yield further improvements.
