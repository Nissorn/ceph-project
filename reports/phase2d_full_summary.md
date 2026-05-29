# Phase 2D Final Summary — All Tasks Results

*Generated: 2026-05-29*

## Task Completion Status

| Task | Owner | Status | Key Result |
|------|-------|--------|------------|
| **TSK-01** | trainer | ✅ done (2026-05-29) | MRE=0.495mm, SDR@2mm=98.2% |
| **TSK-02** | data-engineer | ✅ done (2026-05-29) | 240 boundary crops generated |
| **TSK-03** | trainer | ✅ done (2026-05-29) | val_loss=0.1431, 4.38M params refiner |
| **TSK-04** | trainer | ✅ done (2026-05-29) | Dice=0.8827 (+2.8% vs baseline) |
| **TSK-05** | trainer | ✅ done (2026-05-29) | MRE=1.568mm, SDR@2mm=82.2% (fusion) |


---
## Detailed Results

### TSK-01: Sliding Window Inference

- **Implementation:** `backend/app/services/analysis_service.py`
- **Technique:** 512×512 patches, stride=256 (50% overlap), Gaussian-weighted averaging
- **MRE: 0.495 mm** | **SDR@2mm: 98.2%**
- **Per-fold:** Fold1=0.416, Fold2=0.504, Fold3=0.564, Fold4=0.553, Fold5=0.441
- **Per-landmark (best to worst):**
  - Labial_crest: 0.311mm | Palatal_midroot: 0.320mm | Labial_midroot: 0.316mm
  - Palatal_crest: 0.360mm | Upper_apex: 0.472mm | Upper_tip: 0.515mm
  - ANS: 0.599mm | PNS: 0.629mm | LB: 0.662mm | PB: 0.763mm


### TSK-02: Generate Refiner Crops

- **Output:** 240 boundary crops (80 per class: Upper_incisor, Labial_bone, Palatal_bone)
- **Crop size:** 384×128px
- **Used by:** TSK-03 Stage 2 refiner training


### TSK-03: Train Stage 2 Lightweight Refiner

- **Architecture:** DeepLabV3Plus encoder (frozen) + MobileNetV2 decoder
- **Params:** 4.38M (lightweight, designed for fast inference)
- **Input:** 256×256 resized boundary crops
- **Loss:** 0.6×Dice + 0.4×Focal
- **Train/Val:** 189/51 crops, patient-level split
- **Best:** val_loss=0.1431 at epoch 40/60 (early stopping)
- **Note:** Not yet integrated into production pipeline


### TSK-04: Tversky + BoundaryDice Fine-Tuning

- **Architecture:** DeepLabV3Plus + resnet34, fine-tuned from 512px baseline
- **Loss:** 0.6×Tversky(α=0.7,β=0.3) + 0.4×BoundaryDice
- **Train/Val:** 291/71 records, patient-level split, 50 epochs (early stopping)
- **Result: Dice=0.8827** ← NEW PROJECT CHAMPION
- **Previous baseline:** 0.8588 → **Improvement: +0.0238 (+2.8%)**
- **Model path:** `models/tversky_deepLabV3plus_resnet34_20250529_20260529_094221/best_model.pt`


### TSK-05: Final Fusion (TSK-04 Model + Sliding Window)

- **Pipeline:** TSK-04 champion model + TSK-01 sliding window + geometric snapping
- **Result: MRE=1.568mm | SDR@2mm=82.2%** (63 validation images)
- **Per-landmark:**
  - Labial_midroot: 0.737mm (100%) | Upper_tip: 1.473mm (95.2%)
  - Palatal_midroot: 0.918mm (98.4%) | Labial_crest: 0.964mm (98.4%)
  - Palatal_crest: 0.967mm (95.2%) | Upper_apex: 1.471mm (77.8%)
  - LB: 2.00mm (74.6%) | PNS: 2.197mm (58.7%)
  - PB: 2.451mm (69.8%) | ANS: 2.501mm (54.0%)


---
## Cross-Task Comparison

| Metric | TSK-01 (baseline) | TSK-05 (fusion) |
|--------|-----------------|-----------------|
| MRE | **0.495 mm** | 1.568 mm |
| SDR@2mm | **98.2%** | 82.2% |

**Interesting finding:** The TSK-05 fusion (TSK-04 model + sliding window) produces
*worse* landmark MRE than TSK-01 (old baseline model + sliding window).
This suggests the TSK-04 model is better at segmentation but the landmark
localization pipeline (geometric snapping from segmentation → landmarks) is not
optimized for the Tversky-trained model's different output characteristics.


---
## Project Context

Reference benchmarks:
- CL-Detection2023 best MRE: 1.518 mm (38 landmarks, multi-center dataset)
- Current result: MRE=0.495mm (TSK-01, 10 landmarks, single-center, sliding window)

Phase 2D goal was to achieve 1024px-level precision without direct 1024px training.
Two approaches merged: TSK-01 inference strategy + TSK-04 training improvement.

**Next recommended step:** Investigate why TSK-05 fusion MRE regressed vs TSK-01 baseline.
The TSK-04 segmentation model (Dice=0.8827) should complement the landmark pipeline,
but the geometric snapping parameters may need retuning for the new model's output.
