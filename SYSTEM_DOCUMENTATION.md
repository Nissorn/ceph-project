# Cephalometric Landmark Detection — System Documentation

**Project:** SingDental Ceph Auto ML Pipeline  
**Version:** Phase 2D (Champion: TSK-04 Tversky+BoundaryDice Dice=0.8827)  
**Last Updated:** 2026-05-29  
**Save Location:** `.hermes/plans/2026-05-29_234500-system-documentation.md`

---

## 1. Project Overview

### 1.1 Purpose

Automate detection of 10 cephalometric landmarks on lateral radiographs for orthodontic treatment planning. The system processes a pair of cephalograms (T1=pre-treatment, T2=post-treatment) per patient, computes biomechanical measurements, classifies treatment type, and outputs web-ready JSON for a clinical dashboard.

### 1.2 Key Anatomical Landmarks (10, Hardcoded — Never Infer from File)

| # | Name | Clinical Role |
|---|------|---------------|
| 0 | Upper_tip | Incisal edge of upper central incisor |
| 1 | Upper_apex | Root apex of upper central incisor |
| 2 | Labial_midroot | Labial root midpoint |
| 3 | Labial_crest | Labial alveolar bone crest |
| 4 | Palatal_midroot | Palatal root midpoint |
| 5 | Palatal_crest | Palatal alveolar bone crest |
| 6 | ANS | Anterior Nasal Spine — maxillary superimposition reference |
| 7 | PNS | Posterior Nasal Spine — maxillary superimposition reference |
| 8 | LB | Labial bone level |
| 9 | PB | Palatal bone level |

ANS(6)–PNS(7) form the **maxillary superimposition reference plane**.

### 1.3 Hardware Constraints

- **Device:** `torch.device("mps")` on Mac M4. NEVER `.cuda()` or `"cuda"`.
- **DataLoader:** `num_workers=0` (MPS restriction on Mac M4).
- **Protected GPUs (server):** 0, 3, 7 — never used by our jobs.

---

## 2. Project Structure

```
ceph-project/
├── config.yaml                    # All paths and hyperparams — no hardcoding
├── CLAUDE.md                      # Rules, keypoint names, run commands
├── STATUS.md                      # Current phase, experiment results, GPU status
├── FAILURES.md                    # Historical bugs and fixes (read before coding)
├── context.md                     # Deep background (read only if needed)
│
├── src/
│   ├── data/
│   │   ├── cvat_parser.py        # Parse CVAT XML 1.1 exports
│   │   ├── calibration.py        # Compute mm_per_pixel from 30mm ruler
│   │   ├── quality_filter.py    # Filter low-quality images
│   │   └── splits.py             # Patient-level train/val/test splits
│   │
│   ├── phase2/                   # LANDMARK DETECTION (HRNet-W32)
│   │   ├── model.py              # HRNet-W32 backbone + HeatmapHead + CBAM + EUPE uncertainty head
│   │   ├── heatmap.py            # Gaussian heatmap encoding + soft-argmax decoding
│   │   ├── dataset.py            # CephalometricDataset (patient-level split, no flip)
│   │   ├── augmentation.py      # Albumentations: rotation, zoom, CLAHE, NO horizontal flip
│   │   ├── loss.py               # AdaptiveWingLoss + EUPELoss (uncertainty-weighted)
│   │   ├── train.py              # Training loop with differential LR, early stopping
│   │   ├── evaluate.py           # MRE/SDR evaluation, 5-fold cross-validation
│   │   ├── metrics.py            # Mean Radial Error, Success Detection Rate
│   │   └── inference.py          # Cohort-wide inference with geometric constraints
│   │
│   ├── phase2b/                  # SEGMENTATION (DeepLabV3+)
│   │   ├── segmentation_dataset.py
│   │   └── segmentation.py       # DeepLabV3Plus + resnet34, 4-class (BG+3 structures)
│   │
│   ├── phase2c/                  # CLASSIFIER (treatment classification)
│   │   ├── classifier_dataset.py
│   │   └── classifier.py
│   │
│   ├── phase3/                   # BIOMECHANICS + SUPERIMPOSITION
│   │   ├── biomechanics.py       # U1-PP angle, root position classification (Zhang 2021)
│   │   ├── superimposition.py    # T1/T2 landmark alignment
│   │   ├── heuristics.py         # Treatment type classification from angles
│   │   └── segmentation_preprocess.py  # Mask priority layering (Upper_incisor > Palatal_bone)
│   │
│   └── phase4/                   # OUTPUT
│       ├── convert.py            # Backend JSON format conversion
│       └── visualize.py          # PNG overlay generation
│
├── backend/
│   └── app/
│       ├── main.py               # FastAPI app entry point
│       ├── api/v1/endpoints.py  # /analyze, /health, /metrics endpoints
│       ├── core/config.py        # Environment config
│       ├── models/schemas.py     # Pydantic request/response models
│       └── services/
│           ├── analysis_service.py  # Singleton model loading, inference, geometric snapping
│           └── inference_service.py
│
├── scripts/
│   ├── run_phase1_calibration.py  # Phase 1: parse CVAT XML → calibration.csv + landmarks_clean.json
│   ├── run_phase2_train.py        # Phase 2 landmark training
│   ├── run_phase2b_segmentation.py # Phase 2b segmentation training
│   ├── run_phase3.py              # Phase 3 biomechanics
│   ├── run_pipeline.py            # Full pipeline orchestrator
│   └── [30+ experiment scripts]    # EXP01-05, TTA, fast_grid_search, etc.
│
└── frontend/                      # Astro web app
```

---

## 3. Data Pipeline

### 3.1 Annotation Format

CVAT XML 1.1 export — three annotation types:
- **`Calibration_30mm`** polyline: 2 endpoints of a 30mm ruler object
- **`Incisor_Maxilla_Complex`** skeleton: 10 labeled points (Upper_tip, Upper_apex, etc.)
- **`Upper_incisor` / `Labial_bone` / `Palatal_bone`** polygons: segmentation masks

**File location:** `data/annotations.xml` (NOT `data/raw/annotations/`)

### 3.2 Phase 1 — Calibration (One-time, Re-run on New Exports)

```bash
python3 scripts/run_phase1_calibration.py --cvat_xml data/annotations.xml --output_dir data/processed/
```

**Outputs:**
- `data/processed/calibration.csv` — mm_per_pixel per image
- `data/processed/landmarks_clean.json` — clean landmark records

**Key rules:**
- `mm_per_pixel` varies **per image** — always look up from `calibration.csv` by `image_id`. Never use a global constant.
- Patient-level split: T1+T2 of the same patient always in the same fold.

### 3.3 Per-Image Calibration

```python
mm_per_pixel = 30.0 / pixel_distance(calibration_pt1, calibration_pt2)
```

All landmark coordinates are converted from pixels to mm using this per-image scale factor.

---

## 4. Phase 2 — Landmark Detection

### 4.1 Architecture: HRNet-W32 + CBAM + HeatmapHead

**Backbone:** HRNet-W32 (timm `hrnet_w32`), pretrained on COCO, `global_pool=""` → outputs `[B, 2048, 16, 16]`

**HeatmapHead:**
1. `Conv2d(2048→256, 3×3) + BN + ReLU`
2. **CBAM** (Convolutional Block Attention Module): Channel Attention → Spatial Attention
3. 4× `ConvTranspose2d(256→256, 4×4, stride=2)` → 16→32→64→128→256
4. `Conv2d(256→10, 1×1)` → heatmaps `[B, 10, 256, 256]`

**EUPE Uncertainty Head:**
- `AdaptiveAvgPool2d(1)` → `Linear(256→10)` → softplus → `[B, 10]` σ_k values
- Used in loss: `L_eupe = (1/σ²)·L_regression + λ·log(σ)`

**Forward:** `[B, 3, 512, 512] → (heatmaps [B, 10, 256, 256], uncertainty [B, 10])`

### 4.2 Heatmap Encoding (`encode_heatmaps`)

- Gaussian heatmaps with **per-landmark adaptive sigma** (from `config.yaml`):
  - Small sigma (sharp): Upper_tip(2.0), Labial_crest(2.0), Palatal_crest(2.0) — precision for easy landmarks
  - Large sigma (diffuse): ANS(4.5), PNS(4.5), PB(5.0) — forgiving for low-contrast posterior landmarks
- Center clamped to `[1, size-2]` so Gaussian bleeds past edges
- Kernel size: `2*ceil(3*sigma)+1` (always odd for symmetry)

### 4.3 Heatmap Decoding (`SoftArgmax2D`)

- **Temperature=0.1** (NOT 10.0 — that caused beta≈22025, systematic 10mm center bias)
- `beta = softplus(0.1) ≈ 0.1` → proper spatial selectivity
- `coords = weighted_average(position, softmax(beta * sigmoid(heatmap)))`
- Confidence = `max(sigmoid(heatmap))`

### 4.4 Loss Functions

**AdaptiveWingLoss:**
- Wing-shaped: non-linear for small errors (<θ=0.5), linear for large errors
- Masked by `valid_mask`: only annotated landmarks contribute
- Normalized by **number of valid keypoints** (NOT H×W — that killed gradients)

**EUPELoss (joint uncertainty):**
```
L_eupe = Σ_k [(1/σ_k²)·L_k + λ·log(σ_k)]
```
- Easy landmarks (high contrast) → small σ → high weight
- Hard landmarks (low contrast) → large σ → low weight
- λ=0.1 prevents σ→0 (mode collapse)

### 4.5 Training Configuration

```yaml
training:
  architecture: "hrnet_w32"
  pretrained_source: "coco"
  input_size: [512, 512]
  heatmap_size: [256, 256]
  sigma: 3.0
  landmark_sigmas: [2.0, 3.5, 2.5, 2.0, 3.0, 2.0, 4.5, 4.5, 4.0, 5.0]
  batch_size: 4
  epochs: 150
  lr: 0.001
  weight_decay: 0.002
  freeze_backbone: false
  partial_freeze: true
  backbone_lr: 0.0001
  head_lr: 0.0001
  early_stopping_patience: 15
  eval_strategy: "5fold"
  k_folds: 5
```

### 4.6 Augmentation

- Rotation: ±15°
- Zoom: ±20%
- Brightness/Contrast: ±20%
- CLAHE: enabled
- **NO horizontal flip** — breaks lateral cephalogram anatomy (enforced in augmentation.py)

### 4.7 Evaluation Metrics

- **MRE (Mean Radial Error):** Average Euclidean distance in mm across all landmarks
- **SDR@Xmm (Success Detection Rate):** % of landmarks within X mm of ground truth
- Thresholds: 2.0, 2.5, 3.0, 4.0 mm
- Per-landmark breakdown always reported

---

## 5. Phase 2B — Segmentation

### 5.1 Architecture: DeepLabV3Plus + ResNet34

4-class segmentation:
- Class 0: Background
- Class 1: Upper_incisor
- Class 2: Labial_bone
- Class 3: Palatal_bone

**Champion model (TSK-04):** Dice=0.8827 (Tversky loss + BoundaryDice fine-tuning)

### 5.2 Loss Functions

- **Tversky Loss:** `Tversky(α=0.7, β=0.3)` — penalizes FN more than FP (better for imbalanced)
- **BoundaryDice:** edge-aware Dice for crisp mask boundaries
- Final: `0.6 × Tversky + 0.4 × BoundaryDice`

### 5.3 Sliding Window Inference (Pipeline B)

For high-resolution images, segmentation runs via overlapping 512×512 patches:
- Window size: 512px, stride: 256px (50% overlap)
- Gaussian weighting (σ=128) for seamless stitching
- Landmark model stays at 512×512

---

## 6. Phase 2C — Treatment Classifier

- **Backbone unfreeze:** epoch 20
- **Stage 2 LR:** 1e-5
- **Training noise injection:** σ=15px (approximates expected MRE ≈ 1.5mm)
- **Minimum class support:** 5 samples (classes below this flagged `insufficient_data`)

---

## 7. Phase 3 — Biomechanics

### 7.1 Metrics Calculation (`calculate_metrics`)

- **U1-PP angle:** angle between upper incisor axis (Upper_tip→Upper_apex) and palatal plane (ANS→PNS), in degrees
- **LB/PB apex distance:** perpendicular distance from LB/PB to the tooth axis, in mm (using `mm_per_pixel` from calibration.csv)

### 7.2 Treatment Classification (Zhang 2021)

Based on U1-PP angle zone + root apex position:

| U1-PP Angle | < 105° | 105–115° | > 115° |
|-------------|--------|----------|--------|
| **Labial** | Controlled tipping + torque control | Light controlled tipping | Controlled tipping (high risk) |
| **Midway** | Controlled proclination | Bodily movement (best) | Controlled tipping + torque |
| **Palatal** | Careful movement | Bodily movement + caution | Controlled tipping + apex control |

### 7.3 Phase 3 Thresholds (Nullable)

```yaml
phase3:
  tipping_threshold_deg: null    # Dr. to set
  translation_threshold_mm: null # Dr. to set
```

When null, return `"pending_threshold"` — **never invent defaults**.

---

## 8. Phase 3 — Geometric Snapping

Applied in `src/phase2/inference.py` and `backend/app/services/analysis_service.py`:

### 8.1 Mask Priority Layering

Upper_incisor (class 1) wins over Palatal_bone (class 3):
```
Palatal_bone_corrected = Palatal_bone AND (NOT Upper_incisor)
```
Result: zero overlap, zero gap between structures.

### 8.2 Crest Snapping

- **Labial_crest (idx 3)** → most coronal (min y) point on Labial_bone contour within ±60px of prediction
- **Palatal_crest (idx 5)** → most coronal point on Palatal_bone contour within ±60px

### 8.3 Midroot Snapping

- **Labial_midroot (idx 2)** → rightmost (max x) point on Upper_incisor contour (labial surface)
- **Palatal_midroot (idx 4)** → leftmost (min x) point on Upper_incisor contour (palatal surface)

### 8.4 ANS/PNS Snapping

- **ANS (idx 6)** → nearest point on Palatal_bone contour
- **PNS (idx 7)** → nearest point on Palatal_bone contour (maxillary suture reference)

---

## 9. Backend API

### 9.1 AnalysisService (Singleton)

Both models loaded once at startup, held in memory:

**`/analyze` (POST)**
1. Receive image binary stream
2. Run landmark inference (HRNet-W32) → raw coords
3. Run segmentation (DeepLabV3+ TSK-04) → raw masks
4. Resolve mask overlaps (priority layering)
5. Apply geometric snapping (crest, midroot, ANS/PNS)
6. Extract corrected polygons per class
7. Compute biomechanical metrics (Phase 3)
8. Classify treatment type
9. Return JSON: `{landmarks, masks, polygons, metrics, classification}`

**`/health` (GET)**
- Returns model loading status, device info

**`/metrics` (GET)**
- Returns current validation metrics

### 9.2 Dynamic Device Selection

```python
if sys.platform == "darwin" and platform.machine() == "arm64":
    # Native Mac Metal — use MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    # Linux x86_64 Docker — force CPU (MPS unavailable on x86, avoids qemu crashes)
    device = "cpu"
```

### 9.3 Apple Silicon Docker Stabilization

```python
torch.set_num_threads(1)  # prevents thread-scheduling segfaults in qemu
```

---

## 10. Configuration Management

**Single source of truth: `config.yaml`**

All paths, hyperparams, and thresholds defined there. No hardcoding in Python files.

Key paths:
```yaml
data.raw_dir: "data/raw"
data.processed_dir: "data/processed"
data.annotation_file: "data/annotations.xml"
data.image_dir: "data/raw/images"
data.landmarks_json: "data/processed/landmarks_clean.json"
data.calibration_csv: "data/processed/calibration.csv"
```

---

## 11. Data Never in Git

The following are **git-ignored** (`.gitignore`):
- `.jpg`, `.png` images
- `.xml` (CVAT exports)
- `.json` (annotation data)
- `.csv` (calibration data)
- `.pth` (model checkpoints)
- `outputs/`, `models/`, `data/`

Only processed artifacts (`.py`, `.yaml`, `.md`) are committed.

---

## 12. Critical Rules Summary

| Rule | Reason |
|------|--------|
| No horizontal flip | Breaks lateral cephalogram anatomy |
| Patient-level split | T1+T2 same patient always same fold |
| Per-image calibration | `mm_per_pixel` varies per image |
| No hardcoded paths | All from `config.yaml` |
| Phase 3 thresholds nullable | Return `"pending_threshold"` when null |
| 10 keypoints hardcoded | Never infer from annotation file |
| `num_workers=0` | MPS restriction on Mac M4 |
| `torch.device("mps")` | Never `.cuda()` or `"cuda"` |

---

## 13. File-to-File Dependencies

```
CVAT XML (data/annotations.xml)
  └── cvat_parser.py
        ├── calibration.py → calibration.csv
        └── landmarks_clean.json → phase2/dataset.py

calibration.csv ──look up by image_id──► biomechanics.py (mm_per_pixel)
                                        └── analysis_service.py (geometric snapping)

config.yaml ──look up by key──► all scripts, train.py, inference.py

HRNet-W32 (timm) ──► model.py (HeatmapHead + CBAM + EUPE)
DeepLabV3+ (smp) ──► segmentation.py / inference.py (sliding window)

inference.py ──► backend/analysis_service.py (same logic, FastAPI wrapper)
              ──► phase2/inference.py (standalone cohort inference)

biomechanics.py (MetricsResult) ──► backend/analysis_service.py (combine with snapping)
```

---

## 14. Known Experimental Results

| Exp | Architecture | Encoder | LR | Val Dice | Status |
|-----|-------------|---------|-----|---------|--------|
| Baseline | DeepLabV3+ | resnet34 (512px) | 1e-3 | **0.8588** | Completed |
| EXP-02 | UNetPlusPlus | resnet50 | 3e-4 | 0.7966 | Completed |
| EXP-03 | UNetPlusPlus | resnet50 | 3e-4 | 0.5416 | Completed |
| EXP-04 | DeepLabV3+ | efficientnet-b4 | 3e-4 | 0.5202 | Completed |
| EXP-01 | DeepLabV3+ | resnet50 (1024px) | 5e-4 | 0.5319 | Completed |
| EXP-05 | DeepLabV3+ | resnet50 | 1e-5 | 0.2717 | Completed |
| EXP-00 | DeepLabV3+ | resnet50 (1024px) | 1e-4 | 0.2600 | Aborted |

**Champion: TSK-04 (Tversky+BoundaryDice, Dice=0.8827)**
**Phase 2D TSK-05 Final Evaluation: MRE=1.568mm, SDR@2mm=82.2%**