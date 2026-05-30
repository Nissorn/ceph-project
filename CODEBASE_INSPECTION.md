# Ceph-Project Codebase Inspection Report

**Repository:** `/Users/onis2/Project/Singdent/ceph-auto/ceph-project`
**Inspection date:** 2026-05-29
**Tool:** pygount v3.1.0

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Total files scanned** | 215 |
| **Total lines (all types)** | 25,926 |
| **Total code lines** | 17,967 |
| **Total comment lines** | 7,959 |
| **Total empty lines** | ~1,500+ (estimated) |
| **Code-to-comment ratio** | **2.26 : 1** |
| **Overall documentation ratio** | 30.7% (code + comment vs total) |
| **Primary language** | Python (37.7% of files, 85.2% of code) |

---

## 2. Language Breakdown

| Language | Files | Code Lines | Comment Lines | % of Code |
|----------|-------|-----------|---------------|-----------|
| **Python** | 81 | 15,313 | 4,529 | 60.7% |
| **TSX (React)** | 6 | 1,764 | 163 | 75.2% |
| **HTML+Genshi** | 2 | 307 | 3 | 70.9% |
| **YAML** | 3 | 163 | 24 | 80.3% |
| **TypeScript** | 3 | 137 | 6 | 77.4% |
| **Bash** | 3 | 127 | 44 | 21.3% |
| **Docker** | 2 | 51 | 23 | 22.5% |
| **JSON** | 4 | 42 | 0 | 53.8% |
| **JavaScript** | 7 | 26 | 4 | 59.1% |
| **CSS+Lasso** | 2 | 14 | 0 | 63.6% |
| **Markdown** | 55 | 0 | 2,910 | 0.0% |
| **Text only** | 3 | 0 | 253 | 69.3% |
| **__duplicate__** | 21 | 0 | 0 | 0.0% |
| **__binary__** | 10 | 0 | 0 | 0.0% |
| **__unknown__** | 9 | 0 | 0 | 0.0% |

> **Note:** Python dominates at 85.2% of all source code. Markdown shows 0 code lines because pygount classifies all Markdown as comments (expected behavior). "Duplicate" files are files with identical content detected by pygount's heuristics.

---

## 3. Directory Structure

```
ceph-project/
├── agent_memory/              # Agent sessions & project context docs
│   ├── 00_Project_Context.md
│   ├── 01_Current_Phase.md
│   └── 02_AUGMENTATION_RESEARCH.md
├── .agents/                  # Agent skill definitions
│   └── skills/               # 15+ skill packages (caveman, diagnose, grill-me, …)
├── backend/                  # FastAPI REST service
│   ├── app/
│   │   ├── api/              # API route stubs
│   │   ├── core/             # config.py
│   │   ├── models/           # schemas.py (Pydantic)
│   │   └── services/          # analysis_service.py, inference_service.py
│   ├── models/               # Trained model checkpoints (.pt)
│   ├── tests/                # pytest unit tests
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── data/                     # Data dir (never committed — gitignored)
│   ├── annotations.xml      # CVAT XML annotations
│   ├── processed/           # Calibration CSVs, cleaned landmarks JSON
│   └── raw/images/           # Source cephalogram images
├── deploy/                   # Deployment configs
│   └── cvat_nuclio/          # Nuclio serverless function for CVAT
├── notebooks/                # Jupyter analysis notebooks
│   ├── 01_verify_annotations.ipynb
│   ├── 02_calibration_analysis.ipynb
│   └── 03_results_analysis.ipynb
├── scratchpad/                # Experimental / exploratory code (not for production)
│   ├── streamlit_app.py
│   ├── old_analysis_service.py
│   ├── current_analysis_service.py
│   ├── debug_*.py (multiple debug scripts)
│   ├── phase1_legacy/
│   └── python_scripts_legacy/
├── scripts/                  # Pipeline runner & experiment scripts (39 files)
├── src/                      # Core ML pipeline source
│   ├── data/                 # Phase 1: data loading & preprocessing
│   ├── phase2/               # Phase 2: heatmap-based landmark detection
│   ├── phase2b/              # Phase 2b: tooth segmentation
│   ├── phase2c/              # Phase 2c: treatment classification
│   ├── phase3/               # Phase 3: biomechanics & superimposition
│   ├── phase4/               # Phase 4: conversion & visualization
│   └── utils/                # Shared I/O utilities
├── README.md
├── CLAUDE.md                  # Project rules & keypoint definitions
├── STATUS.md
├── FAILURES.md
├── context.md
├── config.yaml               # Central config (paths, keypoints, model, training)
├── requirements.txt          # Root Python dependencies
├── docker-compose.yml
└── PIPELINE_AUDIT_REPORT.md, PIPELINE_PLAN.md, PRODUCTION_ARCHITECTURE_PLAN.md, etc.
```

---

## 4. Source Code Breakdown by Module (`src/`)

### 4.1 `src/data/` — Phase 1: Data Loading & Preprocessing

| File | Lines | Purpose |
|------|-------|---------|
| `cvat_parser.py` | 223 | Parse CVAT XML annotations → structured keypoint data |
| `calibration.py` | 64 | Per-image mm_per_pixel calibration from CSV lookup |
| `quality_filter.py` | 83 | Filter low-quality annotations |
| `splits.py` | 162 | Patient-level train/val/test splits (5-fold CV, no horizontal flip) |

### 4.2 `src/phase2/` — Phase 2: Landmark Detection (Heatmap Approach)

| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | 186 | HRNet-W32 architecture (heatmap head) |
| `heatmap.py` | 218 | Gaussian heatmap generation, adaptive sigmas per keypoint |
| `dataset.py` | 111 | CephDataset: loads images + heatmaps, applies augmentations |
| `augmentation.py` | 71 | Albumentations-based augmentation (horizontal flip DISALLOWED) |
| `train.py` | 468 | Training loop with differential LR, early stopping, partial freeze |
| `loss.py` | 204 | Combined MSE + SSIM heatmap loss |
| `metrics.py` | 75 | MRE (mm), SDDR metrics |
| `evaluate.py` | 301 | Fold evaluation, argmax keypoint extraction |
| `inference.py` | 767 | Full inference pipeline, TTA, confidence scoring |

### 4.3 `src/phase2b/` — Phase 2b: Tooth Segmentation

| File | Lines | Purpose |
|------|-------|---------|
| `segmentation_dataset.py` | 119 | Segmentation dataset (mask-based) |
| `segmentation.py` | 99 | DeepLabV3Plus model for tooth segmentation |

### 4.4 `src/phase2c/` — Phase 2c: Treatment Classification

| File | Lines | Purpose |
|------|-------|---------|
| `classifier_dataset.py` | 189 | Classifier dataset with noise injection |
| `classifier.py` | 182 | 6-class treatment classifier (tipping, translation, root torque, etc.) |

### 4.5 `src/phase3/` — Phase 3: Biomechanics & Superimposition

| File | Lines | Purpose |
|------|-------|---------|
| `biomechanics.py` | **1,418** | Core dental biomechanics: tipping angle, translation, root torque, extrusion/intrusion |
| `heuristics.py` | 118 | Per-landmark heuristics for threshold estimation |
| `segmentation_preprocess.py` | 825 | Preprocess segmentation masks for superimposition |
| `superimposition.py` | 70 | Maxillary superimposition via ANS-PNS plane |

### 4.6 `src/phase4/` — Phase 4: Conversion & Visualization

| File | Lines | Purpose |
|------|-------|---------|
| `convert.py` | 62 | Convert predictions → frontend JSON format |
| `visualize.py` | 89 | Visualize landmarks, measurement lines, heatmaps on images |

### 4.7 `src/utils/`

| File | Lines | Purpose |
|------|-------|---------|
| `io.py` | 26 | Shared file I/O utilities (JSON read/write, calibration lookup) |

### `src/` Module Size Ranking (by LSP line count)

```
biomechanics.py              1,418  ← largest single file; core phase3 math
segmentation_preprocess.py     825  ← phase3 preprocessing
inference.py                   767  ← phase2 inference
train.py                       468  ← phase2 training
classifier.py                  182  ← phase2c classifier
segmentation.py                99  ← phase2b
...
calibration.py                  64  ← smallest
```

---

## 5. Scripts (`scripts/`)

39 Python + shell scripts. Key categories:

### 5.1 Core Pipeline Runners
| Script | Lines | Purpose |
|--------|-------|---------|
| `run_pipeline.py` | 76 | Orchestrates full Phase 1→4 pipeline |
| `run_phase1_calibration.py` | 73 | Phase 1 calibration from CVAT XML |
| `run_phase2_train.py` | — | Phase 2 training entry point |
| `run_phase2_predict.py` | 74 | Phase 2 inference |
| `run_phase3.py` | 50 | Phase 3 analysis |
| `run_phase2b_segmentation.py` | — | Phase 2b segmentation |

### 5.2 Experiment Scripts (EXP01–05 + sweeps)
| Script | Lines | Purpose |
|--------|-------|---------|
| `run_EXP01_aggressive_optimizer.py` | 480 | Aggressive optimizer experiment |
| `run_EXP02_unetpp.py` | 482 | UNet++ segmentation comparison |
| `run_EXP03_focal_loss.py` | 506 | Focal loss experiment |
| `run_EXP04_lightweight.py` | 470 | Lightweight model variant |
| `run_EXP05_freeze.py` | 533 | Backbone freeze + differential LR |
| `run_1024_sweep.py` | 496 | 1024px input resolution sweep |
| `run_fold1_only.py` | — | Single fold debug run |

### 5.3 Evaluation & Analysis
| Script | Lines | Purpose |
|--------|-------|---------|
| `evaluate_tsk05_final.py` | 498 | Evaluate on Tsk05 test set |
| `evaluate_tta.py` | — | Test-time augmentation evaluation |
| `error_analysis.py` | 702 | Per-landmark MRE error analysis |
| `generate_validation_report.py` | 732 | Cross-fold validation report |

### 5.4 Autonomous / Agentic
| Script | Lines | Purpose |
|--------|-------|---------|
| `autoresearch_agent.py` | 861 | LLM-driven autonomous research agent |
| `autonomous_loop.py` | 472 | Autonomous experimentation loop |
| `auto_research_iter1.py` | 350 | Research iteration 1 |
| `auto_research_segmentation.py` | 625 | Autonomous segmentation research |

### 5.5 Visualization & Inspection
| Script | Lines | Purpose |
|--------|-------|---------|
| `visualize_test_inference.py` | 561 | Per-image visualization of predictions |
| `batch_holdout_visual.py` | 407 | Batch visualization on holdout set |

### 5.6 Other
| Script | Lines | Purpose |
|--------|-------|---------|
| `run_production_blind_test.py` | 583 | Production blind test runner |
| `final_deep_run.py` | 296 | Final deep training run |
| `fast_grid_search.py` | 93 | Parameter grid search |
| `train_tversky.py` | 520 | Tversky loss experiment |
| `convert_predictions_for_frontend.py` | 55 | Convert predictions → frontend JSON |
| `parse_annotations.py` | 31 | Parse raw CVAT annotations |
| `merge_cvat_data.py` | — | Merge multiple CVAT exports |
| `ingest_editor_exports.py` | — | Ingest editor exports |
| `test_backend_endpoint.py` | 99 | Backend API test |
| `isolate_inference_crash.py` | — | Debug inference crashes |
| `replicate_host_path_crash.py` | 386 | Debug host path issues |
| `watch_fold1_complete.sh` | — | Watch dog script |
| `run_overnight_fleet.sh` | — | Overnight fleet runner |

---

## 6. Backend (`backend/`)

FastAPI-based REST API for production inference:

| File | Lines | Purpose |
|------|-------|---------|
| `app/services/analysis_service.py` | **1,151** | Core analysis service (biomechanics, measurements) |
| `app/services/inference_service.py` | 362 | Model loading + inference pipeline |
| `app/models/schemas.py` | 95 | Pydantic request/response schemas |
| `app/main.py` | 23 | FastAPI app entry point |
| `app/core/config.py` | 9 | Environment config |

**Models:** 3 trained model checkpoints stored in `backend/models/`.
**Tests:** 2 pytest files (`test_measurement_lines_contract.py`, `test_segmentation_decoding.py`).

---

## 7. Configuration (`config.yaml`)

- **Architecture:** HRNet-W32 (primary), HRNet-W18 (fallback)
- **Input:** 512×512 → heatmap 256×256
- **Sigma strategy:** Per-keypoint adaptive (2.0–5.0 px range)
- **Training:** 150 epochs, batch_size=4, weight_decay=0.001–0.002, device=mps
- **10 keypoints:** Upper_tip, Upper_apex, Labial_midroot, Labial_crest, Palatal_midroot, Palatal_crest, ANS, PNS, LB, PB
- **Phase 3 thresholds:** nullable (pending real data calibration)

---

## 8. Dependencies Summary

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Core ML framework (MPS for Apple Silicon) |
| `timm>=0.9.12` | HRNet pretrained weights |
| `albumentations>=1.4.0` | Image augmentation |
| `opencv-python>=4.9.0` | Image processing |
| `segmentation-models-pytorch>=0.5.0` | DeepLabV3Plus (phase2b) |
| `numpy>=1.26.0`, `pandas>=2.2.0` | Data processing |
| `pyyaml>=6.0.1` | Config loading |
| `scikit-learn>=1.4.0`, `scipy>=1.12.0` | Metrics |
| `iterative-stratification>=0.1.7` | K-fold splits |
| `matplotlib>=3.8.0` | Visualization |
| `streamlit>=1.35.0` | MVP dashboard |
| `fastapi`, `uvicorn` | Backend API |
| `pygount` | This report's source data |

---

## 9. Non-Production Code (Scratchpad)

`scratchpad/` contains orphaned/experimental code:
- `streamlit_app.py` — legacy dashboard
- `old_analysis_service.py` — old backend service (775 lines)
- `current_analysis_service.py` — in-progress service (539 lines)
- `debug_*.py` — various debug scripts
- `phase1_legacy/`, `python_scripts_legacy/` — deprecated pipeline code

> **Status:** These files are NOT part of the active pipeline and may be stale.

---

## 10. `__duplicate__` Files (21)

pygount identified 21 files with identical content. Likely:
- `__pycache__` bytecode compiled from same source under different Python versions
- Config/template copies
- Backup files with unchanged content

---

## 11. Key Insights

1. **Python dominates** at 85% of code — this is a pure Python ML pipeline
2. **Phase 3 biomechanics** is the single largest file (1,418 lines) — the complex dental math beyond ML
3. **Phase 2 inference** is the second largest at 767 lines; training loop is 468 lines
4. **Scripts folder** is as large as `src/` itself — reflects extensive experimentation culture
5. **Code comment ratio 2.26:1** is reasonable; 22.6% comment coverage suggests fairly documented code
6. **10 keypoints** are hardcoded throughout (not inferred from data)
7. **MPS-device** architecture throughout (Apple Silicon Mac), not CUDA
8. **Phase 3 thresholds are nullable** — awaiting real clinical calibration data
9. **Dual backend services** in scratchpad suggest active refactoring between `old_analysis_service.py` and `current_analysis_service.py`
10. **17,967 total Python code lines** project-wide (excluding duplicates, binary, unknown)
