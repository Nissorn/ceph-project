# STATUS.md — Current State Snapshot
_Updated: 2026-05-08 (P9 + P10 + P11 implemented — full mock pipeline complete)_

## What is done
- **[P9 DONE]** Medical Logic Engine (Biomechanics) — `src/phase3/biomechanics.py`
  - U1-PP angle, LB-Apex distance, PB-Apex distance calculation
  - Zhang et al. 2021 classification (Labial/Midway/Palatal × <105/105-115/>115°)
  - Mock landmark generator + comprehensive built-in self-tests (all passing)
- **[P10 DONE]** Evaluation Metrics — `src/phase2/evaluate.py`
  - `calculate_mre()` — Mean Radial Error in mm across all images and landmarks
  - `calculate_sdr()` — Successful Detection Rate for thresholds [2.0, 2.5, 3.0, 4.0] mm
  - Robust to partially-annotated images; built-in self-tests all passing
- **[P11 DONE]** Streamlit MVP Dashboard — `app.py`
  - Medical dark-theme UI (Inter font, glassmorphism cards, custom CSS)
  - File uploader + 2-second mock inference spinner
  - OpenCV landmark visualisation with dynamic scaling to any image size
  - Draws U1 axis (Upper-tip→Upper-apex), palatal plane (ANS→PNS), LB–PB corridor
  - `st.metric` for angle/distances; styled cards for all 5 classification fields
  - Sidebar calibration input (mm/px), debug JSON expander, clinical disclaimer
- Phase 1 calibration: `scripts/run_phase1_calibration.py` — 104/104 pass QC → `data/processed/calibration.csv`
- `src/data/cvat_parser.py` — parses calibration + skeleton (10 keypoints) + polygons + tags
- mm/pixel: mean 0.0984, range [0.0974, 0.0990] — single scanner, extremely consistent
- **[P1 DONE]** Fixed 10-keypoint mismatch: `src/phase2/dataset.py` + `src/phase2/model.py` (NUM_KEYPOINTS=10)
- **[P2 DONE]** Created `scripts/parse_annotations.py` → generated `data/processed/landmarks_clean.json`
- **[P3 DONE]** Implemented `src/data/splits.py` — 5-fold MultilabelStratifiedKFold + 15% holdout, patient-aware
- **[P4 DONE]** Upgraded `src/phase2/augmentation.py` & `config.yaml` — ±10° via `A.Affine`, added ElasticTransform, GaussNoise, GridDistortion, Perspective (Albumentations 2.x clean, no warnings)
- **[P5 DONE]** Created `src/phase2/loss.py` — AdaptiveWingLoss (Wang et al. ICCV 2019); wired into `src/phase2/train.py` replacing MSELoss
- **[P6 SKIPPED]** Augmentation preview notebook deferred — requires at least 1 annotated image with visible keypoints
- **[P7 DONE]** Created `src/phase2b/segmentation.py` + `src/phase2b/segmentation_dataset.py` — U-Net scaffold with Dice+BCE loss and polygon→mask rasterisation
- **[P8 DONE]** Created `src/phase2c/classifier.py` + `src/phase2c/classifier_dataset.py` — EfficientNet-B3 multi-label scaffold with pos_weight computation, T1-only + Quality_Reject filtering
- Full pipeline plan written: `PIPELINE_PLAN.md` — input/output specs for all phases + 9 priorities
- **[AUDIT 2026-05-07]** Full code audit + 5 bugs fixed; dry-run pipeline test passes; `--debug` mode added to training script

## Planned Work (While Waiting for Annotations)
- *(All P9–P11 planned items are now done. Awaiting Dr. annotations for model training.)*

## Dataset
- **Total:** 382 images (104 paired T1+T2 from 52 patients + 278 T1-only)
- **Incoming:** Dr. will provide 300+ additional ceph images
- **Annotations (current XML):** 2 images exported from CVAT — 0 skeleton landmarks, 2 polygons, 2 calibration
- **Calibration CSV:** 104/104 images calibrated (from earlier full export)

## What is next
- **Waiting on Dr.**: export CVAT XML with skeleton annotations — need 20+ for meaningful training
- **P6 (deferred):** `notebooks/04_augmentation_preview.ipynb` — needs ≥1 annotated image
- **Optional:** Install `segmentation-models-pytorch` when ready to train segmentation
- When annotations arrive: `python scripts/run_phase2_train.py --debug --max-images 10` to smoke-test, then full run

## Active blockers
- Dr. annotation: need 20+ skeleton annotations before meaningful training
- Phase 3 thresholds: tipping_threshold_deg and translation_threshold_mm still null (pending Dr.)

## Known bugs remaining
- None. All bugs from audit logged in FAILURES.md and fixed.

## Key file locations
- Annotation XML: `data/annotations.xml` (NOT data/raw/annotations/)
- Active code: `src/data/` (NOT src/phase1/ — that is legacy)
- Config: `config.yaml` (annotation_file corrected, num_keypoints=10)
- Pipeline plan: `PIPELINE_PLAN.md`
- Failures log: `FAILURES.md` — read before starting any task
- Python venv: `venv/` — always `source venv/bin/activate` before running Python
- **Planned biomechanics:** `src/phase3/biomechanics.py`
- **Planned evaluation:** `src/phase2/evaluate.py`
- **Planned dashboard:** `app.py`