# STATUS.md — Current State Snapshot
_Updated: 2026-05-06_

## What is done
- Phase 1 calibration: `scripts/run_phase1_calibration.py` — 104/104 pass QC → `data/processed/calibration.csv`
- `src/data/cvat_parser.py` — parses calibration + skeleton (10 keypoints) + polygons + tags
- mm/pixel: mean 0.0984, range [0.0974, 0.0990] — single scanner, extremely consistent
- **[P1 DONE]** Fixed 10-keypoint mismatch: `src/phase2/dataset.py` (KEYPOINT_NAMES 10 entries, NUM_KEYPOINTS=len(...)) + `src/phase2/model.py` (NUM_KEYPOINTS=10, docstring updated)
- Full pipeline plan written: `PIPELINE_PLAN.md` — input/output specs for all phases + 9 priorities

## Dataset
- **Total:** 382 images (104 paired T1+T2 from 52 patients + 278 T1-only)
- **Incoming:** Dr. will provide 300+ additional ceph images
- **Annotations:** ~2/104 have 10-keypoint skeleton; 0/104 have polygons; calibration 104/104 done

## What is next (from PIPELINE_PLAN.md)
- **P2:** Create `scripts/parse_annotations.py` → generate `data/processed/landmarks_clean.json`
- **P3:** Implement `src/data/splits.py` — 5-fold StratifiedKFold + 15% holdout, patient-aware
- **P4:** Upgrade `src/phase2/augmentation.py` — ±10° rotation, add ElasticTransform/GaussNoise/GridDistortion
- **P5:** Create `src/phase2/loss.py` — Adaptive Wing Loss (replaces MSE in train.py)
- **P6:** Create `notebooks/04_augmentation_preview.ipynb` — visual verification
- **P7:** Create `src/phase2b/` — U-Net segmentation scaffold
- **P8:** Create `src/phase2c/` — EfficientNet-B3 classifier scaffold
- **Waiting on Dr.**: skeleton annotations (~2/104 done) — need 20+ for training

## Active blockers
- Dr. annotation: need 20+ skeleton annotations before meaningful training
- `data/processed/landmarks_clean.json` not generated yet (P2)
- Phase 3 thresholds: tipping_threshold_deg and translation_threshold_mm still null (pending Dr.)

## Known bugs remaining
- `src/phase2/dataset.py:88` — uses `rec["file_name"]` but `cvat_parser.py` stores key as `"filename"` (no underscore) → KeyError at training time. Fix before first training run.

## Key file locations
- Annotation XML: `data/annotations.xml` (NOT data/raw/annotations/)
- Active code: `src/data/` (NOT src/phase1/ — that is legacy)
- Config: `config.yaml` (annotation_file corrected, num_keypoints=10)
- Pipeline plan: `PIPELINE_PLAN.md`
- Failures log: `FAILURES.md` — read before starting any task
