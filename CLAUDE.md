# CLAUDE.md — Cephalometric Landmark Detection System
# Singapodent Internship Project

## Project Overview

4-phase AI pipeline to detect 8 anatomical landmarks on lateral cephalogram X-rays, measure tooth movement in mm between pre-treatment (T1) and post-treatment (T2) images, and classify the orthodontic treatment type. Dataset: 52 patients × 2 images = 104 JPGs from Singapodent clinic (Ho Chi Minh City, Vietnam). Annotations in CVAT XML 1.1 format.

## Hardware — MPS ONLY

- Machine: Mac Mini M4, Apple Silicon
- **Use `device = torch.device("mps")` everywhere. NEVER use `.cuda()` or `"cuda"`.**
- **`num_workers=0` in all DataLoaders** — MPS does not support multiprocessing workers.

## CRITICAL Rules — Never Violate

1. **No horizontal flip augmentation** — lateral cephalograms have strict anatomical orientation. `horizontal_flip: false` in config.yaml is the law.
2. **Patient-level splits** — T1 and T2 of the same patient must be in the same fold. Split on `patient_id`, never on image index or filename.
3. **Data is gitignored** — `data/raw/` and `data/processed/` are never committed. No `.jpg`, `.xml`, `.json` (data), `.csv` (data), or `.pth` files in git.
4. **Calibration is per-image** — `mm_per_pixel` differs per image. Always look up from `calibration.csv` by `image_id`. No global constant.
5. **Phase 3 thresholds are nullable** — `tipping_threshold_deg` and `translation_threshold_mm` are `null` until Dr. confirms. Return `"pending_threshold"` when null.
6. **All paths via config.yaml** — no hardcoded paths anywhere. Load config with `src/utils/io.py:load_config()`.

## Keypoint Order — Hardcoded as `KEYPOINT_NAMES`

```python
KEYPOINT_NAMES = [
    "Upper_tip",       # 0 — crown tip
    "Upper_apex",      # 1 — root apex
    "Labial_midroot",  # 2
    "Labial_crest",    # 3
    "Palatal_midroot", # 4
    "Palatal_crest",   # 5
    "ANS",             # 6 — superimposition reference
    "PNS",             # 7 — superimposition reference
]
```

This order is confirmed and must never be inferred from file content.

## Current Phase Status

| Phase | Status | Blocker |
|-------|--------|---------|
| Phase 1 — Parsing & Calibration | Ready to implement | None |
| Phase 2 — HRNet Detection | Scaffold only | Need ~20+ annotated images |
| Phase 3 — Classification | Design done | Dr. must confirm thresholds |
| Phase 4 — Clinical Output | Design done | Depends on Phase 2+3 |

## File Structure

```
src/
  phase1/   cvat_parser.py, calibration.py, export.py
  phase2/   dataset.py, model.py, augmentation.py, heatmap.py, train.py, metrics.py
  phase3/   superimposition.py, heuristics.py
  phase4/   convert.py, visualize.py
  utils/    io.py
scripts/    run_phase1.py, run_phase2_train.py, run_phase2_predict.py, run_phase3.py, run_pipeline.py
config.yaml — single source of truth
context.md  — full project journal (read this for deeper context)
```

## Common Commands

```bash
# Parse annotations and extract calibration
python scripts/run_phase1.py --config config.yaml

# Train landmark detector (LOPO-CV)
python scripts/run_phase2_train.py --config config.yaml

# Predict landmarks on a single image
python scripts/run_phase2_predict.py --config config.yaml --image data/raw/images/Patient01_T1.jpg

# Classify treatment for one patient pair
python scripts/run_phase3.py --config config.yaml --patient Patient01

# Run full pipeline end-to-end
python scripts/run_pipeline.py --config config.yaml --patient Patient01
```

## Key Papers

- CL-Detection2023: arxiv:2409.15834 — benchmark, HRNet best performer
- Rank-1 method: arxiv:2309.17143 — super-resolution heatmap head
