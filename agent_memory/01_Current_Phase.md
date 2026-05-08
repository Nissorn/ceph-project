# 01_Current_Phase.md — Immediate Status & Next Action

**Updated:** 2026-05-07 (post-audit, Claude Sonnet 4.6)

---

## Current Phase: WAITING — All P1–P8 code complete, blocked on Dr. annotations

### ✅ What was completed (P1–P8)
- P1: Fixed 10-keypoint mismatch across dataset.py + model.py
- P2: `scripts/parse_annotations.py` → `data/processed/landmarks_clean.json`
- P3: `src/data/splits.py` — patient-aware 5-fold MSKF + 15% holdout
- P4: `src/phase2/augmentation.py` — ±10° Affine, ElasticTransform, GaussNoise, Perspective, CLAHE
- P5: `src/phase2/loss.py` — AdaptiveWingLoss (Wang et al. ICCV 2019)
- P6: Skipped — needs ≥1 annotated image
- P7: `src/phase2b/segmentation.py` + `segmentation_dataset.py` — U-Net scaffold
- P8: `src/phase2c/classifier.py` + `classifier_dataset.py` — EfficientNet-B3 multi-label

### ✅ Post-implementation audit (2026-05-07)
- 5 bugs found and fixed (see FAILURES.md)
- Pipeline dry-run passes: `python scripts/run_phase2_train.py --debug` exits cleanly
- All imports verified OK against Python 3.9 venv

### 📊 Current Data State
- **Images:** 104 (calibration complete) + 300+ incoming from Dr.
- **Annotations (current XML):** 2 images — 0 skeleton landmarks, 2 polygons
- **Calibration:** 104/104 complete (mm/pixel: 0.0974–0.0990)

### 🔄 Blockers
1. **Dr. skeleton annotations** — Need 20+ keypoint-annotated images to train Phase 2
2. **Phase 3 thresholds** — tipping_threshold_deg and translation_threshold_mm still null

---

## ⚡ IMMEDIATE NEXT ACTIONS

When Dr. exports new CVAT XML with skeleton annotations:
```bash
cd /Users/onis2/Project/Singdent/ceph-project
source venv/bin/activate
python scripts/parse_annotations.py              # regenerate landmarks_clean.json
python scripts/run_phase2_train.py --debug --max-images 10   # smoke-test
python scripts/run_phase2_train.py               # full LOPO (52 folds × 100 epochs)
```
No code changes needed — pipeline is ready.

---

## 📋 Workspace Files

| File | Purpose | Status |
|------|---------|--------|
| `agent_memory/00_Project_Context.md` | Master context | ✅ |
| `agent_memory/01_Current_Phase.md` | This file | ✅ Updated 2026-05-07 |
| `FAILURES.md` | All past mistakes — read first | ✅ Updated 2026-05-07 |
| `STATUS.md` | Phase snapshot | ✅ Updated 2026-05-07 |
| `PIPELINE_PLAN.md` | Full pipeline spec | ✅ |
| `CLAUDE.md` | Claude rules | ✅ |

**Python:** Always `source venv/bin/activate` first. venv at `venv/` (Python 3.9.6).
