# Cephalometric Landmark Detection System - Shared Context

**Last Updated:** 2026-05-05  
**Accessible to:** Copilot CLI, Claude Code, Gemini CLI, GitHub Copilot

---

## Project Overview

**Internship at:** Singapodent, Ho Chi Minh City, Vietnam  
**Clinical Goal:** Automate orthodontic tooth movement analysis from before/after lateral cephalogram X-rays

### What the system does:
1. Detects 8 anatomical landmarks on T1/T2 X-ray pairs
2. Superimposes T1 on T2 using ANS-PNS reference plane
3. Measures tooth movement vectors (mm)
4. Classifies treatment type (tipping, translation, torque, etc.)
5. Outputs visual tracing overlay + clinical report

---

## Dataset

| Property | Value |
|----------|-------|
| Current | 104 cephalometric images (52 patients, T1+T2 pairs) |
| Incoming | 300+ additional ceph images from Dr. |
| Format | JPG images + CVAT XML 1.1 annotations |
| Calibration | Ruler-based (0–30mm reference) per image |
| mm/pixel range | 0.0974–0.0990 (highly consistent) |

**Location:** `data/` (gitignored, not in repo)

---

## 10 Keypoints (Anatomical Landmarks)

| Index | Name | Source | Clinical Meaning |
|-------|------|--------|-----------------|
| 0 | Upper_tip | CVAT | Crown tip of upper incisor |
| 1 | Upper_apex | CVAT | Root apex |
| 2 | Labial_midroot | CVAT | Labial surface midroot |
| 3 | Labial_crest | CVAT | Alveolar bone crest (labial) |
| 4 | Palatal_midroot | CVAT | Palatal surface midroot |
| 5 | Palatal_crest | CVAT | Alveolar bone crest (palatal) |
| 6 | ANS | CVAT | Anterior Nasal Spine (superimposition reference) |
| 7 | PNS | CVAT | Posterior Nasal Spine (superimposition reference) |
| 8 | LB | CVAT v2+ | Labial bone level |
| 9 | PB | CVAT v2+ | Palatal bone level |

**Fixed in code:** `KEYPOINT_NAMES` array is hardcoded at 10 entries (indices 0–9). Never revert to 8.

---

## Project Structure

```
src/
├── data/          ← Active Phase 1 code (NOT src/phase1/)
│   ├── cvat_parser.py       (XML → structured data)
│   ├── calibration.py       (mm/pixel calculation)
│   └── quality_filter.py    (QC validation)
├── phase2/        ← HRNet-W32 landmark detection
│   ├── dataset.py
│   ├── model.py
│   ├── augmentation.py      (To be enhanced with Heavy Augmentation)
│   ├── heatmap.py
│   ├── train.py
│   └── metrics.py
├── phase3/        ← Treatment classification
│   ├── superimposition.py
│   └── heuristics.py
└── phase4/        ← Clinical output
    ├── convert.py
    └── visualize.py

scripts/
├── run_phase1_calibration.py
├── run_phase2_train.py
├── run_phase2_predict.py
├── run_phase3.py
└── run_pipeline.py

notebooks/        ← Exploratory analysis

config.yaml       ← All parameters (annotation_file: data/annotations.xml)
context.md        ← Detailed design decisions
CLAUDE.md         ← Instruction for Claude assistant
FAILURES.md       ← Mistakes already made (READ THIS FIRST)
STATUS.md         ← Current phase snapshot
```

---

## Critical Rules (NEVER VIOLATE)

1. **Hardware:** `torch.device("mps")` (Apple Silicon). Never `.cuda()`
2. **No horizontal flip** — breaks lateral cephalogram anatomy
3. **Patient-level splits** — T1+T2 of same patient always in same fold
4. **Per-image calibration** — Look up `mm_per_pixel` from `calibration.csv` by `image_id`; never use global constant
5. **Data never in git** — No `.jpg`, `.xml`, `.json`, `.csv`, `.pth` (data files)
6. **No hardcoded paths** — All from `config.yaml`
7. **Phase 3 thresholds nullable** — Return `"pending_threshold"` when null; never invent defaults

---

## Active Blockers

1. **Dr. Annotations** — ~2/104 images have 10-keypoint skeleton; need ~20+ before Phase 2 is viable
2. **Data Augmentation** — Heavy augmentation pipeline not yet implemented (critical to prevent overfitting)
3. **Phase 3 Thresholds** — `tipping_threshold_deg`, `translation_threshold_mm` still undefined

---

## Next Steps (Priority Order)

1. **Implement Heavy Augmentation** (2–3 days) → 104 images → 500+ variants
   - Use Albumentations with ±15° rotation, ±40% brightness/contrast, elastic deformation
   - Expected gain: 63.8% → 75–80% SDR@2.5mm

2. **Wait for Dr. Annotations** → Parse updated CVAT XML with 10 keypoints

3. **Phase 2 Training** → Once 20+ annotated images available

4. **Optional: Diffusion Synthesis** (Weeks 3–4) → Generate 200–400 synthetic cephalograms
   - DDPM training on best-augmented real images
   - Expected gain: 82–88% SDR@2.5mm

---

## Evaluation Metrics

- **MRE (mm)** — Mean Radial Error (target: < 2.0mm)
- **SDR@2mm** — Success Detection Rate at 2mm tolerance
- **SDR@2.5mm** — Success Detection Rate at 2.5mm tolerance
- **SDR@3mm, SDR@4mm** — Progressively lenient thresholds
- **LOPO-CV** — Leave-One-Patient-Out cross-validation (52 folds)

---

## Reference Papers

1. **CL-Detection2023 Challenge** — arxiv:2409.15834 (benchmark, best MRE 1.518mm)
2. **Rank-1 Cephalometric Method** — arxiv:2309.17143 (HRNet super-resolution heatmap head)
3. **Limited Data Augmentation** — arxiv:2505.06055 (Guo et al., HRNet overfitting analysis)
4. **Diffusion for X-rays** — arxiv:2407.18125 (Di Via et al., synthetic generation)

---

## Collaboration Guidelines

### For any AI tool using this project:

1. **Read these files first (in order):**
   - `agent_memory/00_Project_Context.md` (this file)
   - `agent_memory/01_Current_Phase.md` (immediate status + next action)
   - `FAILURES.md` (mistakes already made)
   - `STATUS.md` (current snapshot)

2. **Then read if needed:**
   - `CLAUDE.md` (detailed rules)
   - `context.md` (full project journal)

3. **Before making changes:**
   - Check `FAILURES.md` for common pitfalls
   - Verify all hardcoded values come from `config.yaml`
   - Ensure dataset paths are gitignored

4. **After completing work:**
   - Update `STATUS.md` with new phase status
   - Update this `agent_memory/01_Current_Phase.md` with immediate next action
   - Add any new learnings to `FAILURES.md` (lessons format)

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python3 scripts/run_phase1_calibration.py --cvat_xml data/annotations.xml --output_dir data/processed/` | Re-parse CVAT + calibrate |
| `python3 scripts/run_phase2_train.py --config config.yaml` | Train landmark detector |
| `python3 scripts/run_phase2_predict.py --config config.yaml --image <path>` | Predict landmarks on single image |
| `python3 scripts/run_phase3.py --config config.yaml --patient <id>` | Classify treatment type |
| `python3 scripts/run_pipeline.py --config config.yaml --patient <id>` | Full end-to-end pipeline |

---

**This is the master context document. All AI tools should consult it before starting work.**
