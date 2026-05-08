# Cephalometric Landmark Detection System

AI pipeline for orthodontic tooth movement analysis from lateral cephalogram X-rays.
Internship project at Singapodent, Ho Chi Minh City, Vietnam.

## What it does

1. Detects 10 anatomical landmarks on before/after (T1/T2) X-ray image pairs
2. Superimposes T1 on T2 using the ANS-PNS reference plane
3. Measures tooth movement vectors in millimeters
4. Classifies the treatment type (tipping, translation, torque, etc.)
5. Outputs visual tracing overlay + structured report for the orthodontist

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd ceph-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (Apple Silicon / MPS backend)
pip install -r requirements.txt

# Place your data (not included — proprietary)
# data/raw/images/     ← patient JPG images
# data/raw/annotations/annotations.xml  ← CVAT XML 1.1 export
```

## Quick Start

```bash
# Phase 1 — Parse CVAT annotations → structured JSON + calibration CSV
python scripts/run_phase1.py --config config.yaml

# Phase 2 — Train landmark detector (requires annotated images)
python scripts/run_phase2_train.py --config config.yaml

# Phase 2 — Predict landmarks on a single image
python scripts/run_phase2_predict.py --config config.yaml --image data/raw/images/Patient01_T1.jpg

# Phase 3 — Classify treatment type for a patient pair
python scripts/run_phase3.py --config config.yaml --patient Patient01

# Full pipeline — all phases end-to-end
python scripts/run_pipeline.py --config config.yaml --patient Patient01
```

## Dataset

**Not included.** The dataset is proprietary clinic data (52 patients, 104 JPG lateral cephalograms + CVAT XML annotations). All files under `data/` are gitignored.

## Hardware Requirements

- Apple Silicon Mac (M-series) — uses MPS backend
- 8 GB RAM minimum, 16 GB recommended
- PyTorch ≥ 2.2.0 with MPS support

## Project Structure

```
src/phase1/     Data parsing and calibration
src/phase2/     HRNet-W32 landmark detection model
src/phase3/     Heuristic treatment classification
src/phase4/     Clinical output (mm conversion + visualization)
scripts/        CLI entry points for each phase
notebooks/      Exploratory analysis and result visualization
config.yaml     All configurable parameters
context.md      Full project journal and design decisions
CLAUDE.md       Instructions for Claude Code assistant
```

## Evaluation

Leave-One-Patient-Out Cross Validation (LOPO-CV), 52 folds.
Metrics: MRE (mm), SDR@2mm, SDR@2.5mm, SDR@3mm, SDR@4mm.
Target: MRE < 2.0 mm for all 10 landmarks.

## References

- CL-Detection2023 Challenge: arxiv:2409.15834
- HRNet rank-1 cephalometric method: arxiv:2309.17143
