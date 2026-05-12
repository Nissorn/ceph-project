# GitHub Copilot Instructions — Cephalometric Landmark Detection
<!-- Rules synced from CLAUDE.md — update CLAUDE.md first, then mirror here. Last synced: 2026-05-12 -->

Read before starting any task:
1. `FAILURES.md` — mistakes already made; do not repeat them
2. `STATUS.md` — current state snapshot
3. `context.md` — only if you need deeper background

---

## Hardware — NEVER violate

- Device: `torch.device("mps")` — NEVER `.cuda()` or `"cuda"`
- `num_workers=0` in all DataLoaders (MPS restriction on Mac M4)

## Rules — NEVER violate

1. **No horizontal flip** — breaks lateral cephalogram anatomy
2. **Patient-level splits** — T1+T2 of same patient always in same fold; split on `patient_id`
3. **Per-image calibration** — `mm_per_pixel` varies per image; always look up from `calibration.csv` by `image_id`; no global constant
4. **Data never in git** — no `.jpg`, `.xml`, `.json` (data), `.csv` (data), `.pth` in commits
5. **No hardcoded paths** — all paths from `config.yaml`
6. **Phase 3 thresholds are nullable** — return `"pending_threshold"` when null; do not invent defaults

## Keypoints — 10 total (hardcoded, never infer from file)

```python
KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS",
    "LB", "PB",   # indices 8–9; annotated in CVAT v2+
]
```

ANS(6)–PNS(7) = maxillary superimposition reference plane.

## Active code layout

```
src/data/    cvat_parser.py, calibration.py, quality_filter.py   ← USE THIS
src/phase1/  legacy scaffold — ignore
src/phase2/  dataset.py, model.py, heatmap.py, augmentation.py, train.py, metrics.py
src/phase3/  superimposition.py, heuristics.py
src/phase4/  convert.py, visualize.py
```

## Annotation XML location

`data/annotations.xml` (NOT `data/raw/annotations/`)

## Run commands

```bash
# Phase 1 — re-run whenever Dr. exports new CVAT XML
python3 scripts/run_phase1_calibration.py --cvat_xml data/annotations.xml --output_dir data/processed/
```

## Key papers

- CL-Detection2023: arxiv:2409.15834 — benchmark (best MRE 1.518 mm)
- Rank-1 method: arxiv:2309.17143 — super-resolution heatmap head
