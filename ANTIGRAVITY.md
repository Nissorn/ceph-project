# Cephalometric Landmark Detection — Antigravity Context

Read before starting any task:
1. `FAILURES.md` — mistakes already made, do not repeat
2. `STATUS.md` — current state snapshot
3. `PIPELINE_PLAN.md` — full phase plan with input/output specs

## Hardware — never violate
- Device: `torch.device("mps")` — NEVER `.cuda()`
- `num_workers=0` in all DataLoaders

## Rules — never violate
1. No horizontal flip augmentation (breaks lateral cephalogram anatomy)
2. Patient-level splits — split on `patient_id`, T1+T2 always in same fold
3. `mm_per_pixel` is per-image from `calibration.csv`, never a global constant
4. No hardcoded paths — all from `config.yaml`
5. Data never committed to git (proprietary clinic data)

## Keypoints (10 total, hardcoded order)
Upper_tip(0), Upper_apex(1), Labial_midroot(2), Labial_crest(3),
Palatal_midroot(4), Palatal_crest(5), ANS(6), PNS(7), LB(8), PB(9)

## Active code directories
- `src/data/` — parser, calibration, quality filter (USE THIS, not src/phase1/)
- `src/phase2/` — HRNet-W32 model, dataset, heatmap, training loop
- `src/phase3/` — superimposition + heuristics
- `src/phase4/` — visualize + convert

## Annotation XML location
`data/annotations.xml` (NOT data/raw/annotations/)

## After every coding session — update STATUS.md
When you finish a task, update STATUS.md:
- Move completed items to "What is done"
- Update blockers
- Note anything that broke in FAILURES.md
