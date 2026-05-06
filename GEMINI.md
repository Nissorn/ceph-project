# GEMINI.md — Cephalometric Landmark Detection

Read before starting any task:
1. `FAILURES.md` — mistakes already made, do not repeat
2. `STATUS.md` — current state (20 lines)

## Hardware — never violate
- Device: `torch.device("mps")` — NEVER `.cuda()`
- `num_workers=0` in all DataLoaders

## Rules — never violate
1. No horizontal flip augmentation
2. Patient-level splits — split on `patient_id`, T1+T2 always together
3. `mm_per_pixel` is per-image — look up from `calibration.csv` by `image_id`
4. No hardcoded paths — all from `config.yaml`
5. Data never committed to git

## Keypoints (10 total, hardcoded)
Upper_tip(0), Upper_apex(1), Labial_midroot(2), Labial_crest(3),
Palatal_midroot(4), Palatal_crest(5), ANS(6), PNS(7), LB(8), PB(9)

## Active code
- `src/data/` — parser, calibration, quality filter (USE THIS)
- `src/phase1/` — legacy, ignore
- `src/phase2/` — HRNet-W32 model, dataset, training loop

## Annotation XML location
`data/annotations.xml` (NOT data/raw/annotations/)
