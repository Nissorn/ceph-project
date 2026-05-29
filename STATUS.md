# STATUS.md — Phase 2C/2D Cephalometric Landmark Detection
_Updated: 2026-05-29 08:23 — Cron job report_

---

## Phase 2C Backlog — ✅ ALL COMPLETE

| Rank | Exp | Architecture | Encoder | LR | Val Dice | Status |
|------|-----|-------------|---------|-----|---------|--------|
| 1 | Baseline | DeepLabV3+ | resnet34 (512px) | 1e-3 | **0.8588** | Completed |
| 2 | EXP-02 | UNetPlusPlus | resnet50 | 3e-4 | 0.7966 | Completed |
| 3 | EXP-03 | UNetPlusPlus | resnet50 | 3e-4 | 0.5416 | Completed |
| 4 | EXP-04 | DeepLabV3+ | efficientnet-b4 | 3e-4 | 0.5202 | Completed |
| 5 | EXP-01 | DeepLabV3+ | resnet50 (1024px) | 5e-4 | 0.5319 | Completed |
| 6 | EXP-05 | DeepLabV3+ | resnet50 | 1e-5 | 0.2717 | Completed |
| 7 | EXP-00 | DeepLabV3+ | resnet50 (1024px) | 1e-4 | 0.2600 | Aborted |

**Peak: 0.8588 (Baseline 512px DeepLabV3+)**

---

## Phase 2D: Post-1024 Pivot — In Progress

Task queue from `product.md`:
- [x] **TSK-01:** Sliding Window Inference (Pipeline B) — implemented in analysis_service.py (Pipeline B — Sliding Window Inference)
- [x] **TSK-02:** Generate Refiner Crops (Pipeline A data) — 240 boundary crops (80/class) extracted at 384×128px from 512px baseline
- [x] **TSK-03:** Train Stage 2 Lightweight Refiner — ✅ COMPLETED GPU 4
  - Architecture: DeepLabV3Plus + MobileNetV2, 4.38M params, no pretrained
  - Input: 256×256 (resized boundary crops)
  - Loss: 0.6×Dice + 0.4×Focal
  - Training: 189 train / 51 val crops, patient-level split
  - Result: best val_loss=0.1431 at epoch 40/60 (early stopping)
  - Output: t_ef460994/best_model.pt + config.json
- [x] **TSK-04:** Tversky + BoundaryDice Fine-Tuning — ✅ COMPLETED GPU 4
  - Architecture: DeepLabV3Plus + resnet34, fine-tuned from 512px baseline
  - Loss: 0.6×Tversky(α=0.7,β=0.3) + 0.4×BoundaryDice
  - Training: 291 train / 71 val records, patient-level split, 40 epochs (early stopping)
  - Result: **best_val_dice=0.8827** vs baseline 0.8588 (+0.024 improvement)
  - Output: models/tversky_deepLabV3plus_resnet34_20250529_20260529_094221/best_model.pt
  - Config: num_classes=4 (BG+3 fg), scripts/train_tversky.py committed to git

---

## Auto-Research Loop (iter1) — 🟡 RUNNING (background)

- **Process:** `auto_research_iter1.py --epochs-per-run 12` (PID 3350953)
- **Uptime:** ~14 hours (since 2026-05-28 ~17:30)
- **Unique experiments:** 374 | **Current best:** Dice=0.6159 (exp#34)
- **Latest:** exp#240 dice=0.1997 | Best this run: 0.6159
- **Watchdog:** PID 1527895, checking every 300s
- **Note:** This loop explores random architecture/loss/augmentation combos. Results independent of Phase 2C backlog.

---

## GPU Status (from watchdog)

| GPU | Util | Memory |
|-----|------|--------|
| 0 | 24% | 7554/32768 MiB (external) |
| 1 | 40% | 7050/32768 MiB (our process) |
| 2 | 11% | 7086/32768 MiB |
| 3 | 17% | 6958/32768 MiB |

**Protected GPUs (never use):** 0, 3, 7 — external users

---

## Key Rules
- GPUs 1, 2, 4, 5: available for our use
- No horizontal flip — breaks lateral cephalogram anatomy
- Patient-level splits always — split on `patient_id`
- Data never in git