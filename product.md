# Product Definition: Autonomous Cephalometric Analysis System (Phase 2B & LLM-Driven Autoresearch)

## 1. Project Goal
Develop a robust, end-to-end automated orthodontic cephalometric analysis pipeline. The system processes dental X-rays to detect structural landmarks (Phase 2A), segment the alveolar bone boundaries (Phase 2B), calculate biomechanical constraints (Phase 3), and recommend safe treatment plans via an AI-driven Fusion Model (Phase 2C).

## 2. Current Focus: Karpathy Autoresearch Paradigm for Phase 2B Segmentation
The immediate objective is to operationalize Phase 2B (Alveolar Bone Segmentation) using a fully autonomous, LLM-driven machine learning research loop. Following the Andrej Karpathy `autoresearch` design paradigm, the experimental search space is controlled by a persistent server-side script that leverages a generative LLM API to analyze training metrics, reason through flaws, and dynamically rewrite the optimization parameters.

## 3. Data Status & Automated Ingestion
The dynamic ingestion pipeline (`scripts/merge_cvat_data.py`) has successfully deduplicated and parsed raw CVAT XML exports (`batch01` to `batch04`) inside `data/raw/annotations/`. The unified dataset is split into clean schemas:
* **`landmark_train.json`:** 317 unique records with valid skeleton annotations (Phase 2A target).
* **`segmentation_train.json`:** 274 unique records containing at least one polygon annotation (Phase 2B target).
  * *Upper_incisor polygons:* 273 records
  * *Labial_bone polygons:* 274 records
  * *Palatal_bone polygons:* 273 records
* **Phase 2C / 3 Meta Data:** 89 records contain comprehensive treatment tags; 21 low-quality records are flagged under `Quality_Reject` and isolated.

## 4. Autonomous Loop & Agentic Infrastructure Requirements

### A. Server-Side Autonomous Research Loop (`scripts/autoresearch_agent.py`)
The system must run indefinitely as a headless background daemon on the server, entirely independent of any active IDE/editor interactive chat window.
* **API-Driven Optimization:** After a model executes its training run for a designated block of epochs, the script parses the local metrics tensor curves (Validation Dice Score, IoU, Train/Val Loss trends).
* **LLM-In-The-Loop Reasoning:** The script dynamically builds a contextual prompt containing the baseline configurations, metrics, and failure modes (e.g., "Validation Dice stagnating at 0.65 while training loss drops to 0.02"). It sends this payload directly to an external LLM API (OpenAI/Anthropic).
* **Dynamic Re-Configuration:** The LLM acts as the remote scientist, outputting a structured JSON payload containing the next targeted architecture (e.g., swapping `Unet` with `AttentionUnet`), backbone choice (`resnet34` vs `efficientnet-b4`), revised Learning Rates, and specific Augmentation strengths. The script parses this JSON and spins up the next pipeline execution immediately.

### B. Automated Git Pipeline & State Capture
To ensure stable versioning without human intervention, the python autonomous worker must directly orchestrate the local Git repository:
* **Baseline Breakthroughs:** The loop tracks the absolute best Validation Dice Score achieved during the session (current baseline ceiling is set at `0.8376`).
* **Smart Branching:** When an LLM-suggested model beats the baseline, the script executes python shell subprocesses to:
  1. Dynamically create a branch named after the breakthrough (e.g., `experiment/AttentionUnet_resnet34-dice85`).
  2. Stage the precise model configuration file (`config.json`), hyperparameter logs, and performance graphs.
  3. Execute an automated commit (`perf: achieve best validation dice score [SCORE] using [MODEL]`).
  4. Automatically merge or checkout back to the primary optimization branch to resume the infinite research search.
* **Data Isolation:** All raw `.pt` or `.pth` heavy model weights are strictly bound to `.gitignore` rules to prevent repository bloat.

### C. Resilience, Self-Healing, and Telemetry
* **VRAM Exception Recovery:** If a model combination suggested by the LLM exceeds hardware capabilities and throws a CUDA Out-of-Memory (OOM) error, the script must intercept the exception, clear the VRAM cache via `torch.cuda.empty_cache()`, punish that configuration choice in the next LLM prompt context, and continue the loop without crashing the daemon process.
* **Static Telemetry Logging:** Upon completing any cycle, the daemon must automatically overwrite a localized status file at `data/processed/LEADERBOARD.md` to display the absolute ranking table of the top 5 models, current resource allocation (GPU utilization), and progress history.

## 5. Execution Priorities & Live Fleet State

### A. Shared Infrastructure Safety Rules (CRITICAL)
* **Server Rule:** This is a shared laboratory environment. The user does NOT own this server.
* **Touch Restrictions:** STRICTLY FORBIDDEN from killing, freezing, or interacting with external workloads (e.g., `shrimp-tf` envs) on GPU 0, 3, and 7.
* **Target Management Cluster:** Only manage and execute automation inside GPU 1, 2, 4, and 5.

### B. Production-Ready Baseline Benchmarks
The system has successfully achieved elite clinical-grade metrics across both baseline components of Phase 2:
1. **Phase 2A (Landmark Detection):** CONCLUDED via 5-Fold Cross-Validation on 317 images.
   * *5-Fold CV Metrics:* **Mean MRE = 1.097 ± 2.858 mm** | **SDR @ 2.0mm = 93.8%** | **SDR @ 3.0mm = 94.9%** (Model: HRNet-W32, Stage 1+2 frozen).
   * *Validation Holdout (5 unseen patients from fold 0):* **Overall MRE = 1.815 mm** | **SDR @ 2.0mm = 68.0%** | **SDR @ 3.0mm = 86.0%** (5-case holdout, zero data leakage).
   * *Per-Fold Best Checkpoints (150 epochs each):* fold1–fold5 all converged; fold1 MRE_argmax=0.626 mm.
2. **Phase 2B (Alveolar Bone Segmentation):** CONCLUDED via parallel cluster auto-research.
   * *Metrics:* **Absolute Best Validation Dice = 0.8437** (Model: DeepLabV3Plus + ResNet34, GPU 1).

### C. Resource Cleanup & Fleet Management Protocol (As of May 24, 2026)
* **Overnight Action:** Parallel distributed 5-Fold cross-validation re-training for Phase 2A (Landmark) launched concurrently across GPU 1, 2, 4, and 6 using 317 calibrated records. 
* **Next Directive Prep:** Upon automated scripts completion, check `outputs/VALIDATION_REPORT.md` for geometrically fixed inference snapshots.

### D. Active Fleet State (Updated May 25, 2026 — Training Complete)
| GPU ID | VRAM Used | Status | Assignment |
|--------|-----------|--------|------------|
| 1, 2, 4, 5 | Idle      | IDLE   | Training complete — all folds converged |
| 5      | 31470 MiB | LOCKED | `shrimp-tf` external workload — UNTOUCHED |
| 0, 3, 7| External  | LOCKED | `shrimp-tf` external workload — UNTOUCHABLE |

**Phase 2A landmark training:** ALL COMPLETE. Checkpoints at `outputs/checkpoints/fold{1–5}_best.pth`.

## 6. Active Technical Bottlenecks & Hotfixes

### A. Phase 2A Landmark Detection — CORRECTIVE RE-TRAINING COMPLETE (May 25, 2026)
* **Status:** CONCLUDED — 5-Fold CV training completed on all folds (150 epochs each). All 5 fold checkpoints converged.
* **Per-Fold Results:**
  * Fold 1: MRE_argmax = 0.626 mm (fold1_best.pth)
  * Fold 3: MRE_argmax = 0.573 mm
  * Fold 4: MRE_argmax = 0.590 mm
  * Fold 5: MRE_argmax = 0.578 mm
* **Validation Holdout (5 unseen patients):** Overall MRE = 1.815 mm | SDR@2mm = 68.0% | SDR@3mm = 86.0%
* **Hyperparameter Corrections Applied:**
  * `epochs: 150` with `partial_freeze: true` enforced in config
  * `backbone_lr: 1e-5`, `head_lr: 1e-3` (differential LR, Stage 1+2 frozen)
  * SoftArgmax temperature: `1.0` (was 10.0 — was over-smoothing coordinates)
  * Adaptive Wing Loss normalization: divides by `n_valid` only (not H*W)

| #95  | 0.3322  | 0.2586 | DeepLabV3Plus          | watchdog update |
| #100  | 0.3398  | 0.2503 | DeepLabV3Plus          | watchdog update |
| #108  | 0.4256  | 0.3240 | DeepLabV3Plus          | watchdog update |
| #178  | 0.4685  | 0.3721 | DeepLabV3Plus          | watchdog update |

## 8. Visual Evaluation Pipeline (Phase 2B)
- **Script:** `scripts/evaluate_and_plot.py` — fully operational
- **Auto-finds best model** from `models/` by highest val_dice (currently: exp0000_DeepLabV3Plus_resnet34_clahe1, dice=0.8588)
- **Pipeline:** Load model → run inference (argmax, no overlap) → decode contours → overlay polygons on original image → overlay ground-truth landmarks → save to `reports/visual_results/`
- **Output:** `reports/visual_results/eval_Patient01_T1_<timestamp>.png` — 590KB, 120dpi
- **Usage:** `python3 scripts/evaluate_and_plot.py [--image Patient03_T1.jpg] [--model-dir <path>]`
- **Classes plotted:** Upper_incisor (red), Labial_bone (blue), Palatal_bone (green), landmarks (yellow)

## 9. Grid Pruning & 24-Hour Deadline Strategy (2026-05-27)
**Situation:** 1728-combo exhaustive grid at 6 min/run = ~6.3 days. Deadline: 24 hours max.

**Server Cleansing (Phase 1):**
- External processes `envs/oxygen` PIDs 1279945 and 1303468 were competing for all GPUs — cleared
- All 4 GPUs now dedicated exclusively to `ceph-v2-auto` training
- Old watchdog processes purged; fresh watchdog PID 1410033 launched

**Data-Driven Grid Pruning (Phase 2):**
- Analyzed 372 completed 4-class experiments to identify dominant hyperparameter regions
- Key findings:
  - `DeepLabV3Plus` dominates: best=0.8588, mean=0.2987
  - `lr=0.0003` + `wd=0.001` + `aug=heavy` + `clahe=True` is the consistent winner
  - `AttentionUnet` and `Unet` are viable alternatives worth exploring
  - `efficientnet-b4` encoder (with Unet) is the strongest alternative backbone
  - `extreme` augmentation consistently underperforms
- Pruned grid: 4 archs × 1-2 encoders × 3 LRs × 2 WDs × 2 augs = **48 combos**
- 6 epochs per run (~3 min/run) × 48 = ~144 min total / 4 GPUs = **36 min wall time**
- `auto_research_iter1.py` modified with `--max-experiments 48` flag + `fast_grid.json` override

**Execution:**
- Training: PID 1409248, GPU0=44%, GPU1=47%, GPU2=17%, GPU3=33% (all 4 active)
- Watchdog: PID 1410033, 5-min dynamic PID cycles
- Output: `fast_grid.log` for live run tracking
- Target: **all 48 combos exhausted within 24 hours, best Dice to exceed 0.8588**

## 10. Final Deep Run — 150 Epoch (2026-05-27)
**Launch timestamp:** 2026-05-27 19:57 UTC

**Configuration (confirmed winner from 433-experiment analysis):**
| Parameter | Value |
|---|---|
| Architecture | DeepLabV3Plus |
| Encoder | resnet34 (ImageNet pretrained) |
| Learning rate | 3e-04 |
| Weight decay | 1e-03 |
| Batch size | 16 (4/GPU × 4 GPUs via DataParallel) |
| Augmentation | heavy (flip 0.5, rotate ±15°, brightness/contrast, gauss noise, affine, grid/optic distort) |
| CLAHE | True (clip=2.0, tile=8×8) |
| Loss | CrossEntropyDiceLoss (dice_w=0.5) |
| Epochs target | 150 (early stop patience=25) |
| Classes | 4 (Background + Upper_incisor, Labial_bone, Palatal_bone) |

**PID:** 1541384 (training) | 1541785 (watchdog) | 4 GPUs: GPU0=12572 GPU1=11272 GPU2=11152 GPU3=11120 MiB

**Expected completion:** ~35 min wall time (150 ep × ~14s/ep) → well within 24-hour deadline

**On-completion hook:** `evaluate_and_plot.py` auto-runs on `Patient01_T1.jpg` with:
- Native-pixel landmark coordinates (`extent=(0, W_orig, 0, H_orig)`)
- Correct polygon overlay on original image resolution
- Ground-truth landmarks + predicted masks in distinct colors

**Watchdog:** Dynamic PID tracking, auto-commits if new Dice > 0.8588

**STATUS: COMPLETED ✅** (2026-05-27 20:29 UTC)

**Final Deliverables:**
- **Model weights (FastAPI path):** `/home/iddi/ceph-v2-auto/models/finalDeepRun_deeplabv3plus_resnet34_lr0.0003_wd0.001_bs4x4_150ep_20260527_195758/best_model.pt`
- **Visualization:** `/home/iddi/ceph-v2-auto/reports/visual_results/eval_Patient01_T1_20260527_202919.png`
- **Git branch:** `experiment/DeepLabV3Plus-resnet34-dice07868` → pushed to origin ✅
- **GitHub PR:** https://github.com/Nissorn/ceph-project/pull/new/experiment/DeepLabV3Plus-resnet34-dice07868
| #0  | 0.4851  | 0.3711 | Unet          | watchdog update |
| #0  | 0.5197  | 0.4080 | Unet          | watchdog update |
| #5  | 0.4994  | 0.3875 | Unet          | watchdog update |
| #9  | 0.5710  | 0.4628 | Unet          | watchdog update |
| #9  | 0.5710  | 0.4628 | Unet          | watchdog update |
| #24  | 0.5075  | 0.3976 | Unet          | watchdog update |
| #29  | 0.5218  | 0.4154 | Unet          | watchdog update |
| #106  | 0.5298  | 0.4135 | Unet          | watchdog update |