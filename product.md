# Ceph V2 Auto — Product Requirements

## Project Context
End-to-end cephalometric analysis pipeline using deep learning on lateral cephalograms:
- **Phase 1**: Calibration + landmark detection
- **Phase 2A**: Heatmap-based landmark detection (MRE ~1.1mm)
- **Phase 2B**: Semantic segmentation (4-class: Background, Upper_Incisor, Labial, Palatal) — COMPLETE
- **Phase 2C**: High-res 1024x1024 segmentation — IN PROGRESS

---

## Phase 2C: High-Res 1024×1024 Segmentation

**STATUS: IN PROGRESS (Phase 2C.1 — Hyperparameter Optimization Retrain)**

### Objective
Push boundary precision for all segmentation classes — especially Upper_incisor — by moving to 1024×1024 input resolution with ResNet-50 encoders, evaluating two architectures in a sequential sweep.

### Phase 2C.1 Changes (vs Phase 2C original)
Phase 2C initial run (2026-05-28 ~11:06–13:45 UTC, ~2h39m) produced severe underfitting:
- DeepLabV3+ (resnet50) @ 1024px: **dice=0.5225** (vs 512px ceiling 0.8588)
- UNet++ (resnet50) @ 1024px: **dice=0.5397** (vs 512px ceiling 0.8588)

Root cause analysis:
1. LR 3e-4 too high for ResNet-50 @ 1024px — destabilizing with small effective bs=32
2. 100 epochs insufficient for larger encoder + higher res to converge
3. No warmup — large LR jump from init destabilizes early training
4. Effective bs=32 too small — only ~9 gradient updates per epoch with 294 train images

Phase 2C.1 hyperparameter fixes:
- **LR**: 1e-4 (reduced from 3e-4) with Linear Warmup 1e-5 → 1e-4 over 10 epochs
- **Scheduler**: Cosine Annealing (base=1e-4, min=1e-6) after warmup
- **Accumulation**: 4 → effective bs=64 (4 × 4 GPUs × 4 = 64) — 19 updates/epoch
- **Max epochs**: 200 (increased from 100)
- **Patience**: 30 (increased from 15)

---

### Data Ingestion
- **Parser**: `src/data/cvat_parser.py`
- **New function**: `parse_all_cvat_batches(annotations_dir, pattern="annotations_batch*.xml")` consumes all 4 batch XMLs and deduplicates by `image_id`, keeping the record with more keypoint annotations when duplicates occur.
- **Batch files**: `data/raw/annotations/annotations_batch{01,02,03,04}.xml`
- **Output merged JSON**: `data/processed/segmentation_train.json`
- **Stats (post-merge, 2026-05-28)**:
  - Total unique images: 381
  - Records with polygons: 362
  - Per-class: Upper_incisor=361, Labial_bone=362, Palatal_bone=362
  - Train/Val split: 294/68 (patient-level, seed=42, 20% holdout)

### Training Configuration (Phase 2C.1)
| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | (1024, 1024) |
| Encoders | resnet50 (both models) |
| Classes | 4 (Background=0, Upper_Incisor=1, Labial=2, Palatal=3) |
| Batch size per GPU | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 4 × 4 GPUs × 4 = **64** |
| AMP (autocast + GradScaler) | Enabled |
| Learning rate | 1e-4 (base) |
| LR warmup | 1e-5 → 1e-4 linear over 10 epochs |
| LR schedule | Cosine Annealing (min 1e-6) after warmup |
| Weight decay | 1e-3 |
| Dice weight | 0.5 |
| Max epochs | 200 per model |
| Early stopping patience | 30 |
| Augmentation | heavy (HFlip, Rotate±15°, BrightnessContrast, GaussNoise, Affine, GridDistortion, OpticalDistortion, CLAHE) |
| CLAHE | Enabled |
| Loss | CrossEntropyDiceLoss (CE + Dice, equal weight) |

### GPU Allocation
- **STRICT**: Only GPUs 0, 1, 2, 3 are used for training.
- External PIDs on other GPUs are never touched.
- GPU enforcement: `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"`

### Model Architectures (Sequential Sweep)
1. **Model A**: `smp.DeepLabV3Plus(encoder_name="resnet50", classes=4)`
2. **Model B**: `smp.UnetPlusPlus(encoder_name="resnet50", classes=4)`

### Output Directories (Phase 2C.1)
- `models/exp{TIMESTAMP}_DeepLabV3Plus_resnet50_1024_opt/best_model.pt` + `config.json`
- `models/exp{TIMESTAMP}_UnetPlusPlus_resnet50_1024_opt/best_model.pt` + `config.json`

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/run_1024_sweep.py` | Main sweep driver — trains both models sequentially (Phase 2C.1 optimized version) |
| `scripts/merge_cvat_data.py` | Batch merger before multi-XML parser was added |
| `scripts/evaluate_and_plot.py` | Post-training visual evaluation on Patient01_T1.jpg |

### Memory Safety (V100 32GB @ 1024px)
- bs=4/GPU with AMP fits in ~3 GB per GPU during forward pass
- 4-GPU bs=16 peak: ~6.4 GB — safe for V100 32GB
- BatchNorm note: `.train()` is called on the DataParallel model to avoid "Expected more than 1 value per channel" error in BatchNorm2d

### Known Issues / Fixes Applied
- **BatchNorm DataParallel crash**: model must be set to `.train()` mode before forward pass.
- **Double extension bug**: CVAT exports `.jpg.jpeg` — `_normalize_filename()` strips last 5 chars (`.jpeg`) to get correct disk filename.
- **Phase 2C underfitting**: LR=3e-4 too high, only 100 epochs, no warmup → fixed in 2C.1.

### 24-Hour Time Budget
- ~294 training images, effective bs=64 → ~5 steps/epoch
- DeepLabV3+ at 1024×1024: ~60-90s/epoch → 200 epochs ≈ 3.5h (worst case)
- U-Net++ at 1024×1024: ~80-120s/epoch → 200 epochs ≈ 4.5h (worst case)
- Total estimated: ~8 hours worst-case — well within 24h budget, ~16h headroom

---

## Phase 2B Results (Predecessor — for reference)
- **Best Dice**: 0.7868 (Ep147/150)
- **Model**: `smp.DeepLabV3Plus(resnet34)` | lr=3e-4 | wd=1e-3 | heavy aug | CLAHE
- **Phase 2B STATUS**: COMPLETE

---

## Phase 2A Landmark Detection Results (Reference)
- **5-fold CV MRE**: 1.097 ± 2.858 mm
- **Model**: HRNet-W32 + Adaptive Gaussian heatmaps
- **Phase 2A STATUS**: COMPLETE

---

## Shared Cluster Rules (Do Not Violate)
- **External PIDs to NEVER touch**: PIDs on GPUs 0, 3, 7 (external users)
- **Our GPUs**: 0, 1, 2, 3 (Phase 2C training)
- **Auto-isolate rule**: Any GPU at 0% utilization for >3 consecutive checks with >3GB VRAM = zombie — kill and respawn on our GPUs only.
- **Git exit codes**: Exit 1/128 on git push/pull — non-blocking, training continues independently.
- **Stale exit notifications**: Do not act on old process exit notifications from previous sessions.

## User Preferences
- Confirm before saves (all file creations/modifications require explicit user confirmation).
- No monitoring messages during active background training — wait for completion notification.
- Plain-text terminal output, no markdown.
| #29  | 0.6159  | 0.5211 | Unet          | watchdog update |
| #29  | 0.6159  | 0.5211 | Unet          | watchdog update |