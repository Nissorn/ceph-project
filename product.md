# Ceph V2 Auto — Product Requirements

## Project Context
End-to-end cephalometric analysis pipeline using deep learning on lateral cephalograms:
- **Phase 1**: Calibration + landmark detection
- **Phase 2A**: Heatmap-based landmark detection (MRE ~1.1mm)
- **Phase 2B**: Semantic segmentation (4-class: Background, Upper_Incisor, Labial, Palatal)
- **Phase 2C**: High-res 1024x1024 segmentation sweep + multi-XML ingestion ← **CURRENT**

---

## Phase 2C: High-Res 1024×1024 Segmentation Sweep

**STATUS: IN PROGRESS**

### Objective
Push boundary precision for all segmentation classes — especially Upper_incisor — by moving to 1024×1024 input resolution with ResNet-50 encoders, evaluating two architectures in a sequential sweep.

### Data Ingestion
- **Parser**: `src/data/cvat_parser.py`
- **New function**: `parse_all_cvat_batches(annotations_dir, pattern="annotations_batch*.xml")` consumes all 4 batch XMLs and deduplicates by `image_id`, keeping the record with more keypoint annotations when duplicates occur.
- **Batch files**: `data/raw/annotations/annotations_batch{01,02,03,04}.xml`
- **Output merged JSON**: `data/processed/segmentation_train.json` (replaces legacy single-file approach)
- **Stats (post-merge, 2026-05-28)**:
  - Total unique images: 381
  - Records with polygons: 362
  - Per-class: Upper_incisor=361, Labial_bone=362, Palatal_bone=362
  - Train/Val split: 289/73 (patient-level, seed=42, 20% holdout)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | (1024, 1024) |
| Encoders | resnet50 (both models) |
| Classes | 4 (Background=0, Upper_Incisor=1, Labial=2, Palatal=3) |
| Batch size per GPU | 4 |
| Effective batch size | 4 × N_GPUs × 2 (grad accum) = 32 (4 GPUs) |
| Gradient accumulation steps | 2 |
| AMP (autocast + GradScaler) | Enabled |
| Learning rate | 3e-4 |
| Weight decay | 1e-3 |
| Dice weight | 0.5 |
| Scheduler | CosineAnnealingLR |
| Max epochs | 100 per model |
| Early stopping patience | 15 |
| Augmentation | heavy (HFlip, Rotate±15°, BrightnessContrast, GaussNoise, Affine, GridDistortion, OpticalDistortion) |
| CLAHE | Enabled (apply_clahe=True) |
| Loss | CrossEntropyDiceLoss (CE + Dice, equal weight) |

### GPU Allocation
- **STRICT**: Only GPUs 0, 1, 2, 3 are used for training.
- External PIDs on GPU 5 (kanjana's process) are never touched.
- GPU enforcement: `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"`
- GPU state checked via `nvidia-smi` before any action.

### Model Architectures (Sequential Sweep)
1. **Model A**: `smp.DeepLabV3Plus(encoder_name="resnet50", classes=4)`
2. **Model B**: `smp.UnetPlusPlus(encoder_name="resnet50", classes=4)`

### Output Directories
- `models/exp{TIMESTAMP}_DeepLabV3Plus_resnet50_1024/best_model.pt` + `config.json`
- `models/exp{TIMESTAMP}_UnetPlusPlus_resnet50_1024/best_model.pt` + `config.json`

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/run_1024_sweep.py` | Main sweep driver — trains both models sequentially |
| `scripts/merge_cvat_data.py` | (Already existed) Batch merger before multi-XML parser was added |
| `scripts/evaluate_and_plot.py` | Post-training visual evaluation on Patient01_T1.jpg |

### Memory Safety (V100 32GB @ 1024px)
- bs=4/GPU with AMP fits in ~3 GB per GPU during forward pass
- 4-GPU bs=16 peak: ~6.4 GB — safe for V100 32GB
- BatchNorm note: `.train()` is called on the DataParallel model to avoid "Expected more than 1 value per channel" error in BatchNorm2d

### Known Issues / Fixes Applied
- **BatchNorm DataParallel crash**: DeepLabV3+ ASPP module uses BatchNorm with `track_running_stats`. When DataParallel scatters batches across GPUs, individual replicas may receive batch_size=1. Fix: model must be set to `.train()` mode before forward pass.
- **Duplicate patient_id in batch02**: `Patient100_T1.jpg.jpeg` etc. have double extension — logged as warning, doesn't affect segmentation records.

### 24-Hour Time Budget Estimate
- ~289 training images, effective bs=32 → ~9 steps/epoch
- DeepLabV3+ at 1024×1024: ~50-80s/epoch → 80 epochs ≈ 70 min
- U-Net++ at 1024×1024: ~60-90s/epoch → 80 epochs ≈ 80 min
- Total estimated: ~3 hours — well within 24h budget

---

## Phase 2B Results (Predecessor — for reference)
- **Best Dice**: 0.7868 (Ep147/150)
- **Model**: `smp.DeepLabV3Plus(resnet34)` | lr=3e-4 | wd=1e-3 | heavy aug | CLAHE
- **Git branch**: `experiment/DeepLabV3Plus-resnet34-dice07868`
- **Classes**: 4 (Background, Upper_Incisor, Labial, Palatal)
- **Phase 2B STATUS**: COMPLETE

---

## Phase 2A Landmark Detection Results (Reference)
- **5-fold CV MRE**: 1.097 ± 2.858 mm
- **Model**: HRNet-W32 + Adaptive Gaussian heatmaps
- **Phase 2A STATUS**: COMPLETE

---

## Shared Cluster Rules (Do Not Violate)
- **External PIDs to NEVER touch**: 1931218 (GPU 5/6/7 — shrimp-tf user kanjana)
- **Our GPUs**: 1, 2, 4, 5 (Phase 2B baseline ceiling: 0.8376)
- **Auto-isolate rule**: Any GPU at 0% utilization for >3 consecutive checks with >3GB VRAM = zombie — kill and respawn on our GPUs only.
- **Git exit codes**: Exit 1/128 on git push/pull — non-blocking, training continues independently.
- **Stale exit notifications**: Do not act on old process exit notifications from previous sessions.

## User Preferences
- Confirm before saves (all file creations/modifications require explicit user confirmation).
- No monitoring messages during active background training — wait for completion notification.
- Plain-text terminal output, no markdown.
| #0  | 0.5460  | 0.4351 | Unet          | watchdog update |
| #3  | 0.5647  | 0.4634 | Unet          | watchdog update |