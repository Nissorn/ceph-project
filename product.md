# 🚀 Phase 2C: High-Res 1024x1024 Autonomous Segmentation Sweep

## 🎯 Objective
Find the optimal hyperparameter and architecture combination for 1024x1024 Cephalometric Bone/Tooth segmentation using only ~290 training images.
**Baseline to beat:** DeepLabV3+ (ResNet34, 512px) -> Dice: 0.8588

## 🏆 Leaderboard (Ledger)
| Rank | Exp ID | Architecture | Encoder | LR | Batch (Eff) | Loss Func | Max Dice | Status |
| 4 | EXP-02 | UNetPlusPlus | resnet50 | 3e-4 | 32 | Dice+CE | 0.7966 | Completed |
| 5 | EXP-02 | UNetPlusPlus | resnet50 | 3e-4 | 32 | Dice+CE | 0.7966 | Completed |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | DeepLabV3+ | resnet34 (512px) | 1e-3 | 32 | Dice+CE | 0.8588 | Completed |
| 2 | EXP-00 | DeepLabV3+ | resnet50 (1024px)| 1e-4 | 64 | Dice+CE | 0.2600 | Aborted (Underfit) |
| 3 | EXP-01 | DeepLabV3+ | resnet50 (1024px) | 5e-4 | 32 | Dice+CE | 0.5319 | Completed |

---

## 🧪 Experiment Backlog (Hypotheses List)
*Agent Instruction: Pick the first `[ ]` pending experiment, execute it, update the status to `[x]` (Success) or `[-]` (Failed/OOM), and log the result in the Leaderboard.*

- [x] **EXP-01: The "Aggressive Optimizer"** — Max Dice: 0.5319 (Completed)
  - **Hypothesis:** 1024px with ResNet50 needs a stronger learning rate to overcome the large effective batch size.
  - **Config:** DeepLabV3+, resnet50, LR `5e-4`, Accumulation `2` (Eff. Batch 32), Loss `Dice + CrossEntropy`, Epochs `200`, Linear Warmup 10 epochs.

- [ ] **EXP-02: Architecture Shift (U-Net++ Focus)**
  - **Hypothesis:** U-Net++ nested decoders might capture the thin bone boundaries better than DeepLab at high resolution.
  - **Config:** U-Net++, resnet50, LR `3e-4`, Accumulation `2` (Eff. Batch 32), Loss `Dice + CrossEntropy`, Epochs `150`, Linear Warmup 10 epochs.

- [ ] **EXP-03: The "Focal Loss" Boundary Booster**
  - **Hypothesis:** CrossEntropy dominates the loss with background pixels. Focal Loss will force the model to focus on the hard-to-predict thin bone plates.
  - **Config:** Best architecture from EXP-01/02, LR `3e-4`, Loss `DiceLoss + FocalLoss`, Eff. Batch 32.

- [ ] **EXP-04: Lightweight Encoder (Combat Overfitting)**
  - **Hypothesis:** ResNet50 is too large for 290 images and causes capacity shock. Reverting to a smaller encoder at 1024px might generalize better.
  - **Config:** DeepLabV3+, `efficientnet-b4` (or `resnet34`), LR `3e-4`, Eff. Batch 32, Loss `Dice + CE`.

- [ ] **EXP-05: Transfer Learning (Freeze Backbone)**
  - **Hypothesis:** Gradients are destroying the ImageNet pre-trained weights early on. We should freeze the encoder for the first 10 epochs.
  - **Config:** Best model so far. `freeze_encoder=True` for epochs 1-10, then unfreeze with a lower LR (`1e-5`).

| #117  | 0.5764  | 0.4713 | Unet          | watchdog update |