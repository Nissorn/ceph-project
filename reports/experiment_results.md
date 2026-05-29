# Experiment Results — Cephalometric Landmark Detection

*Generated: 2026-05-29 | Total completed runs: 1764*

## Top 10 Overall
| Rank | Dice | Architecture | Encoder | GPU | CLAHE | LR | Loss |
|------|------|-------------|---------|-----|-------|----|------|
| 1 | 0.8827 | DeepLabV3+ | resnet34 | N/A | N/A | 0.0003 | Tversky(alpha=0.7,beta=0.3) +  |
| 2 | 0.8588 | DeepLabV3+ | resnet34 | N/A | Yes | 0.0003 | — |
| 3 | 0.8450 | UNet | efficientnet-b4 | N/A | N/A | 0.0003 | — |
| 4 | 0.8437 | DeepLabV3+ | resnet34 | N/A | N/A | 0.001 | — |
| 5 | 0.8430 | UNet | efficientnet-b4 | N/A | N/A | 0.001 | — |
| 6 | 0.8429 | UNet | efficientnet-b4 | N/A | N/A | 0.001 | — |
| 7 | 0.8418 | UNet | efficientnet-b4 | N/A | N/A | 0.001 | — |
| 8 | 0.8417 | UNet | efficientnet-b4 | N/A | N/A | 0.001 | — |
| 9 | 0.8409 | AttentionUnet | resnet34 | N/A | N/A | 0.001 | — |
| 10 | 0.8406 | DeepLabV3+ | resnet34 | N/A | N/A | 0.001 | — |

## Best by Architecture
| Architecture | Runs | Best Dice | Encoder | GPU | CLAHE |
|--------------|------|-----------|---------|-----|-------|
| DeepLabV3+ | 472 | 0.8827 | resnet34 | N/A | N/A |
| UNet | 996 | 0.8450 | efficientnet-b4 | N/A | N/A |
| AttentionUnet | 95 | 0.8409 | resnet34 | N/A | N/A |
| Linknet | 149 | 0.8334 | resnet34 | N/A | N/A |
| UNetPlusPlus | 3 | 0.7966 | resnet50 | N/A | N/A |
| Other | 1 | 0.5416 | resnet50 | N/A | N/A |

## Phase 2C Backlog (Official Experiments)
| Rank | Exp | Architecture | Encoder | LR | Val Dice | Status |
|------|-----|-------------|---------|-----|---------|--------|
| Baseline | DeepLabV3+ | resnet34 | 1e-3 | 0.8588 | ✅ Completed |
| EXP-02 | UNetPlusPlus | resnet50 | 3e-4 | 0.7966 | ✅ Completed |
| EXP-03 | UNetPlusPlus | resnet50 | 3e-4 | 0.5416 | ✅ Completed |
| EXP-04 | DeepLabV3+ | efficientnet-b4 | 3e-4 | 0.5202 | ✅ Completed |
| EXP-01 | DeepLabV3+ | resnet50 | 5e-4 | 0.5319 | ✅ Completed |
| EXP-05 | DeepLabV3+ | resnet50 | 1e-5 | 0.2717 | ✅ Completed |
| EXP-00 | DeepLabV3+ | resnet50 | 1e-4 | 0.2600 | ❌ Aborted |
| TSK-04 | DeepLabV3+ (Tversky) | resnet34 | 3e-4 | 0.8827 | ✅ NEW BEST |

## DeepLabV3+ Encoder Breakdown
| Encoder | Runs | Best Dice |
|---------|------|-----------|
| resnet34 | 467 | 0.8827 |
| resnet50 | 4 | 0.5587 |
| efficientnet-b4 | 1 | 0.5202 |

## TSK-04 Result (Tversky Fine-Tune)
- **Location:** `models/tversky_deepLabV3plus_resnet34_20250529_20260529_094221/`
- **Dice: 0.8827** ← NEW BEST, beats previous champion (0.8588) by +0.0238
- **Architecture:** DeepLabV3+ + resnet34, fine-tuned from 512px baseline
- **Loss:** Tversky(alpha=0.7, beta=0.3) + BoundaryDice (0.6/0.4 weighting)
- **Epochs:** 50 (early stopped with patience)
- **Model size:** 22.5M params