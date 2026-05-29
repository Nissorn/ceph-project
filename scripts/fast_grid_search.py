#!/usr/bin/env python3
"""
Fast Focused Grid Search -- 48 Runs
Based on analysis of 372 completed 4-class experiments:
- DeepLabV3Plus dominates (best=0.8588, mean=0.2987)
- lr=0.0003 + wd=0.001 + aug=heavy + clahe=True is the winning recipe
- 4 arch x 1 enc x 3 LRs x 2 WDs x 2 aug = 48 runs
- 6 epochs per run (~3 min) x 48 = ~144 min / 4 GPUs = 36 min wall time
Target: exhaust within 24h
"""
import argparse, json, logging, os, random, subprocess, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                   datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("fast_grid")

# Architecture + encoder mapping (data-driven)
ARCH_ENCODER_MAP = {
    "DeepLabV3Plus":  "resnet34",
    "AttentionUnet": "resnet34",
    "Unet":           "efficientnet-b4",
    "Linknet":        "resnet34",
}

# Hyperparameter grid (pruned from 1728 to 48)
LRS        = [0.0003, 0.001, 0.0001]
WDS        = [0.001, 0.0005]
AUGS       = ["heavy", "medium"]
BATCH_SIZE = 4
NUM_CLASSES = 4
USE_CLAHE  = True
EPOCHS_PER_RUN = 6   # focused search, not full convergence

MODELS_DIR  = Path("models")
SCRIPT_PATH = Path("scripts/auto_research_iter1.py")

# Build the 48-combo grid
GRID = [
    {"arch_name": arch, "encoder_name": enc, "lr": lr,
     "weight_decay": wd, "aug_name": aug,
     "batch_size": BATCH_SIZE, "num_classes": NUM_CLASSES,
     "use_clahe": USE_CLAHE}
    for arch, enc in ARCH_ENCODER_MAP.items()
    for lr in LRS
    for wd in WDS
    for aug in AUGS
]
GRID_SIZE = len(GRID)
log.info(f"Pruned grid: {GRID_SIZE} combos | {EPOCHS_PER_RUN} epochs/run")


def run_experiment(cfg, experiment_index):
    """Run a single experiment and record val_dice."""
    suffix = subprocess.run(["date", "+%Y%m%d_%H%M%S"],
                            capture_output=True, text=True).stdout.strip()
    run_id = (f"exp{experiment_index:04d}_{cfg['arch_name']}_{cfg['encoder_name']}_"
              f"lr{cfg['lr']}_wd{cfg['weight_decay']}_bs{BATCH_SIZE}_"
              f"aug{cfg['aug_name']}_clahe{int(USE_CLAHE)}_{suffix}")
    model_dir = MODELS_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    ts = subprocess.run(["date", "+%Y-%m-%dT%H:%M:%S.%3N"],
                        capture_output=True, text=True).stdout.strip()
    init_cfg = {
        **cfg, "experiment_index": experiment_index,
        "run_id": run_id,
        "timestamp": ts,
        "val_dice": 0.0, "val_iou": 0.0,
        "epochs": EPOCHS_PER_RUN,
    }
    (model_dir / "config.json").write_text(json.dumps(init_cfg, indent=2))

    # Set up a fixed grid file with only this one combo
    grid_file = Path("data/processed/fast_grid.json")
    grid_file.write_text(json.dumps([cfg]))

    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--epochs-per-run", str(EPOCHS_PER_RUN),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    val_dice = 0.0
    if result.returncode == 0:
        for line in reversed(result.stdout.splitlines()):
            if "Dice=" in line:
                try:
                    val_dice = float(line.split("Dice=")[1].split()[0])
                except: pass
                break
    else:
        log.warning(f"  Exp #{experiment_index} failed (exit={result.returncode})")

    # Update config with result
    cfg["val_dice"] = val_dice
    cfg["experiment_index"] = experiment_index
    cfg["run_id"] = run_id
    cfg["timestamp"] = ts
    (model_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    log.info(f"  Exp #{experiment_index}: {cfg['arch_name']} | dice={val_dice:.4f}")
    return val_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=0)
    args = parser.parse_args()

    grid = GRID[args.start_idx:]
    if args.max_runs > 0:
        grid = grid[:args.max_runs]

    total = len(grid)
    log.info(f"Running {total}/{GRID_SIZE} combos from index {args.start_idx}")
    best_dice = 0.0

    for i, cfg in enumerate(grid):
        exp_idx = args.start_idx + i
        dice = run_experiment(cfg, exp_idx)
        if dice > best_dice:
            best_dice = dice
            log.info(f"  *** New best: {best_dice:.4f}")
        log.info(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    log.info(f"DONE. Best Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
