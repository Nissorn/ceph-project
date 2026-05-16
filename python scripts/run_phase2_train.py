#!/usr/bin/env python3
"""Phase 2: Run 5-Fold Cross-Validation training for landmark detection."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase2.train import run_kfold_training
from src.utils.io import load_config, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 2: 5-Fold CV landmark detection training")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default="outputs/kfold_metrics.json",
                        help="Path to write metrics JSON")
    parser.add_argument("--debug", action="store_true",
                        help="Dry-run: 1 fold, 2 epochs — smoke test only")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap annotated images used (for quick smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    landmarks_json = cfg["data"]["landmarks_json"]
    if not Path(landmarks_json).exists():
        print(f"ERROR: {landmarks_json} not found. Run phase 1 first.")
        sys.exit(1)

    if args.debug:
        print("DEBUG mode: 1 fold, 2 epochs — smoke test only\n")
    else:
        n_folds = cfg["training"].get("k_folds", 5)
        epochs = cfg["training"].get("epochs", 100)
        print(f"Starting {n_folds}-Fold CV training on device: {cfg['training']['device']}")
        print(f"Max {epochs} epochs per fold with early stopping (patience=15).\n")

    metrics = run_kfold_training(args.config, debug=args.debug, max_images=args.max_images)

    ensure_dir(str(Path(args.output).parent))
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nPhase 2 5-Fold CV results:")
    if metrics.get("note") == "no_data":
        print("  No annotated images — training skipped. Waiting for Dr.'s landmark annotations.")
    else:
        print(f"  MRE: {metrics['mre_mean_mm']:.3f} ± {metrics['mre_std_mm']:.3f} mm")
        if "fold_metrics" in metrics:
            print("  Per-fold MRE:")
            for fm in metrics["fold_metrics"]:
                print(f"    Fold {fm['fold']}: {fm['mre']:.2f} mm")
        for k, v in metrics.items():
            if k.startswith("sdr_"):
                print(f"  {k}: {v*100:.1f}%")
    print(f"\n  Full metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
