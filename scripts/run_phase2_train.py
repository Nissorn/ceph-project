#!/usr/bin/env python3
"""Phase 2: Run LOPO cross-validation training for landmark detection."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase2.train import run_lopo_training
from src.utils.io import load_config, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 2: LOPO landmark detection training")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default="outputs/lopo_metrics.json",
                        help="Path to write metrics JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)

    landmarks_json = cfg["data"]["landmarks_json"]
    if not Path(landmarks_json).exists():
        print(f"ERROR: {landmarks_json} not found. Run phase 1 first.")
        sys.exit(1)

    print(f"Starting LOPO training on device: {cfg['training']['device']}")
    print("This will take a long time (52 folds × 100 epochs).\n")

    metrics = run_lopo_training(args.config)

    ensure_dir(str(Path(args.output).parent))
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nPhase 2 LOPO results:")
    print(f"  MRE: {metrics['mre_mean_mm']:.3f} ± {metrics['mre_std_mm']:.3f} mm")
    for k, v in metrics.items():
        if k.startswith("sdr_"):
            print(f"  {k}: {v*100:.1f}%")
    print(f"\n  Full metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
