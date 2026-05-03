#!/usr/bin/env python3
"""Phase 2: Predict landmarks on a single image using a trained model checkpoint."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import cv2
import numpy as np

from src.phase2.model import CephalometricModel
from src.phase2.heatmap import decode_heatmaps
from src.utils.io import load_config


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    input_size: tuple[int, int],
    heatmap_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coords [N,2], confidence [N]) in original image pixel space."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    img_resized = cv2.resize(img, (input_size[1], input_size[0]))
    tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(tensor)
        if pred.shape[-2:] != tuple(heatmap_size):
            pred = torch.nn.functional.interpolate(
                pred, size=heatmap_size, mode="bilinear", align_corners=False
            )
        coords, confidence = decode_heatmaps(pred.cpu(), input_size)

    coords_np = coords[0].numpy()
    conf_np = confidence[0].numpy()

    # Scale back to original image size
    scale_x = orig_w / input_size[1]
    scale_y = orig_h / input_size[0]
    coords_np[:, 0] *= scale_x
    coords_np[:, 1] *= scale_y

    return coords_np, conf_np


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Predict landmarks on single image")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image", required=True, help="Path to input JPG image")
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"])
    input_size = tuple(cfg["model"]["input_size"])
    heatmap_size = tuple(cfg["model"]["heatmap_size"])
    kp_names = cfg["keypoints"]["names"]
    low_conf_thresh = cfg["evaluation"]["confidence_low_threshold"]

    model = CephalometricModel(num_keypoints=cfg["keypoints"]["num_keypoints"], pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    coords, confidence = predict_image(args.image, model, device, input_size, heatmap_size)

    print(f"\nPredictions for: {args.image}")
    print(f"{'Landmark':<20} {'x':>8} {'y':>8} {'conf':>8} {'flag'}")
    print("-" * 55)
    results = []
    for i, name in enumerate(kp_names):
        flag = "⚠ LOW" if confidence[i] < low_conf_thresh else ""
        print(f"{name:<20} {coords[i,0]:>8.1f} {coords[i,1]:>8.1f} {confidence[i]:>8.3f}  {flag}")
        results.append({
            "name": name, "x": float(coords[i, 0]),
            "y": float(coords[i, 1]), "confidence": float(confidence[i])
        })

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"image": args.image, "keypoints": results}, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
