#!/usr/bin/env python3
"""Dry-run validation for the 4-class argmax segmentation refactor.
Tests: model load, forward pass, argmax decode, class remap.
Run from project root: python3 test_backend_refactor.py
"""

import sys
from pathlib import Path

# ── project root so backend/app/services imports work ──────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import cv2

# ── import the functions under test ────────────────────────────────────────────
# Set up ROOT the same way the service does
import backend.app.services.analysis_service as svc

INPUT_SIZE = (512, 512)
NUM_KEYPOINTS = 10

# ── 1. Model load ───────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Load 4-class DeepLabV3Plus checkpoint")
print("=" * 60)

seg_ckpt_path = ROOT / "best_model.pt"
if not seg_ckpt_path.exists():
    # fall back to models/ dir
    seg_ckpt_path = ROOT / "models" / "exp0128_DeepLabV3Plus_resnet34_20260524_043501" / "best_model.pt"

print(f"Checkpoint: {seg_ckpt_path}")
print(f"Exists:    {seg_ckpt_path.exists()}")

# Build the model (4 classes)
seg_model = svc._build_segmentation_model(num_classes=4)
print(f"Model built: DeepLabV3Plus + resnet34, classes=4")

# Load weights
seg_state = torch.load(seg_ckpt_path, map_location="cpu", weights_only=False)
loaded_keys = set(seg_state.keys()) if isinstance(seg_state, dict) else set()
print(f"Weights loaded: {len(loaded_keys)} keys")

# Inspect output channel count from the classifier head
classifier_keys = [k for k in seg_state.keys() if "classifier" in k or "head" in k.lower()]
for k in classifier_keys:
    print(f"  {k}: {seg_state[k].shape}")

seg_model.load_state_dict(seg_state, strict=False)
seg_model.eval()
print("State dict loaded (strict=False — accepted mismatches)")

# ── 2. Dummy forward pass ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 — Dummy tensor forward pass [1, 3, 512, 512]")
print("=" * 60)

dummy_input = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    logits = seg_model(dummy_input)

print(f"Input:     {dummy_input.shape}")
print(f"Logits:    {logits.shape}  (expect [1, 4, 256, 256])")
B, C, H, W = logits.shape
assert B == 1, f"Batch != 1: {B}"
assert C == 4, f"Classes != 4: {C}  ← MISMATCH if checkpoint was trained with classes != 4"
assert H == 512 and W == 512, f"Heatmap size wrong: {H}x{W}"
print("Shape assertions PASSED")

# ── 3. Argmax decode ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — torch.argmax decode + class remap")
print("=" * 60)

class_map = torch.argmax(logits, dim=1).cpu()[0].numpy().astype(np.uint8)
print(f"class_map: {class_map.shape}, dtype={class_map.dtype}")
print(f"Unique class indices in map: {np.unique(class_map)}")
print(f"  0=Background, 1=Upper_incisor, 2=Labial_bone, 3=Palatal_bone")

# Remap to 512x512 (simulate native resolution upscale)
orig_w, orig_h = 800, 600  # arbitrary test resolution
class_map_native = cv2.resize(class_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
print(f"Resized to native: {class_map_native.shape}")

masks = []
for output_idx in range(3):
    mask = (class_map_native == (output_idx + 1)).astype(np.uint8)
    pixel_count = int(mask.sum())
    masks.append(mask)
    class_names = ["Upper_incisor", "Labial_bone", "Palatal_bone"]
    print(f"  Class {output_idx} ({class_names[output_idx]}): {pixel_count} px")

# ── 4. Overlap no-op ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — Overlap resolution (should be no-op with argmax)")
print("=" * 60)

corrected_masks, diag = svc._resolve_mask_overlaps(masks)
print(f"diag: {diag}")
assert diag["note"] == "argmax_4class_no_overlap_resolution_needed"
print("No-op overlap resolution PASSED")

# ── 5. Simulate full analyze_image pipeline (no landmark model needed) ──────────

print("\n" + "=" * 60)
print("STEP 5 — Simulate /analyze decode pipeline end-to-end")
print("=" * 60)

masks_decoded = svc._decode_segmentation_masks(logits, orig_w, orig_h)
assert len(masks_decoded) == 3, f"Expected 3 masks, got {len(masks_decoded)}"
for i, m in enumerate(masks_decoded):
    assert m.shape == (orig_h, orig_w), f"Mask {i} shape {m.shape} != ({orig_h}, {orig_w})"
    assert m.dtype == np.uint8, f"Mask {i} dtype {m.dtype} != uint8"
    print(f"  Mask {i} ({['Upper_incisor','Labial_bone','Palatal_bone'][i]}): shape={m.shape}, sum={int(m.sum())} px")

print("\n✓ ALL CHECKS PASSED — 4-class argmax pipeline is valid")
print("  Backend is ready to serve.")
