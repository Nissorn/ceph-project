#!/usr/bin/env python3
"""Debug why soft-argmax always returns (256,256)."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/iddi/ceph-v2')
from src.phase2.heatmap import SoftArgmax2D

input_size = (512, 512)
H, W = 256, 256

print("=" * 60)
print("Test 1: Random uniform heatmaps (simulating random init model)")
print("=" * 60)

# Simulate random-init model output: logits ~ N(0, 1)
torch.manual_seed(42)
logits = torch.randn(2, 10, H, W)  # 2 images, 10 keypoints

soft = SoftArgmax2D(temperature=10.0)
coords, conf = soft(logits, input_size)
print(f"coords:\n{coords[0]}\n")  # First image, all 10 keypoints
print(f"All same? {all(torch.allclose(coords[0][i], coords[0][0]) for i in range(10))}")
print(f"Dist from center: {torch.sqrt(((coords[0] - 256)**2).sum(-1)).mean():.2f}px")

print()
print("=" * 60)
print("Test 2: Heatmaps with one clear peak")
print("=" * 60)

logits2 = torch.zeros(1, 1, H, W)
logits2[0, 0, 100, 150] = 5.0  # Clear peak at (100, 150) in heatmap space
coords2, _ = soft(logits2, input_size)
print(f"Peak at (100,150) heatmap → input ({100/255*512:.1f}, {150/255*512:.1f})")
print(f"soft-argmax result: {coords2[0, 0].numpy()}")

print()
print("=" * 60)
print("Test 3: What does sigmoid(logits) look like for random init?")
print("=" * 60)

logits3 = torch.randn(1, 10, H, W)
sig = torch.sigmoid(logits3)
print(f"sigmoid range: [{sig.min():.4f}, {sig.max():.4f}], mean={sig.mean():.4f}")
print(f"exp(beta*sig) range: [{torch.exp(0.1*sig.min()):.4f}, {torch.exp(0.1*sig.max()):.4f}]")
print(f"Weight ratio (max/min): {torch.exp(0.1*sig.max())/torch.exp(0.1*sig.min()):.4f}")

print()
print("=" * 60)
print("Test 4: Check raw logits stats (what does model output at init?)")
print("=" * 60)

# What does the actual model output at init for our images?
from src.phase2.model import HeatmapHead, build_hrnet
backbone = build_hrnet(num_keypoints=10, pretrained=False)
model = torch.nn.Sequential(backbone, HeatmapHead(in_channels=2048, num_keypoints=10))
dummy = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    out = model(dummy)
print(f"Model init output stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
print(f"After sigmoid: min={torch.sigmoid(out).min():.4f}, max={torch.sigmoid(out).max():.4f}, mean={torch.sigmoid(out).mean():.4f}")

# What about with pretrained=True?
print()
print("With pretrained=True:")
backbone2 = build_hrnet(num_keypoints=10, pretrained=True)
model2 = torch.nn.Sequential(backbone2, HeatmapHead(in_channels=2048, num_keypoints=10))
with torch.no_grad():
    out2 = model2(dummy)
print(f"Output stats: min={out2.min():.4f}, max={out2.max():.4f}, mean={out2.mean():.4f}")
print(f"After sigmoid: min={torch.sigmoid(out2).min():.4f}, max={torch.sigmoid(out2).max():.4f}, mean={torch.sigmoid(out2).mean():.4f}")

# Now test soft-argmax on pretrained model output
soft10 = SoftArgmax2D(temperature=10.0)
coords3, _ = soft10(out2.cpu(), input_size)
print(f"\nPretrained model soft-argmax coords (temp=10):\n{coords3[0]}")
print(f"All same? {all(torch.allclose(coords3[0][i], coords3[0][0]) for i in range(10))}")

print("\nDone.")