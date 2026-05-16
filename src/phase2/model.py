"""HRNet-W32 wrapper for 10-keypoint cephalometric landmark detection.
Outputs heatmaps [B, K, H, W] for use with heatmap loss + decoding."""

import torch
import torch.nn as nn


NUM_KEYPOINTS = 10


def build_hrnet(num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True) -> nn.Module:
    """
    Build HRNet-W32 with a heatmap regression head.

    Uses global_pool='' to get [B, 2048, 16, 16] feature maps instead of
    a flattened [B, 2048] classification vector. A lightweight conv head
    then produces [B, K, 256, 256] heatmaps.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required. Install with: pip install timm")

    backbone = timm.create_model(
        "hrnet_w32",
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
    )
    # backbone output with global_pool='': [B, 2048, 16, 16]
    return backbone


class HeatmapHead(nn.Module):
    """
    Learned transposed-convolution upsampling head.

    Architecture: [B, 2048, 16, 16]
      → Conv2d(2048, 256, 3x3) + BN + ReLU
      → 4× transposed_conv2d layers (each 2× spatial upscale)
         16→32→64→128→256
      → Conv2d(256, num_keypoints, 1x1)

    vs old head: single bilinear 16→128 (8×) which was too coarse.
    """

    def __init__(self, in_channels: int = 2048, num_keypoints: int = NUM_KEYPOINTS):
        super().__init__()

        # Channel reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Learned upsampling via transposed convolutions
        # 16 → 32 → 64 → 128 → 256 (4 stages, each 2×)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)

        # Final 1×1 to produce per-keypoint channels
        self.head = nn.Conv2d(128, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)      # [B, 256, 16, 16]
        x = self.up1(x)         # [B, 256, 32, 32]
        x = self.up2(x)         # [B, 256, 64, 64]
        x = self.up3(x)         # [B, 256, 128, 128]
        x = self.up4(x)         # [B, 128, 256, 256]
        x = self.head(x)        # [B, K, 256, 256]
        return x


class CephalometricModel(nn.Module):
    """
    Wraps HRNet-W32 backbone + HeatmapHead.

    Forward: [B, 3, 512, 512] → [B, K, 256, 256] heatmaps.
    Use heatmap.py decode functions to extract (x, y) + confidence.
    """

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True):
        super().__init__()
        self.backbone = build_hrnet(num_keypoints, pretrained)
        self.head = HeatmapHead(in_channels=2048, num_keypoints=num_keypoints)
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # [B, 2048, 16, 16]
        heatmaps = self.head(features)       # [B, K, 256, 256]
        return heatmaps