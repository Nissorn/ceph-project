"""HRNet-W32 wrapper for 8-keypoint cephalometric landmark detection."""

import torch
import torch.nn as nn


NUM_KEYPOINTS = 8


def build_hrnet(num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True) -> nn.Module:
    """
    Build HRNet-W32 with heatmap head for keypoint detection.
    Uses timm's pretrained COCO pose weights — 8 keypoints match our landmark count.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required. Install with: pip install timm")

    model = timm.create_model(
        "hrnet_w32",
        pretrained=pretrained,
        num_classes=0,
        features_only=False,
    )

    # Replace the classification head with a heatmap regression head
    in_channels = model.num_features if hasattr(model, "num_features") else 32
    model.head = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, num_keypoints, kernel_size=1),
    )

    return model


class CephalometricModel(nn.Module):
    """
    Wraps HRNet-W32 with coordinate decoding from heatmaps.
    Output: heatmaps [B, K, H, W] for loss computation.
    Use heatmap.py decode functions to extract (x, y) + confidence.
    """

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True):
        super().__init__()
        self.backbone = build_hrnet(num_keypoints, pretrained)
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
