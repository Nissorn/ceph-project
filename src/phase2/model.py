"""HRNet-W32 + CBAM wrapper for 10-keypoint cephalometric landmark detection.
Outputs heatmaps [B, K, H, W] for use with heatmap loss + decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_KEYPOINTS = 10


# ─────────────────────────────────────────────────────────────────────────────
# CBAM: Convolutional Block Attention Module
# Paper: arxiv:1807.06521 — helps model focus on subtle bone edges at
# posterior landmarks (ANS, PNS, PB, LB) and suppress soft-tissue noise.
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel attention: exploit inter-channel relationship to recalibrate
    feature importance. Uses both max-pool and avg-pool paths + shared MLP.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        b, c, h, w = x.shape

        # Average-pool + max-pool over spatial dimensions → [B, C]
        avg_out = self.shared_mlp(x.mean(dim=[2, 3]))      # [B, C]
        max_out = self.shared_mlp(x.amax(dim=[2, 3]))     # [B, C]

        # Combine and sigmoid-gate each channel
        attention = torch.sigmoid(avg_out + max_out)      # [B, C]
        return x * attention.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]


class SpatialAttention(nn.Module):
    """
    Spatial attention: exploit intra-spatial relationship to highlight
    discriminative landmark regions and suppress irrelevant background.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        # Pool along channel axis → [B, 1, H, W]
        avg_out = x.mean(dim=1, keepdim=True)   # avg over channels
        max_out = x.amax(dim=1, keepdim=True)    # max over channels
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(spatial_input))  # [B, 1, H, W]
        return x * attention  # [B, C, H, W]


class CBAM(nn.Module):
    """
    CBAM = Channel Attention → Spatial Attention (sequential).
    Injects after any feature tensor to recalibrate channel + spatial response.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


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
    """

    def __init__(self, in_channels: int = 2048, num_keypoints: int = NUM_KEYPOINTS):
        super().__init__()

        # Channel reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # CBAM recalibrates 256-channel features before spatial upsampling.
        # Helps the model focus on bone edges vs soft-tissue at posterior landmarks.
        self.cbam = CBAM(channels=256, reduction=16, spatial_kernel=7)

        # Learned upsampling via transposed convolutions
        # 16 → 32 → 64 → 128 → 256 (4 stages, each 2×)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)

        # Final 1×1 to produce per-keypoint heatmaps
        self.head = nn.Conv2d(256, num_keypoints, kernel_size=1)

        # ── EUPE Uncertainty Head ──────────────────────────────────────────────
        # Lightweight head predicting per-landmark uncertainty σ.
        # Architecture: [B, 256, 256, 256] → GlobalAvgPool → [B, 256]
        #   → Linear(256, num_keypoints) → [B, K] raw uncertainty scores
        # σ = softplus(raw) to ensure σ > 0.
        # Used in loss: L_eupe = (1/σ²) * L_mse + λ * log(σ)
        self.uncertainty_pool = nn.AdaptiveAvgPool2d(1)
        self.uncertainty_fc = nn.Linear(256, num_keypoints)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.reduce(x)      # [B, 256, 16, 16]
        x = self.cbam(x)        # [B, 256, 16, 16] — channel + spatial recalibration
        x = F.relu(self.up1(x)) # [B, 256, 32, 32]
        x = F.relu(self.up2(x)) # [B, 256, 64, 64]
        x = F.relu(self.up3(x)) # [B, 256, 128, 128]
        x = F.relu(self.up4(x)) # [B, 256, 256, 256]

        heatmaps = self.head(x)  # [B, K, 256, 256]

        # EUPE uncertainty: pool 256×256 features → [B, 256] → [B, K] uncertainty
        u = self.uncertainty_pool(x)            # [B, 256, 1, 1]
        u = u.view(u.size(0), -1)                # [B, 256]
        uncertainty = F.softplus(self.uncertainty_fc(u))  # [B, K], σ > 0

        return heatmaps, uncertainty


class CephalometricModel(nn.Module):
    """
    Wraps HRNet-W32 backbone + HeatmapHead.

    Forward: [B, 3, 512, 512] → (heatmaps [B, K, 256, 256], uncertainty [B, K])
    Uncertainty values σ_k encode per-landmark prediction confidence.
    Use heatmap.py decode functions to extract (x, y) + confidence.
    """

    def __init__(self, num_keypoints: int = NUM_KEYPOINTS, pretrained: bool = True):
        super().__init__()
        self.backbone = build_hrnet(num_keypoints, pretrained)
        self.head = HeatmapHead(in_channels=2048, num_keypoints=num_keypoints)
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)           # [B, 2048, 16, 16]
        heatmaps, uncertainty = self.head(features)  # [B, K, 256, 256], [B, K]
        return heatmaps, uncertainty