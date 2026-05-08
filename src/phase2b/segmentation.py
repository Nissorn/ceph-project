"""U-Net segmentation model for cephalometric polygon detection.

Segments three regions per image:
  Channel 0: Upper_incisor
  Channel 1: Labial_bone
  Channel 2: Palatal_bone

Architecture: UNet with ResNet-34 encoder (pretrained ImageNet)
Library:      segmentation-models-pytorch (smp)

Install:
    pip install segmentation-models-pytorch

NOTE: Currently 0/104 images have polygon annotations.
This scaffold is ready to train the moment Dr. provides polygon data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Polygon label ordering — must be consistent everywhere
POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]
NUM_SEG_CLASSES = len(POLYGON_CLASSES)


def build_segmentation_model(
    num_classes: int = NUM_SEG_CLASSES,
    encoder_name: str = "resnet34",
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a UNet segmentation model with a pretrained ResNet-34 encoder.

    Args:
        num_classes:  Number of output channels (= number of polygon types).
        encoder_name: timm/smp encoder name. resnet34 is lightweight and accurate.
        pretrained:   Whether to load ImageNet weights for the encoder.

    Returns:
        nn.Module ready for training.

    Raises:
        ImportError: if segmentation-models-pytorch is not installed.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation-models-pytorch is required for P7.\n"
            "Install with: pip install segmentation-models-pytorch"
        )

    encoder_weights = "imagenet" if pretrained else None

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,  # Raw logits — sigmoid applied in loss/inference
    )
    return model


class SegmentationLoss(nn.Module):
    """
    Combined Dice + BCE loss for segmentation.

    Dice handles class imbalance within masks (foreground << background).
    BCE stabilises training, especially early on.
    Both terms weighted equally at 0.5.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] raw logits
            target: [B, C, H, W] float binary masks in {0, 1}
        Returns:
            Scalar combined loss.
        """
        bce_loss = self.bce(pred, target)

        # Dice loss
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(2, 3))
        union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice.mean()

        return 0.5 * bce_loss + 0.5 * dice_loss
