"""
Phase 2c — Treatment Recommendation Engine
==========================================
Fusion model: EfficientNet-B3 image branch + MLP scalar branch.

Scalar input tensor shape: [B, 6]
    [0] u1_pp_angle_deg
    [1] lb_apex_dist_mm   (labial bone-to-apex clearance)
    [2] pb_apex_dist_mm   (palatal bone-to-apex clearance)
    [3] apex_labial       one-hot: root_apex_position == "Labial"
    [4] apex_midway       one-hot: root_apex_position == "Midway"
    [5] apex_palatal      one-hot: root_apex_position == "Palatal"

Training schedule (staged):
    Stage 1 (epoch < backbone_unfreeze_epoch):
        EfficientNet-B3 fully frozen. Train MLP branch + fusion head only. LR = 1e-4.
    Stage 2 (epoch >= backbone_unfreeze_epoch):
        Unfreeze EfficientNet blocks[5], blocks[6], conv_head. Joint fine-tune. LR = 1e-5.
    Config key: phase2c.backbone_unfreeze_epoch (default 20)
"""

from __future__ import annotations

import torch
import torch.nn as nn


TREATMENT_CLASSES = [
    "Uncontrolled_tipping",
    "Controlled_tipping",
    "Translation",
    "Root_torque",
    "Extrusion",
    "Intrusion",
]
NUM_TREATMENT_CLASSES = len(TREATMENT_CLASSES)

APEX_POSITION_LABELS = ["Labial", "Midway", "Palatal"]
SCALAR_DIM = 6  # [angle, lb_dist, pb_dist, apex_labial, apex_midway, apex_palatal]

_MLP_HIDDEN = 64
_MLP_OUT = 32
_EFFICIENTNET_FEATURES = 1536  # EfficientNet-B3 global-pooled feature dim


class _ScalarMLP(nn.Module):
    """Small MLP branch that encodes the 6-dim Phase 3 scalar vector."""

    def __init__(self, input_dim: int = SCALAR_DIM, hidden: int = _MLP_HIDDEN, out: int = _MLP_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionClassifier(nn.Module):
    """
    EfficientNet-B3 image branch + scalar MLP branch, fused before classification head.

    Args:
        num_classes: Number of treatment output classes. Default 6.
        pretrained:  Load ImageNet weights for EfficientNet-B3. Default True.
    """

    def __init__(self, num_classes: int = NUM_TREATMENT_CLASSES, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required: pip install timm")

        # Image branch — global-pooled features, no classifier head
        self.cnn = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,  # feature extractor only
        )
        cnn_out_dim = self.cnn.num_features  # 1536 for B3

        # Scalar branch
        self.mlp = _ScalarMLP(input_dim=SCALAR_DIM, hidden=_MLP_HIDDEN, out=_MLP_OUT)

        # Fusion head
        self.head = nn.Linear(cnn_out_dim + _MLP_OUT, num_classes)

        # Start with backbone frozen (Stage 1)
        self.freeze_backbone()

    def forward(self, image: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:   [B, 3, H, W] float32 in [0, 1]
            scalars: [B, 6] float32 — Phase 3 scalar vector

        Returns:
            logits [B, num_classes] — apply sigmoid for probabilities
        """
        img_feat = self.cnn(image)           # [B, 1536]
        scalar_feat = self.mlp(scalars)      # [B, 32]
        fused = torch.cat([img_feat, scalar_feat], dim=1)  # [B, 1568]
        return self.head(fused)              # [B, num_classes]

    # ------------------------------------------------------------------
    # Training schedule helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Stage 1: freeze all EfficientNet-B3 parameters."""
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_top_blocks(self) -> None:
        """Stage 2: unfreeze blocks[5], blocks[6], and conv_head of EfficientNet-B3."""
        for block_idx in (5, 6):
            for param in self.cnn.blocks[block_idx].parameters():
                param.requires_grad = True
        for param in self.cnn.conv_head.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> list:
        """Return only parameters with requires_grad=True (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]


def compute_pos_weights(
    records: list[dict],
    min_support: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.

    Classes with fewer than min_support positives receive neutral weight (1.0)
    and are returned in insufficient_classes so callers can exclude them from
    AUC/F1 evaluation and log them as "insufficient_data".

    Returns:
        pos_weights:          [NUM_TREATMENT_CLASSES] — use directly with BCEWithLogitsLoss
        active_mask:          [NUM_TREATMENT_CLASSES] bool — True = enough data to train
        insufficient_classes: list of class names below min_support threshold
    """
    n_total = len(records)
    pos_counts = torch.zeros(NUM_TREATMENT_CLASSES)
    for r in records:
        treatments = set(r.get("treatment", []))
        for i, cls in enumerate(TREATMENT_CLASSES):
            if cls in treatments:
                pos_counts[i] += 1.0

    neg_counts = n_total - pos_counts
    active_mask = pos_counts >= min_support
    insufficient_classes = [
        TREATMENT_CLASSES[i] for i in range(NUM_TREATMENT_CLASSES) if not active_mask[i]
    ]

    # Rare classes get neutral weight (1.0) so BCEWithLogitsLoss still runs without blowing up.
    # Their contribution to the loss is semantically meaningless but numerically stable.
    # The training loop should zero out their gradient contribution or simply ignore their AUC.
    pos_weights = torch.where(
        active_mask,
        neg_counts / pos_counts.clamp(min=1.0),
        torch.ones(NUM_TREATMENT_CLASSES),
    )
    return pos_weights, active_mask, insufficient_classes


def apex_position_to_onehot(apex_pos: str) -> list[float]:
    """
    Encode root_apex_position string as one-hot.
    "Labial" → [1, 0, 0]
    "Midway" → [0, 1, 0]
    "Palatal"→ [0, 0, 1]
    """
    idx = APEX_POSITION_LABELS.index(apex_pos)
    one_hot = [0.0, 0.0, 0.0]
    one_hot[idx] = 1.0
    return one_hot
