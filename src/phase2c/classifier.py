"""EfficientNet-B3 multi-label treatment classifier.

Classifies which orthodontic treatment movements are present in a
T1 cephalogram image. This is a multi-label classification problem
because a patient can undergo multiple treatment types simultaneously.

Output classes (6 total):
    0: Uncontrolled_tipping
    1: Controlled_tipping
    2: Translation
    3: Root_torque
    4: Extrusion
    5: Intrusion

Architecture: EfficientNet-B3 (pretrained ImageNet), final FC replaced.
Loss:         BCEWithLogitsLoss with per-class positive weights for imbalance.
Metric:       Per-class AUC, macro-F1.

NOTE: Only T1 images are used. T2 images have no treatment labels.
      Quality_Reject and Low_Visibility images are excluded from training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Strict ordering — never change without retraining all models
TREATMENT_CLASSES = [
    "Uncontrolled_tipping",
    "Controlled_tipping",
    "Translation",
    "Root_torque",
    "Extrusion",
    "Intrusion",
]
NUM_TREATMENT_CLASSES = len(TREATMENT_CLASSES)


def build_classifier(
    num_classes: int = NUM_TREATMENT_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build EfficientNet-B3 multi-label classifier.

    Args:
        num_classes: Number of treatment classes. Default 6.
        pretrained:  If True, load ImageNet weights via timm.

    Returns:
        nn.Module with output shape [B, num_classes] (raw logits).
        Apply sigmoid for probabilities.

    Raises:
        ImportError: if timm is not installed.
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required. Install with: pip install timm"
        )

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def compute_pos_weights(records: list[dict]) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.

    Formula: pos_weight[i] = n_negative[i] / n_positive[i]
    This corrects for class imbalance — rare treatment types get higher weight.

    Args:
        records: Filtered list of records (T1 only, no Quality_Reject).

    Returns:
        Tensor of shape [NUM_TREATMENT_CLASSES] with per-class weights.
    """
    n_total = len(records)
    pos_counts = torch.zeros(NUM_TREATMENT_CLASSES)

    for r in records:
        treatments = set(r.get("treatment", []))
        for i, cls in enumerate(TREATMENT_CLASSES):
            if cls in treatments:
                pos_counts[i] += 1.0

    neg_counts = n_total - pos_counts
    # Clamp to avoid division by zero for unseen classes
    pos_counts_safe = pos_counts.clamp(min=1.0)
    pos_weights = neg_counts / pos_counts_safe
    return pos_weights
