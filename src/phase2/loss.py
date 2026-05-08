"""Adaptive Wing Loss for heatmap regression.

Source: Wang et al., "Adaptive Wing Loss for Robust Face Alignment
via Heatmap Regression" (ICCV 2019).

Key idea:
- Background pixels (GT ≈ 0): small penalty — model shouldn't be punished for
  missing near-zero regions.
- Foreground pixels (GT ≈ 1): large non-linear penalty — errors near the actual
  landmark location are penalised much more aggressively than MSE would.

This is critical for medical landmark detection where ~99% of heatmap pixels
are background zeros. MSELoss would learn to output a flat near-zero map and
still achieve low numerical loss while completely missing every keypoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap-based landmark detection.

    Parameters (from paper):
        omega  (float): Controls height of the wing function. Default 14.
        theta  (float): Boundary between linear and non-linear regions. Default 0.5.
        epsilon (float): Curvature of the non-linear region. Default 1.
        alpha   (float): Controls the rate of increase of the wing. Default 2.1.

    Usage:
        criterion = AdaptiveWingLoss()
        loss = criterion(pred_heatmaps, gt_heatmaps, valid_mask)

        # Or without mask (all pixels contribute):
        loss = criterion(pred_heatmaps, gt_heatmaps)
    """

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
    ):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Adaptive Wing Loss.

        Args:
            pred:   [B, K, H, W] — predicted heatmaps
            target: [B, K, H, W] — ground truth Gaussian heatmaps
            mask:   [B, K] bool  — True = landmark is annotated for that image.
                                   If None, all keypoints contribute.

        Returns:
            Scalar loss value.
        """
        # Per-pixel absolute error
        delta = (target - pred).abs()

        # Adaptive Wing formula
        alpha_minus_target = self.alpha - target
        A = (
            self.omega
            * (1 / (1 + (self.theta / self.epsilon) ** alpha_minus_target))
            * (alpha_minus_target)
            * ((self.theta / self.epsilon) ** (alpha_minus_target - 1))
            * (1 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(
            1 + (self.theta / self.epsilon) ** alpha_minus_target
        )

        # Wing-shaped loss:
        #   if |delta| < theta: non-linear wing region
        #   if |delta| >= theta: linear region
        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + (delta / self.epsilon) ** alpha_minus_target),
            A * delta - C,
        )

        if mask is not None:
            # mask: [B, K] -> [B, K, 1, 1] for broadcasting over H, W
            mask_4d = mask.unsqueeze(-1).unsqueeze(-1).float()
            losses = losses * mask_4d
            n_valid = mask_4d.sum().clamp(min=1.0)
            # Normalise by number of valid keypoints * spatial size
            H, W = pred.shape[-2:]
            return losses.sum() / (n_valid * H * W)

        return losses.mean()
