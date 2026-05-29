"""Loss functions for cephalometric landmark detection.

- AdaptiveWingLoss: per-pixel wing loss for foreground-biased heatmap regression.
- EUPELoss: Uncertainty-aware loss that learns per-landmark noise sigma.
  Based on the EUPE (Expected Uncertainty Parameter Estimation) principle —
  jointly predict landmark coordinates AND per-landmark uncertainty σ_k.
  Loss: L = (1/σ²) * L_regression + λ * log(σ)
  where σ is learned per keypoint (not global). This lets easy landmarks
  (high contrast: Palatal_crest, Labial_crest) learn small σ and hard landmarks
  (low contrast: ANS, PNS, PB, LB) learn large σ, naturally balancing gradient
  magnitudes across landmarks without manual tuning.
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
            # Normalise by number of valid keypoints only — NOT spatial size.
            # Dividing by H*W was killing gradient magnitudes and preventing learning.
            # Each keypoint contributes H*W spatial positions to the loss; we want
            # the average loss per keypoint (not per spatial pixel).
            return losses.sum() / n_valid

        return losses.mean()


# ─────────────────────────────────────────────────────────────────────────────
# EUPE Loss: Uncertainty-Weighted Heatmap Regression
# Paper: Various works on uncertainty-aware landmark detection.
# Key insight: jointly learn per-landmark noise σ_k so easy landmarks
# (Palatal_crest, Labial_crest — high contrast) get small σ and hard landmarks
# (ANS, PNS, PB, LB — low contrast) get large σ. The loss:
#   L = (1/σ_k²) * L_mse + λ * log(σ_k)
# The first term weights down high-loss samples (reducing gradient from noisy
# difficult landmarks), the second term prevents σ collapsing to 0.
#
# Implementation note:
#   σ_k is predicted by the head as a per-keypoint scalar. The loss above is
#   computed per-pixel then aggregated by keypoint. The weighting term (1/σ²)
#   is therefore applied to the whole heatmap for each keypoint.
#
# λ (reg_lambda) = 0.1 was chosen empirically: too small lets σ→0 (mode
# collapse), too large makes all σ similar (no benefit over uniform).
# ─────────────────────────────────────────────────────────────────────────────

class EUPELoss(nn.Module):
    """
    Uncertainty-weighted loss with per-keypoint learned sigma.

    L_eupe = sum_k [ (1/σ_k²) * L_k + λ * log(σ_k) ]
    where L_k is the AdaptiveWingLoss for keypoint k.

    Args:
        reg_lambda: Weight for the log-sigma regularization term (default 0.1).
                    Prevents sigma from collapsing to 0 (mode collapse).
    """

    def __init__(self, reg_lambda: float = 0.1):
        super().__init__()
        self.reg_lambda = reg_lambda

    def forward(
        self,
        pred_heatmaps: torch.Tensor,      # [B, K, H, W] predicted
        gt_heatmaps: torch.Tensor,        # [B, K, H, W] GT
        uncertainty: torch.Tensor,        # [B, K] predicted σ_k per keypoint
        mask: Optional[torch.Tensor] = None,  # [B, K] bool
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns: (total_loss, log_dict) where log_dict contains
                 per-component losses for monitoring.
        """
        B, K = uncertainty.shape   # [B, K], σ_k > 0

        # Compute per-pixel wing loss for each keypoint
        delta = (pred_heatmaps - gt_heatmaps).abs()
        alpha_minus_target = 2.1 - gt_heatmaps  # alpha = 2.1 in AdaptiveWing
        omega, theta, epsilon = 14.0, 0.5, 1.0

        A = omega * (1 / (1 + (theta / epsilon) ** alpha_minus_target)) \
            * alpha_minus_target * ((theta / epsilon) ** (alpha_minus_target - 1)) \
            * (1 / epsilon)
        C = theta * A - omega * torch.log(
            1 + (theta / epsilon) ** alpha_minus_target
        )
        losses_per_px = torch.where(
            delta < theta,
            omega * torch.log(1 + (delta / epsilon) ** alpha_minus_target),
            A * delta - C,
        )  # [B, K, H, W]

        # Apply mask if provided: [B, K] → [B, K, 1, 1]
        if mask is not None:
            mask_4d = mask.unsqueeze(-1).unsqueeze(-1).float()
            losses_per_px = losses_per_px * mask_4d

        # Aggregate per keypoint: mean over H×W spatial pixels → [B, K]
        loss_per_kp = losses_per_px.mean(dim=[-1, -2])  # [B, K]

        # Uncertainty weighting: (1/σ_k²) * loss_k
        sigma_sq = uncertainty.pow(2)  # [B, K], σ > 0
        weighted_loss = loss_per_kp / sigma_sq   # [B, K]

        # Regularization: λ * log(σ_k) — prevents σ → 0
        reg_term = self.reg_lambda * torch.log(uncertainty)  # [B, K]

        if mask is not None:
            # Count valid keypoints: [B, K] → scalar
            n_valid = mask.sum(dim=-1).clamp(min=1.0)  # [B]
            n_total = n_valid.sum()
            base_loss = (weighted_loss + reg_term) * mask.float()  # [B, K]
            loss = base_loss.sum() / n_total
        else:
            loss = (weighted_loss + reg_term).mean()

        log_dict = {
            "base": loss_per_kp.mean().item(),
            "weighted": weighted_loss.mean().item(),
            "reg": reg_term.mean().item(),
            "sigma_mean": uncertainty.mean().item(),
        }
        return loss, log_dict
