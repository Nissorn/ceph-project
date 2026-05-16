"""Gaussian heatmap encoding with proper edge clamping + soft-argmax decoding."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmax2D(nn.Module):
    """
    Differentiable soft-argmax for sub-pixel keypoint localization.
    Replaces naive argmax which is quantized to integer heatmap cells.

   forward(heatmaps, input_size) -> (coords [B, N, 2], confidence [B, N])
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        # Beta (inverse temperature) — NON-LEARNABLE buffer.
        # Using a Parameter allowed training to collapse beta → ~0.1, which destroyed
        # soft-argmax spatial selectivity (near-uniform weighted average = center).
        # Fixed at initialization value throughout training so soft-argmax stays sharp.
        self.register_buffer("beta", torch.tensor(float(temperature)))

    def forward(
        self,
        heatmaps: torch.Tensor,
        input_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmaps: [B, N, H, W] — raw logits (no sigmoid applied here)
            input_size: (H_in, W_in)

        Returns:
            coords:    [B, N, 2] float32 in input pixel space (x, y)
            confidence: [B, N] float32 — peak heatmap value (after sigmoid)
        """
        B, N, H, W = heatmaps.shape
        device = heatmaps.device

        # sigmoid for normalized confidence scores
        conf = torch.sigmoid(heatmaps)

        # Soft-argmax: compute spatial expectation from sigmoid-normalized heatmaps
        # Use self.beta (learnable log-temperature) for sharpness control
        beta = F.softplus(self.beta)  # ensure positive

        # Reshape to [B*N, H*W] for matrix ops
        flat_conf = conf.view(B * N, H * W)

        # Spatial coordinate grids (normalized to [0, 1] then scaled to input_size)
        # x_coords: [W] values from 0 to W-1
        # y_coords: [H] values from 0 to H-1
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_coords = torch.arange(H, device=device, dtype=torch.float32)

        # Weighted average: sum(pos_i * exp(beta * conf_i)) / sum(exp(beta * conf_i))
        exp_c = torch.exp(beta * flat_conf)  # [B*N, H*W]
        sum_exp = exp_c.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Softmax-like normalization
        weights = exp_c / sum_exp  # [B*N, H*W], sum = 1

        # Compute weighted average for x and y
        # weights shape: [B*N, H*W]
        # x_coords: [W] → [1, 1, W] → [B*N, W]
        # y_coords: [H] → [H, 1] → [H, W] → [1, 1, H, W]
        x_w = weights.view(B * N, H, W) @ x_coords  # [B*N, H, W] @ [W] → [B*N, H]
        # Actually: weights @ x_coords gives [B*N, W] @ [W] = [B*N] — need proper broadcasting
        x_w = (weights.view(B * N, H, W) * x_coords.view(1, 1, W)).sum(dim=-1)  # [B*N, H]
        y_w = (weights.view(B * N, H, W) * y_coords.view(1, H, 1)).sum(dim=-2)  # [B*N, W]

        # Sum over spatial dims to get weighted coordinates
        x_out = x_w.sum(dim=-1)  # [B*N]
        y_out = y_w.sum(dim=-1)  # [B*N]

        # Scale to input_size (heatmap coords are already in [0, H-1]/[0, W-1])
        x_out = x_out / (W - 1) * input_size[1]  # normalize then scale
        y_out = y_out / (H - 1) * input_size[0]

        coords = torch.stack([x_out, y_out], dim=-1).view(B, N, 2)

        # Confidence = max of sigmoid heatmap
        confidence, _ = conf.view(B * N, H * W).max(dim=-1)
        confidence = confidence.view(B, N)

        return coords, confidence


def encode_heatmaps(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    heatmap_size: tuple[int, int],
    sigma: float = 2.0,
    input_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Encode (x, y) keypoints as Gaussian heatmaps.

    Args:
        keypoints: [N, 2] float32 in input_size pixel space
        valid_mask: [N] bool
        heatmap_size: (H_hm, W_hm)
        sigma: Gaussian std dev in heatmap pixels
        input_size: (H_in, W_in)

    Returns:
        heatmaps: [N, H_hm, W_hm] float32 in [0, 1]
    """
    N = len(keypoints)
    H, W = heatmap_size

    # Scale factor: keypoints in input space → heatmap space
    scale_x = W / input_size[1]
    scale_y = H / input_size[0]

    heatmaps = np.zeros((N, H, W), dtype=np.float32)

    # Pre-compute Gaussian kernel once
    size = int(6 * sigma + 1)
    half = size // 2
    xs = np.arange(size, dtype=np.float32) - half
    ys = np.arange(size, dtype=np.float32) - half
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    gaussian = np.exp(-(xv**2 + yv**2) / (2 * sigma**2)).astype(np.float32)

    for i in range(N):
        if not valid_mask[i]:
            continue

        # Map keypoint to heatmap coordinates
        x_raw = keypoints[i, 0] * scale_x
        y_raw = keypoints[i, 1] * scale_y

        # Clamp center to [1, size-2] so the Gaussian can bleed past edges.
        # Using floor() (not round()) for correct off-by-half behaviour.
        x0 = int(np.clip(np.floor(x_raw), 1, W - 2))
        y0 = int(np.clip(np.floor(y_raw), 1, H - 2))

        # Determine patch region (clipped to heatmap bounds)
        x_lo = max(0, x0 - half)
        x_hi = min(W, x0 + half + 1)
        y_lo = max(0, y0 - half)
        y_hi = min(H, y0 + half + 1)

        # Gaussian patch offset — how much of the kernel falls inside the heatmap
        g_x_lo = max(0, half - x0)
        g_x_hi = g_x_lo + (x_hi - x_lo)
        g_y_lo = max(0, half - y0)
        g_y_hi = g_y_lo + (y_hi - y_lo)

        # Max-pool with existing (handles overlapping near boundaries)
        heatmaps[i, y_lo:y_hi, x_lo:x_hi] = np.maximum(
            heatmaps[i, y_lo:y_hi, x_lo:x_hi],
            gaussian[g_y_lo:g_y_hi, g_x_lo:g_x_hi],
        )

    return heatmaps


def decode_heatmaps(
    heatmaps: torch.Tensor,
    input_size: tuple[int, int] = (512, 512),
    use_soft_argmax: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode heatmaps to (x, y) coordinates and confidence scores.

    Args:
        heatmaps: [B, N, H_hm, W_hm] — raw logits (no sigmoid)
        input_size: (H_in, W_in)
        use_soft_argmax: if False, fall back to argmax (for debugging)

    Returns:
        coords: [B, N, 2] float32 in input_size pixel space (x, y)
        confidence: [B, N] float32 — peak heatmap value
    """
    if use_soft_argmax:
        soft_argmax = SoftArgmax2D(temperature=10.0)
        coords, confidence = soft_argmax(heatmaps, input_size)
    else:
        # Fallback: naive argmax (quantized)
        B, N, H, W = heatmaps.shape
        conf = torch.sigmoid(heatmaps)
        flat = conf.view(B * N, -1)
        confidence, flat_idx = flat.max(dim=-1)
        x = (flat_idx % W).float() / W * input_size[1]
        y = (flat_idx // W).float() / H * input_size[0]
        coords = torch.stack([x, y], dim=-1).view(B, N, 2)
        confidence = confidence.view(B, N)

    return coords, confidence