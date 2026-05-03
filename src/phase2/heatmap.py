"""Gaussian heatmap encoding and decoding with confidence scores."""

import numpy as np
import torch
import torch.nn.functional as F


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

    Returns:
        heatmaps: [N, H_hm, W_hm] float32 in [0, 1]
    """
    N = len(keypoints)
    H, W = heatmap_size
    heatmaps = np.zeros((N, H, W), dtype=np.float32)

    scale_x = W / input_size[1]
    scale_y = H / input_size[0]

    for i in range(N):
        if not valid_mask[i]:
            continue
        x = keypoints[i, 0] * scale_x
        y = keypoints[i, 1] * scale_y

        x0, y0 = int(round(x)), int(round(y))
        size = int(6 * sigma + 1)
        xs = np.arange(0, size) - size // 2
        ys = np.arange(0, size) - size // 2
        xv, yv = np.meshgrid(xs, ys)
        gaussian = np.exp(-(xv**2 + yv**2) / (2 * sigma**2)).astype(np.float32)

        x_lo = max(0, x0 - size // 2)
        x_hi = min(W, x0 + size // 2 + 1)
        y_lo = max(0, y0 - size // 2)
        y_hi = min(H, y0 + size // 2 + 1)

        g_x_lo = max(0, -(x0 - size // 2))
        g_x_hi = g_x_lo + (x_hi - x_lo)
        g_y_lo = max(0, -(y0 - size // 2))
        g_y_hi = g_y_lo + (y_hi - y_lo)

        heatmaps[i, y_lo:y_hi, x_lo:x_hi] = np.maximum(
            heatmaps[i, y_lo:y_hi, x_lo:x_hi],
            gaussian[g_y_lo:g_y_hi, g_x_lo:g_x_hi],
        )

    return heatmaps


def decode_heatmaps(
    heatmaps: torch.Tensor,
    input_size: tuple[int, int] = (512, 512),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode heatmaps to (x, y) coordinates and confidence scores.

    Args:
        heatmaps: [B, N, H_hm, W_hm]
        input_size: (H_in, W_in) — used to scale coordinates back

    Returns:
        coords: [B, N, 2] float32 in input_size pixel space
        confidence: [B, N] float32 in [0, 1] — peak heatmap value
    """
    B, N, H, W = heatmaps.shape
    flat = heatmaps.view(B, N, -1)
    confidence, flat_idx = flat.max(dim=-1)

    x = (flat_idx % W).float()
    y = (flat_idx // W).float()

    x = x / W * input_size[1]
    y = y / H * input_size[0]

    coords = torch.stack([x, y], dim=-1)
    return coords, confidence
