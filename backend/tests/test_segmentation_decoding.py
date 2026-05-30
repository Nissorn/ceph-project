"""
Tests for segmentation mask decoding behavior under 4-class argmax architecture.

Architecture: 4-class (Background=0, Upper_incisor=1, Labial_bone=2, Palatal_bone=3).
Decoding via argmax over channel dimension.
"""
import numpy as np
import torch

from app.services.analysis_service import _decode_segmentation_masks


# ── Tracer bullet: background pixels excluded ──────────────────────────────

def test_background_pixels_belong_to_no_class():
    """A pixel where the background channel is highest has no class mask.

    Argmax assigns such pixels to background (channel 0), which is excluded from foreground masks.
    """
    # Background logit (channel 0) is high, other channels are low
    logits = torch.full((1, 4, 4, 4), -5.0)
    logits[0, 0, :, :] = 5.0

    masks = _decode_segmentation_masks(logits, orig_w=4, orig_h=4)

    combined = sum(m.astype(int) for m in masks)
    assert (combined == 0).all(), (
        "Background pixels (where channel 0 is highest) must appear in ZERO foreground masks"
    )


# ── High-confidence foreground pixel lands in the right class ──────────────

def test_high_logit_pixel_assigned_to_that_class_only():
    """A pixel with only class 2 (Labial_bone) logit high appears in mask 1 (Labial_bone) and not in 0 or 2."""
    logits = torch.full((1, 4, 4, 4), -5.0)
    logits[0, 0, :, :] = 0.0  # background baseline
    # Make the top-left pixel strongly activate class 2 only (Labial_bone)
    logits[0, 2, 0, 0] = 5.0

    masks = _decode_segmentation_masks(logits, orig_w=4, orig_h=4)

    assert masks[1][0, 0] == 1, "Pixel with high class-2 logit must appear in mask 1 (Labial_bone)"
    assert masks[0][0, 0] == 0, "Same pixel must NOT appear in mask 0 (Upper_incisor)"
    assert masks[2][0, 0] == 0, "Same pixel must NOT appear in mask 2 (Palatal_bone)"


# ── Size correctness: output shape matches (orig_h, orig_w) ────────────────

def test_masks_resized_to_native_resolution():
    """Masks must be returned at the requested native resolution, not 512×512."""
    logits = torch.randn(1, 4, 512, 512)
    orig_w, orig_h = 800, 600

    masks = _decode_segmentation_masks(logits, orig_w=orig_w, orig_h=orig_h)

    assert len(masks) == 3
    for i, mask in enumerate(masks):
        assert mask.shape == (orig_h, orig_w), (
            f"Mask {i} shape {mask.shape} != expected ({orig_h}, {orig_w})"
        )

