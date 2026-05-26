"""
Tests for segmentation mask decoding behavior.

Architecture: 3 foreground-only sigmoid heads (no background class channel).
Each class is decoded independently via sigmoid + threshold. This means:
  - Background pixels (all logits low) appear in ZERO masks — correct.
  - Foreground pixels appear in the mask of their class.
  - Rare overlap between classes is handled downstream by _resolve_mask_overlaps.

argmax is WRONG here because it forces every background pixel to be assigned
to the highest-scoring foreground class, destroying small minority classes
like Upper_incisor whose logits are dominated by the larger bone structures.
"""
import numpy as np
import torch

from app.services.analysis_service import _decode_segmentation_masks

THRESHOLD = 0.6  # must match implementation


# ── Tracer bullet: background pixels excluded ──────────────────────────────

def test_background_pixels_belong_to_no_class():
    """A pixel where all sigmoid outputs are below threshold has no class mask.

    argmax would always assign such pixels to the highest-scoring foreground
    class, which destroys minority classes like Upper_incisor.
    """
    # All logits strongly negative → sigmoid << threshold for every class
    logits = torch.full((1, 3, 4, 4), -5.0)

    masks = _decode_segmentation_masks(logits, orig_w=4, orig_h=4)

    combined = sum(m.astype(int) for m in masks)
    assert (combined == 0).all(), (
        "Background pixels (all logits low) must appear in ZERO class masks"
    )


# ── High-confidence foreground pixel lands in the right class ──────────────

def test_high_logit_pixel_assigned_to_that_class_only():
    """A pixel with only class 1 logit high appears in mask 1 and not in 0 or 2."""
    logits = torch.full((1, 3, 4, 4), -5.0)
    # Make the top-left pixel strongly activate class 1 only
    logits[0, 1, 0, 0] = 5.0

    masks = _decode_segmentation_masks(logits, orig_w=4, orig_h=4)

    assert masks[1][0, 0] == 1, "Pixel with high class-1 logit must appear in mask 1"
    assert masks[0][0, 0] == 0, "Same pixel must NOT appear in mask 0"
    assert masks[2][0, 0] == 0, "Same pixel must NOT appear in mask 2"


# ── Size correctness: output shape matches (orig_h, orig_w) ────────────────

def test_masks_resized_to_native_resolution():
    """Masks must be returned at the requested native resolution, not 512×512."""
    logits = torch.randn(1, 3, 512, 512)
    orig_w, orig_h = 800, 600

    masks = _decode_segmentation_masks(logits, orig_w=orig_w, orig_h=orig_h)

    assert len(masks) == 3
    for i, mask in enumerate(masks):
        assert mask.shape == (orig_h, orig_w), (
            f"Mask {i} shape {mask.shape} != expected ({orig_h}, {orig_w})"
        )
