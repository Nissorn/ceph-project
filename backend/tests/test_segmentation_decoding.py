"""
Tests for segmentation mask decoding behavior.

Behavior under test: _decode_segmentation_masks produces mutually-exclusive
binary masks — the model was trained with CrossEntropyLoss (multi-class, not
multi-label), so argmax must resolve competition between classes.
"""
import numpy as np
import torch
import pytest

from app.services.analysis_service import _decode_segmentation_masks


# ── Tracer bullet: mutual exclusivity ──────────────────────────────────────

def test_masks_are_mutually_exclusive():
    """No pixel may belong to more than one class."""
    # All logits positive → sigmoid+threshold would claim every pixel for every
    # class simultaneously (the bug).  Argmax must pick exactly one winner.
    logits = torch.ones(1, 3, 4, 4) * 2.0
    logits[0, 0, :2, :2] = 10.0   # class 0 dominates top-left quadrant
    logits[0, 1, 2:, 2:] = 10.0   # class 1 dominates bottom-right quadrant

    masks = _decode_segmentation_masks(logits, orig_w=4, orig_h=4)

    combined = sum(m.astype(int) for m in masks)
    assert (combined <= 1).all(), (
        "Overlapping masks detected: some pixels claimed by more than one class"
    )


# ── Full coverage: every pixel assigned ────────────────────────────────────

def test_every_pixel_assigned_to_exactly_one_class():
    """Argmax is total — every pixel must belong to exactly one class."""
    logits = torch.randn(1, 3, 8, 8)

    masks = _decode_segmentation_masks(logits, orig_w=8, orig_h=8)

    combined = sum(m.astype(int) for m in masks)
    assert (combined == 1).all(), (
        "Some pixels have no class assignment — coverage is not total"
    )


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
