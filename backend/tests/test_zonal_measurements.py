"""
test_global_minimum.py
========================
TDD tests for the redesigned `calculate_global_minimum()` function.

Clinical contract:
  Sweeps the ENTIRE working length of the root (alveolar crest + 1.5mm offset →
  apex), casts perpendicular rays from the TOOTH SURFACE (not from the axis
  center), and finds the SINGLE absolute minimum distance to each bone plate.

  Returns a dict with exactly 4 keys:
    labial_line:  [[x_tooth, y_tooth], [x_bone, y_bone]]  — thinnest labial gap
    palatal_line: [[x_tooth, y_tooth], [x_bone, y_bone]]  — thinnest palatal gap
    labial_mm:    float — pre-computed mm value
    palatal_mm:   float — pre-computed mm value

  Origin (x1, y1) of each line MUST be at the tooth surface, not at the axis.
  Palatal bone is on the -u1_perp side; Labial bone is on the +u1_perp side.

Synthetic geometry (corrected from previous version — mask sides were swapped):
  Tooth:        columns 56–71  (full height)
  Labial bone:  columns 74–88  (+x side = u1_perp direction)
  Palatal bone: columns 40–54  (-x side = -u1_perp direction)

  tip = (64, 10), apex = (64, 110) → u1_unit = [0, 1], u1_perp = [1, 0]
  Labial surface of tooth ≈ x=71; bone starts at x=74.  Gap ≈ 3 px.
  Palatal surface of tooth ≈ x=56; bone ends at x=54.   Gap ≈ 2 px.
"""
import math
import numpy as np
import pytest

from app.services.analysis_service import (
    calculate_global_minimum,
    _get_u1_perp,
    MASK_IDX_UPPER_INCISOR,
    MASK_IDX_LABIAL_BONE,
    MASK_IDX_PALATAL_BONE,
)

MM_PER_PIXEL = 0.1   # 10 px per mm — synthetic calibration

EXPECTED_KEYS = {"labial_line", "palatal_line", "labial_mm", "palatal_mm"}


def _make_masks(h: int = 128, w: int = 128):
    """
    Corrected synthetic masks (anatomically consistent with u1_perp = [1, 0]):

      Tooth:        columns 56–71  (full height)
      Labial bone:  columns 74–88  (RIGHT of tooth = +x = u1_perp direction)
      Palatal bone: columns 40–54  (LEFT  of tooth = -x = -u1_perp direction)
    """
    tooth   = np.zeros((h, w), dtype=np.uint8)
    labial  = np.zeros((h, w), dtype=np.uint8)
    palatal = np.zeros((h, w), dtype=np.uint8)

    tooth[:, 56:72]  = 1   # tooth: cols 56–71
    labial[:, 74:89] = 1   # labial bone: cols 74–88 (RIGHT, +x direction)
    palatal[:, 40:55] = 1  # palatal bone: cols 40–54 (LEFT, -x direction)

    # masks[MASK_IDX_UPPER_INCISOR]=0, MASK_IDX_LABIAL_BONE=1, MASK_IDX_PALATAL_BONE=2
    return [tooth, labial, palatal]


def _make_axis(h: int = 128):
    """Vertical tooth axis: tip bottom, apex top (standard ceph orientation)."""
    tip  = np.array([64.0, 10.0], dtype=np.float32)
    apex = np.array([64.0, float(h - 10)], dtype=np.float32)
    vec  = apex - tip
    u1_unit = vec / np.linalg.norm(vec)
    u1_perp = _get_u1_perp(u1_unit)
    return tip, apex, u1_unit, u1_perp


# ── Cycle 1 — exactly 4 required keys ────────────────────────────────────────

def test_global_min_returns_four_keys():
    """calculate_global_minimum must return exactly labial_line, palatal_line,
    labial_mm, palatal_mm — no more, no less."""
    masks = _make_masks()
    tip, apex, u1_unit, u1_perp = _make_axis()

    result = calculate_global_minimum(
        tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL
    )

    assert isinstance(result, dict), "Result must be a dict"
    assert set(result.keys()) == EXPECTED_KEYS, (
        f"Got keys {set(result.keys())} — expected {EXPECTED_KEYS}"
    )


# ── Cycle 2 — each line is [[x1,y1],[x2,y2]] ─────────────────────────────────

def test_global_min_lines_are_two_xy_pairs():
    """Each line value must be [[x1,y1],[x2,y2]] — a 2-element list of 2-element lists."""
    masks = _make_masks()
    tip, apex, u1_unit, u1_perp = _make_axis()

    result = calculate_global_minimum(
        tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL
    )

    for key in ("labial_line", "palatal_line"):
        seg = result[key]
        assert isinstance(seg, list),  f"{key}: must be list"
        assert len(seg) == 2,          f"{key}: must have 2 points"
        for pt in seg:
            assert isinstance(pt, list), f"{key}: point must be list"
            assert len(pt) == 2,         f"{key}: each point must be [x, y]"


# ── Cycle 3 — all coordinates finite ─────────────────────────────────────────

def test_global_min_coords_finite():
    """No NaN or Inf in any coordinate or mm value."""
    masks = _make_masks()
    tip, apex, u1_unit, u1_perp = _make_axis()

    result = calculate_global_minimum(
        tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL
    )

    for key in ("labial_line", "palatal_line"):
        for pt in result[key]:
            for v in pt:
                assert math.isfinite(v), f"{key}: coord {v!r} is not finite"
    assert math.isfinite(result["labial_mm"]),  "labial_mm is not finite"
    assert math.isfinite(result["palatal_mm"]), "palatal_mm is not finite"


# ── Cycle 4 — labial line origin is at TOOTH SURFACE (not axis center) ───────

def test_labial_line_origin_is_at_tooth_surface():
    """
    The origin [x1, y1] of labial_line must be at the labial tooth surface,
    NOT at the axis center.

    With tooth at columns 56–71 and u1_perp=[1,0] (rightward = labial),
    the labial tooth surface is at x≈71. The origin must be within ±2 px of that.
    """
    masks = _make_masks()
    tip, apex, u1_unit, u1_perp = _make_axis()

    result = calculate_global_minimum(
        tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL
    )

    x1_labial = result["labial_line"][0][0]
    # Labial tooth surface = rightmost edge of tooth ≈ col 71
    assert abs(x1_labial - 71.0) <= 3.0, (
        f"Labial origin x={x1_labial:.1f} must be ≈71 (labial tooth surface), "
        f"not at axis center (≈64). Bug: origin is NOT on tooth surface."
    )


# ── Cycle 5 — labial line end is in the labial bone ──────────────────────────

def test_labial_line_end_is_in_bone_region():
    """
    The endpoint [x2, y2] of labial_line must land inside the labial bone region
    (columns 74–88). If x2 > x1 (rightward), this confirms the line crosses the gap.
    """
    masks = _make_masks()
    tip, apex, u1_unit, u1_perp = _make_axis()

    result = calculate_global_minimum(
        tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL
    )

    x1, _  = result["labial_line"][0]
    x2, _  = result["labial_line"][1]
    # Line must go rightward (from tooth surface toward labial bone)
    assert x2 > x1, (
        f"Labial line must go rightward (+x): x1={x1:.1f}, x2={x2:.1f}"
    )
    # Endpoint should be inside/at labial bone (cols 74–88)
    assert 73.0 <= x2 <= 90.0, (
        f"Labial endpoint x2={x2:.1f} must be in labial bone region (74–88)"
    )


# ── Cycle 6 — mm values match pixel segment length × calibration ─────────────

def test_mm_values_match_pixel_distance():
    """
    labial_mm must equal (Euclidean pixel length of labial_line) × mm_per_pixel.
    This verifies the pre-computed value is consistent with the segment geometry.
    """
    masks = _make_masks()
    tip, apex, u1_unit, u1_perp = _make_axis()

    result = calculate_global_minimum(
        tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL
    )

    for key, mm_key in [("labial_line", "labial_mm"), ("palatal_line", "palatal_mm")]:
        (x1, y1), (x2, y2) = result[key]
        px_dist = math.hypot(x2 - x1, y2 - y1)
        expected_mm = px_dist * MM_PER_PIXEL
        actual_mm = result[mm_key]
        assert abs(actual_mm - expected_mm) < 0.01, (
            f"{mm_key}={actual_mm:.4f} != pixel_dist×mpp={expected_mm:.4f} "
            f"(px_dist={px_dist:.2f}, mmp={MM_PER_PIXEL})"
        )


# ── Cycle 7 — independence: no mutation of any input ─────────────────────────

def test_global_minimum_does_not_mutate_inputs():
    """
    calculate_global_minimum must not modify tip, apex, u1_unit, u1_perp, or masks.
    """
    masks = _make_masks()
    tip, apex, u1_unit, u1_perp = _make_axis()

    tip_before    = tip.copy()
    apex_before   = apex.copy()
    u1_unit_before = u1_unit.copy()
    u1_perp_before = u1_perp.copy()
    sums_before   = [m.sum() for m in masks]

    calculate_global_minimum(tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL)

    np.testing.assert_array_equal(tip,     tip_before,     err_msg="tip was mutated")
    np.testing.assert_array_equal(apex,    apex_before,    err_msg="apex was mutated")
    np.testing.assert_array_equal(u1_unit, u1_unit_before, err_msg="u1_unit was mutated")
    np.testing.assert_array_equal(u1_perp, u1_perp_before, err_msg="u1_perp was mutated")
    for i, (m, s) in enumerate(zip(masks, sums_before)):
        assert m.sum() == s, f"masks[{i}] pixel sum changed from {s} to {m.sum()}"


# ── Cycle 8 — Custom cervical offset ─────────────────────────────────────────

def test_global_min_custom_cervical_offset():
    """
    Verify that passing a custom cervical_offset_mm doesn't break coordinate
    finiteness and structure. It shifts the starting boundary (t_start), which
    might or might not change the overall minimum depending on the mask shape,
    but the return contract should remain identical.
    """
    masks = _make_masks()
    tip, apex, u1_unit, u1_perp = _make_axis()

    result_default = calculate_global_minimum(
        tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL
    )
    result_custom = calculate_global_minimum(
        tip, apex, u1_unit, u1_perp, masks, MM_PER_PIXEL, cervical_offset_mm=3.0
    )

    # Output structure must remain exactly the same
    assert set(result_custom.keys()) == EXPECTED_KEYS
    for key in ("labial_line", "palatal_line"):
        assert len(result_custom[key]) == 2
        for pt in result_custom[key]:
            assert len(pt) == 2
            assert math.isfinite(pt[0]) and math.isfinite(pt[1])

