"""
test_global_minimum.py
========================
TDD tests for the redesigned `calculate_global_minimum()` function.

Clinical contract:
  Sweeps the ENTIRE working length of the root (alveolar crest + offset -> apex).
  Pre-computes all offsets (0.0 to 5.0 mm) in a single pass for zero-latency UX.
  Finds the SINGLE absolute minimum distance to each bone plate for each offset.

  Returns a dict mapping string offsets to exactly 4 keys:
    labial_line:  [[x_tooth, y_tooth], [x_bone, y_bone]]
    palatal_line: [[x_tooth, y_tooth], [x_bone, y_bone]]
    labial_mm:    float — pre-computed mm value
    palatal_mm:   float — pre-computed mm value
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
EXPECTED_OFFSETS = [f"{round(off * 0.1, 1):.1f}" for off in range(51)]

def _make_masks(h: int = 128, w: int = 128):
    tooth   = np.zeros((h, w), dtype=np.uint8)
    labial  = np.zeros((h, w), dtype=np.uint8)
    palatal = np.zeros((h, w), dtype=np.uint8)
    tooth[:, 56:72]  = 1   
    labial[:, 74:89] = 1   
    palatal[:, 40:55] = 1  
    return [tooth, labial, palatal]

def _make_axis(h: int = 128):
    tip  = np.array([64.0, 10.0], dtype=np.float32)
    apex = np.array([64.0, float(h - 10)], dtype=np.float32)
    vec  = apex - tip
    u1_unit = vec / np.linalg.norm(vec)
    u1_perp = _get_u1_perp(u1_unit)
    # Synthetic crest points slightly above the tip
    lc = tip + 15.0 * u1_unit
    pc = tip + 20.0 * u1_unit
    return tip, apex, lc, pc, u1_unit, u1_perp

def test_global_min_returns_dict_of_offsets():
    masks = _make_masks()
    tip, apex, lc, pc, u1_unit, u1_perp = _make_axis()
    result = calculate_global_minimum(tip, apex, lc, pc, u1_unit, u1_perp, masks, MM_PER_PIXEL)
    
    assert isinstance(result, dict)
    assert set(result.keys()) == set(EXPECTED_OFFSETS)
    for off in EXPECTED_OFFSETS:
        assert set(result[off].keys()) == EXPECTED_KEYS

def test_global_min_lines_are_two_xy_pairs():
    masks = _make_masks()
    tip, apex, lc, pc, u1_unit, u1_perp = _make_axis()
    result = calculate_global_minimum(tip, apex, lc, pc, u1_unit, u1_perp, masks, MM_PER_PIXEL)
    
    for off in EXPECTED_OFFSETS:
        res = result[off]
        for key in ("labial_line", "palatal_line"):
            seg = res[key]
            assert isinstance(seg, list) and len(seg) == 2
            assert all(isinstance(pt, list) and len(pt) == 2 for pt in seg)

def test_global_min_coords_finite():
    masks = _make_masks()
    tip, apex, lc, pc, u1_unit, u1_perp = _make_axis()
    result = calculate_global_minimum(tip, apex, lc, pc, u1_unit, u1_perp, masks, MM_PER_PIXEL)
    
    for off in EXPECTED_OFFSETS:
        res = result[off]
        for key in ("labial_line", "palatal_line"):
            for pt in res[key]:
                for v in pt:
                    assert math.isfinite(v)
        assert math.isfinite(res["labial_mm"])
        assert math.isfinite(res["palatal_mm"])

def test_mm_values_match_pixel_distance():
    masks = _make_masks()
    tip, apex, lc, pc, u1_unit, u1_perp = _make_axis()
    result = calculate_global_minimum(tip, apex, lc, pc, u1_unit, u1_perp, masks, MM_PER_PIXEL)
    
    for off in EXPECTED_OFFSETS:
        res = result[off]
        for key, mm_key in [("labial_line", "labial_mm"), ("palatal_line", "palatal_mm")]:
            (x1, y1), (x2, y2) = res[key]
            px_dist = math.hypot(x2 - x1, y2 - y1)
            expected_mm = px_dist * MM_PER_PIXEL
            actual_mm = res[mm_key]
            assert abs(actual_mm - expected_mm) < 0.01

def test_global_minimum_does_not_mutate_inputs():
    masks = _make_masks()
    tip, apex, lc, pc, u1_unit, u1_perp = _make_axis()
    
    tip_before = tip.copy()
    sums_before = [m.sum() for m in masks]
    
    calculate_global_minimum(tip, apex, lc, pc, u1_unit, u1_perp, masks, MM_PER_PIXEL)
    
    np.testing.assert_array_equal(tip, tip_before)
    for i, (m, s) in enumerate(zip(masks, sums_before)):
        assert m.sum() == s

def test_global_min_monotonic_increase():
    """
    CRITICAL VECTOR MATH FIX TEST:
    As cervical_offset_mm increases, the search window strictly shrinks (moves apically).
    Therefore, the minimum distance in the shrinking subset MUST be >= the minimum in the superset.
    """
    # Create a mask where bone thickness increases apically
    h, w = 128, 128
    masks = _make_masks(h, w)
    
    # Taper labial bone so it gets thicker as y increases (moves apically)
    # y=10 is tip (coronal), y=118 is apex (apical)
    # At y=20 (near crest), bone starts at x=74 (gap=3)
    # At y=100 (apical), bone starts at x=80 (gap=9)
    for y in range(20, 110):
        masks[1][y, 74:89] = 0  # clear old bone
        start_x = 74 + int(6 * (y - 20) / 90)
        masks[1][y, start_x:89] = 1

    tip, apex, lc, pc, u1_unit, u1_perp = _make_axis(h)
    result = calculate_global_minimum(tip, apex, lc, pc, u1_unit, u1_perp, masks, MM_PER_PIXEL)
    
    labial_mms = [result[off]["labial_mm"] for off in EXPECTED_OFFSETS]
    
    # As offset increases, we search a smaller window (further down the root where gap is wider)
    # so the minimum gap found should increase monotonically
    for i in range(1, len(labial_mms)):
        assert labial_mms[i] >= labial_mms[i-1], (
            f"Offset math bug: gap decreased from {labial_mms[i-1]} to {labial_mms[i]} "
            f"when offset increased from {EXPECTED_OFFSETS[i-1]} to {EXPECTED_OFFSETS[i]}"
        )


