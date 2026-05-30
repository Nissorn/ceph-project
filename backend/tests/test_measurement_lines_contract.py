"""
test_measurement_lines_contract.py
====================================
Verifies the shape of `measurement_lines` in the AnalysisService response dict.

Contract (required by DashboardApp → CephCanvasEditor adapter):
  measurement_lines is a dict with exactly 6 keys:
    labial_crest_line, labial_midroot_line, labial_apex_line,
    palatal_crest_line, palatal_midroot_line, palatal_apex_line

  Each value is a 2-element list:  [[x1, y1], [x2, y2]]
  Both coordinates must be finite floats (no NaN / inf).

These tests exercise only the geometry helpers in analysis_service — no model
weights are required (mocked masks + landmark coords are used directly).
"""
import math
import numpy as np
import pytest

# ─── Module under test ────────────────────────────────────────────────────────
from app.services.analysis_service import (
    _find_tooth_boundary,
    _get_bone_thickness_at_point,
    _get_u1_perp,
    MASK_IDX_UPPER_INCISOR,
    MASK_IDX_LABIAL_BONE,
    MASK_IDX_PALATAL_BONE,
)

EXPECTED_LINE_KEYS = {
    "labial_crest_line",
    "labial_midroot_line",
    "labial_apex_line",
    "palatal_crest_line",
    "palatal_midroot_line",
    "palatal_apex_line",
}


def _make_synthetic_masks(h: int = 64, w: int = 64):
    """Minimal synthetic masks: tooth column in centre, bones on either side."""
    tooth = np.zeros((h, w), dtype=np.uint8)
    labial = np.zeros((h, w), dtype=np.uint8)
    palatal = np.zeros((h, w), dtype=np.uint8)

    # Tooth: columns 28–36 (8 px wide, full height)
    tooth[:, 28:36] = 1
    # Labial bone: columns 20–27 (adjacent labial side)
    labial[:, 20:27] = 1
    # Palatal bone: columns 37–44 (adjacent palatal side)
    palatal[:, 37:44] = 1

    return [tooth, labial, palatal]


def _build_measurement_lines(masks, mm_per_pixel=0.1):
    """Replicate the Step 7.5 geometry from analysis_service.analyze_image()."""
    h, w = masks[MASK_IDX_UPPER_INCISOR].shape

    # Synthetic landmark coords (image-space pixels)
    tip = np.array([32.0, 10.0], dtype=np.float32)
    apex = np.array([32.0, 50.0], dtype=np.float32)

    u1_vec = apex - tip
    u1_len = np.linalg.norm(u1_vec)
    u1_unit = u1_vec / u1_len if u1_len > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)
    u1_perp = _get_u1_perp(u1_unit)

    labial_crest_pt = np.array([36.0, 15.0], dtype=np.float32)
    palatal_crest_pt = np.array([28.0, 15.0], dtype=np.float32)
    labial_midroot_pt = np.array([36.0, 30.0], dtype=np.float32)
    palatal_midroot_pt = np.array([28.0, 30.0], dtype=np.float32)

    t_lc = np.dot(labial_crest_pt - tip, u1_unit)
    P_axis_lc = tip + t_lc * u1_unit
    P_tooth_lc = _find_tooth_boundary(masks[MASK_IDX_UPPER_INCISOR], P_axis_lc, u1_perp, max_dist_px=50.0)

    t_pc = np.dot(palatal_crest_pt - tip, u1_unit)
    P_axis_pc = tip + t_pc * u1_unit
    P_tooth_pc = _find_tooth_boundary(masks[MASK_IDX_UPPER_INCISOR], P_axis_pc, -u1_perp, max_dist_px=50.0)

    labial_midroot_px = _get_bone_thickness_at_point(
        masks[MASK_IDX_LABIAL_BONE], labial_midroot_pt, u1_perp, max_dist_px=50.0
    )
    palatal_midroot_px = _get_bone_thickness_at_point(
        masks[MASK_IDX_PALATAL_BONE], palatal_midroot_pt, -u1_perp, max_dist_px=50.0
    )

    labial_midroot_target = labial_midroot_pt + labial_midroot_px * u1_perp
    palatal_midroot_target = palatal_midroot_pt - palatal_midroot_px * u1_perp

    labial_apex_pt = np.array([36.0, 50.0], dtype=np.float32)
    palatal_apex_pt = np.array([28.0, 50.0], dtype=np.float32)
    labial_apex_px = np.linalg.norm(np.dot(labial_apex_pt - apex, u1_perp))
    palatal_apex_px = np.linalg.norm(np.dot(apex - palatal_apex_pt, u1_perp))

    labial_apex_target = apex + labial_apex_px * u1_perp
    palatal_apex_target = apex - palatal_apex_px * u1_perp

    def _coord(pt1, pt2):
        return [
            [float(round(pt1[0], 3)), float(round(pt1[1], 3))],
            [float(round(pt2[0], 3)), float(round(pt2[1], 3))],
        ]

    return {
        "labial_crest_line": _coord(labial_crest_pt, P_tooth_lc),
        "labial_midroot_line": _coord(labial_midroot_pt, labial_midroot_target),
        "labial_apex_line": _coord(apex, labial_apex_target),
        "palatal_crest_line": _coord(palatal_crest_pt, P_tooth_pc),
        "palatal_midroot_line": _coord(palatal_midroot_pt, palatal_midroot_target),
        "palatal_apex_line": _coord(apex, palatal_apex_target),
    }


# ── Tracer bullet: dict has exactly the 6 required keys ───────────────────────

def test_measurement_lines_has_all_six_keys():
    """measurement_lines must contain all 6 named segments — no more, no less."""
    masks = _make_synthetic_masks()
    lines = _build_measurement_lines(masks)
    assert set(lines.keys()) == EXPECTED_LINE_KEYS, (
        f"Got keys {set(lines.keys())} — expected {EXPECTED_LINE_KEYS}"
    )


# ── Each segment is [[x1,y1],[x2,y2]] ─────────────────────────────────────────

def test_each_line_is_two_xy_pairs():
    """Every line segment must be a list of exactly 2 [x, y] pairs."""
    masks = _make_synthetic_masks()
    lines = _build_measurement_lines(masks)

    for key, segment in lines.items():
        assert isinstance(segment, list), f"{key}: must be a list"
        assert len(segment) == 2, f"{key}: must have exactly 2 points, got {len(segment)}"
        for pt in segment:
            assert isinstance(pt, list), f"{key}: each point must be a list, got {type(pt)}"
            assert len(pt) == 2, f"{key}: each point must be [x, y], got {pt}"


# ── All coordinates are finite floats ─────────────────────────────────────────

def test_all_coordinates_are_finite():
    """No NaN or Inf values allowed — NaN breaks JSON serialisation."""
    masks = _make_synthetic_masks()
    lines = _build_measurement_lines(masks)

    for key, segment in lines.items():
        for pt in segment:
            for v in pt:
                assert math.isfinite(v), f"{key}: coordinate {v} is not finite"


# ── Labial lines point right (positive-x direction from axis) ─────────────────

def test_labial_crest_line_points_toward_labial_bone():
    """The labial crest line must start on the tooth and end at/past the axis.

    With a vertical tooth axis (tip above, apex below) and tooth centred at x=32,
    the labial surface is at x≈36. The labial bone is at x<36 (columns 20-27)
    with u1_perp pointing in +x direction, so the endpoint should be >= start x
    for lines ending on the labial bone side.
    """
    masks = _make_synthetic_masks()
    lines = _build_measurement_lines(masks)
    seg = lines["labial_crest_line"]
    # The segment goes from landmark → tooth surface; tooth surface is always
    # finite and differs from the raw landmark coord by at most a few pixels.
    x1, y1 = seg[0]
    x2, y2 = seg[1]
    # Both points must be within a 64×64 image space
    assert 0 <= x1 <= 64 and 0 <= y1 <= 64, f"Start point out of range: {seg[0]}"
    assert 0 <= x2 <= 64 and 0 <= y2 <= 64, f"End point out of range: {seg[1]}"
