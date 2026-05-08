"""
Phase 3 — Medical Logic Engine (Biomechanics)
==============================================
Implements U1-PP angle calculation and treatment-biomechanics classification
based on Zhang et al. 2021 for upper central incisor root position planning.

Landmark keys (10 total, hardcoded per GEMINI.md):
    'Upper_tip'       — incisal tip of upper central incisor
    'Upper_apex'      — root apex of upper central incisor
    'ANS'             — Anterior Nasal Spine
    'PNS'             — Posterior Nasal Spine
    'LB'              — Labial bone landmark
    'PB'              — Palatal bone landmark
    'Labial_crest'    — Labial alveolar crest
    'Palatal_crest'   — Palatal alveolar crest
    'Labial_midroot' — Labial midroot
    'Palatal_midroot'— Palatal midroot

Usage:
    python src/phase3/biomechanics.py
"""

import math
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANDMARK_KEYS = [
    "Upper_tip",
    "Upper_apex",
    "ANS",
    "PNS",
    "LB",
    "PB",
    "Palatal_crest",
    "Labial_crest",
    "Labial_midroot",
    "Palatal_midroot",
]

# Zhang et al. 2021 — angle zone boundaries (degrees)
ANGLE_LOW = 105.0
ANGLE_HIGH = 115.0

# Root apex position difference threshold (mm)
# If |LB_dist - PB_dist| < this, the apex is considered "Midway"
POSITION_THRESHOLD_MM = 0.2

REQUIRED_METRIC_KEYS = [
    "u1_pp_angle_deg",
    "lb_apex_dist_mm",
    "pb_apex_dist_mm",
]

REQUIRED_CLASSIFICATION_KEYS = [
    "Root apex position",
    "Incisor condition",
    "Preferred biomechanics",
    "Biomechanics to avoid",
    "Clinical implication",
]

# ---------------------------------------------------------------------------
# Classification lookup table — Zhang et al. 2021
# Rows: root apex position (Labial / Midway / Palatal)
# Cols: U1-PP angle zone  (<105 / 105-115 / >115)
# ---------------------------------------------------------------------------

_CLASSIFICATION_TABLE: Dict[str, Dict[str, Dict[str, str]]] = {
    "Labial": {
        "<105": {
            "Incisor condition": "Retroclined incisor with apex near labial bone",
            "Preferred biomechanics": "Light controlled tipping with torque control",
            "Biomechanics to avoid": "Uncontrolled proclination, labial root torque",
            "Clinical implication": "Uprighting is possible but labial cortical bone must be preserved",
        },
        "105-115": {
            "Incisor condition": "Normal inclination with apex close to labial plate",
            "Preferred biomechanics": "Light controlled tipping or torque maintenance",
            "Biomechanics to avoid": "Bodily movement forward, uncontrolled tipping",
            "Clinical implication": "Avoid further labial displacement of the apex",
        },
        ">115": {
            "Incisor condition": "Proclined incisor with apex near labial bone",
            "Preferred biomechanics": "Controlled tipping during retraction with strict torque control",
            "Biomechanics to avoid": "Uncontrolled tipping, labial root torque",
            "Clinical implication": "High risk; strict torque control is required",
        },
    },
    "Midway": {
        "<105": {
            "Incisor condition": "Retroclined incisor with apex centrally located",
            "Preferred biomechanics": "Controlled proclination or bodily movement if bone allows",
            "Biomechanics to avoid": "Uncontrolled tipping",
            "Clinical implication": "Favorable prognosis",
        },
        "105-115": {
            "Incisor condition": "Normal inclination with centered apex",
            "Preferred biomechanics": "Bodily movement (translation)",
            "Biomechanics to avoid": "Uncontrolled tipping",
            "Clinical implication": "Most favorable condition",
        },
        ">115": {
            "Incisor condition": "Proclined incisor with centered apex",
            "Preferred biomechanics": "Controlled tipping with torque control during retraction",
            "Biomechanics to avoid": "Uncontrolled tipping",
            "Clinical implication": "Safe if torque is well controlled",
        },
    },
    "Palatal": {
        "<105": {
            "Incisor condition": "Retroclined incisor with apex near palatal bone",
            "Preferred biomechanics": "Careful movement; labial crown/root control may be required",
            "Biomechanics to avoid": "Palatal root torque, further retroclination",
            "Clinical implication": "Risk of palatal cortical perforation",
        },
        "105-115": {
            "Incisor condition": "Normal inclination with apex near palatal plate",
            "Preferred biomechanics": "Bodily movement with caution",
            "Biomechanics to avoid": "Excessive palatal root torque",
            "Clinical implication": "Monitor palatal bone limits",
        },
        ">115": {
            "Incisor condition": "Proclined incisor with apex near palatal bone",
            "Preferred biomechanics": "Controlled tipping during retraction with apex control",
            "Biomechanics to avoid": "Retraction causing further palatal displacement of apex",
            "Clinical implication": "Retraction possible but avoid excessive palatal pressure",
        },
    },
}


# ---------------------------------------------------------------------------
# Mock landmark generator
# ---------------------------------------------------------------------------

def mock_landmarks() -> Dict[str, Tuple[float, float]]:
    """Return a realistic set of mock landmark pixel coordinates.

    Coordinates are (x, y) in image-pixel space, consistent with a
    standard cephalometric radiograph orientation:
      - x increases to the right (labial → palatal roughly ≈ anterior → posterior)
      - y increases downward

    Returns
    -------
    dict mapping each of the 10 landmark keys to an (x, y) tuple.
    """
    return {
        # Upper incisor — tip is lower (higher y) and more labial (lower x)
        # than apex; tooth angles ~ 110° relative to palatal plane
        "Upper_tip":        (310.0, 480.0),
        "Upper_apex":       (295.0, 340.0),
        # Palatal plane — ANS is anterior (left), PNS is posterior (right)
        "ANS":              (250.0, 410.0),
        "PNS":              (520.0, 415.0),
        # LB and PB flank the root apex (apex at ~295, 340)
        # LB is slightly labial to the apex → small positive Δx
        "LB":               (302.0, 340.0),   # 7 px labial to apex
        "PB":               (280.0, 340.0),   # 15 px palatal to apex
        # Crests and mid-root — adjacent to the upper incisor socket
        "Labial_crest":     (316.0, 395.0),
        "Palatal_crest":    (278.0, 390.0),
        "Labial_midroot":  (308.0, 415.0),
        "Palatal_midroot": (284.0, 412.0),
    }


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def _vec2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Return the 2-D vector from p1 to p2."""
    return (p2[0] - p1[0], p2[1] - p1[1])


def _dot2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _norm2d(v: Tuple[float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def _euclidean_px(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_metrics(
    landmarks: Dict[str, Tuple[float, float]],
    mm_per_pixel: float = 0.0984,
) -> Dict[str, float]:
    """Calculate biomechanical metrics from landmark pixel coordinates.

    Parameters
    ----------
    landmarks : dict
        Mapping of landmark name → (x, y) pixel coordinates.
        Must contain at minimum: 'Upper-tip', 'Upper-apex', 'ANS', 'PNS',
        'LB', 'PB'.
    mm_per_pixel : float
        Calibration factor for this image (mm per pixel).  Default matches
        the dataset mean (0.0984 mm/px from calibration.csv).

    Returns
    -------
    dict with keys:
        u1_pp_angle_deg  — U1-to-palatal-plane angle in degrees
        lb_apex_dist_mm  — distance from LB to root apex in mm
        pb_apex_dist_mm  — distance from PB to root apex in mm
    """
    _required = ("Upper_tip", "Upper_apex", "ANS", "PNS", "LB", "PB")
    missing = [k for k in _required if k not in landmarks]
    if missing:
        raise ValueError(f"Missing required landmarks: {missing}")

    tip   = landmarks["Upper_tip"]
    apex  = landmarks["Upper_apex"]
    ans   = landmarks["ANS"]
    pns   = landmarks["PNS"]
    lb    = landmarks["LB"]
    pb    = landmarks["PB"]

    # ── U1-PP angle ──────────────────────────────────────────────────────────
    # Long axis of U1: from apex toward tip (apex → tip)
    u1_vec = _vec2d(apex, tip)
    # Palatal plane: from ANS toward PNS (ANS → PNS)
    pp_vec = _vec2d(ans, pns)

    denom = _norm2d(u1_vec) * _norm2d(pp_vec)
    if denom == 0.0:
        raise ValueError(
            "Cannot compute U1-PP angle: zero-length vector detected. "
            "Check that 'Upper_tip' != 'Upper_apex' and 'ANS' != 'PNS'."
        )
    cos_theta = max(-1.0, min(1.0, _dot2d(u1_vec, pp_vec) / denom))
    raw_angle = math.degrees(math.acos(cos_theta))
    # Cephalometric U1-PP is typically the postero-inferior angle (~110 deg for normal)
    # If the vector math yields an acute angle, we take the supplementary angle.
    u1_pp_angle_deg = 180.0 - raw_angle if raw_angle < 90.0 else raw_angle

    # ── LB-Apex and PB-Apex distances ────────────────────────────────────────
    lb_apex_px = _euclidean_px(lb, apex)
    pb_apex_px = _euclidean_px(pb, apex)

    lb_apex_mm = lb_apex_px * mm_per_pixel
    pb_apex_mm = pb_apex_px * mm_per_pixel

    return {
        "u1_pp_angle_deg": u1_pp_angle_deg,
        "lb_apex_dist_mm": lb_apex_mm,
        "pb_apex_dist_mm": pb_apex_mm,
    }


# ---------------------------------------------------------------------------
# Treatment classification
# ---------------------------------------------------------------------------

def _get_angle_zone(u1_pp_angle: float) -> str:
    """Map U1-PP angle to one of three Zhang et al. 2021 zones."""
    if u1_pp_angle < ANGLE_LOW:
        return "<105"
    elif u1_pp_angle <= ANGLE_HIGH:
        return "105-115"
    else:
        return ">115"


def _get_apex_position(lb_apex_dist_mm: float, pb_apex_dist_mm: float) -> str:
    """Classify root apex as Labial, Midway, or Palatal.

    Logic:
        diff = LB_dist - PB_dist
        If  diff > +POSITION_THRESHOLD_MM  → apex is closer to PB → Palatal
        If  diff < -POSITION_THRESHOLD_MM  → apex is closer to LB → Labial
        Otherwise                          → Midway
    """
    diff = lb_apex_dist_mm - pb_apex_dist_mm
    if diff > POSITION_THRESHOLD_MM:
        return "Palatal"
    elif diff < -POSITION_THRESHOLD_MM:
        return "Labial"
    else:
        return "Midway"


def classify_treatment(
    u1_pp_angle: float,
    lb_apex_dist: float,
    pb_apex_dist: float,
) -> Dict[str, str]:
    """Classify biomechanical treatment recommendation (Zhang et al. 2021).

    Parameters
    ----------
    u1_pp_angle : float
        U1-to-palatal-plane angle in degrees.
    lb_apex_dist : float
        Distance from LB landmark to root apex in mm.
    pb_apex_dist : float
        Distance from PB landmark to root apex in mm.

    Returns
    -------
    dict with exactly these keys:
        "Root apex position"      — "Labial" | "Midway" | "Palatal"
        "Incisor condition"       — descriptive string
        "Preferred biomechanics"  — recommended biomechanical approach
        "Biomechanics to avoid"   — contraindicated movements
        "Clinical implication"    — clinical narrative
    """
    apex_position = _get_apex_position(lb_apex_dist, pb_apex_dist)
    angle_zone    = _get_angle_zone(u1_pp_angle)

    entry = _CLASSIFICATION_TABLE[apex_position][angle_zone]

    return {
        "Root apex position":     apex_position,
        "Incisor condition":      entry["Incisor condition"],
        "Preferred biomechanics": entry["Preferred biomechanics"],
        "Biomechanics to avoid":  entry["Biomechanics to avoid"],
        "Clinical implication":   entry["Clinical implication"],
    }


# ---------------------------------------------------------------------------
# Built-in tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Phase 3 — Biomechanics Engine  |  Built-in Self-Test")
    print("=" * 70)

    # ── Step 1: Generate mock landmarks ──────────────────────────────────────
    lm = mock_landmarks()
    print("\n[1] Mock landmarks loaded:")
    for name, (x, y) in lm.items():
        print(f"    {name:<22} ({x:7.1f}, {y:7.1f}) px")

    # ── Step 2: Calculate metrics ─────────────────────────────────────────────
    MM_PER_PX = 0.0984          # dataset mean from calibration.csv
    metrics = calculate_metrics(lm, mm_per_pixel=MM_PER_PX)

    print(f"\n[2] Calculated metrics  (mm_per_pixel = {MM_PER_PX}):")
    print(f"    U1-PP angle       : {metrics['u1_pp_angle_deg']:7.2f} °")
    print(f"    LB-Apex distance  : {metrics['lb_apex_dist_mm']:7.3f} mm")
    print(f"    PB-Apex distance  : {metrics['pb_apex_dist_mm']:7.3f} mm")

    # ── Step 3: Classify treatment ────────────────────────────────────────────
    result = classify_treatment(
        u1_pp_angle  = metrics["u1_pp_angle_deg"],
        lb_apex_dist = metrics["lb_apex_dist_mm"],
        pb_apex_dist = metrics["pb_apex_dist_mm"],
    )

    print("\n[3] Treatment classification:")
    for key, value in result.items():
        # Wrap long values for readability
        if len(value) > 55:
            print(f"    {key}:")
            words = value.split(" ")
            line, col = "        ", 8
            for w in words:
                if col + len(w) + 1 > 72:
                    print(line)
                    line, col = "        " + w + " ", 8 + len(w) + 1
                else:
                    line += w + " "
                    col  += len(w) + 1
            if line.strip():
                print(line)
        else:
            print(f"    {key}: {value}")

    # ── Step 4: Assertions ────────────────────────────────────────────────────
    print("\n[4] Running assertions …")

    # All required metric keys present
    for k in REQUIRED_METRIC_KEYS:
        assert k in metrics, f"Missing metric key: '{k}'"

    # All required classification keys present
    for k in REQUIRED_CLASSIFICATION_KEYS:
        assert k in result, f"Missing classification key: '{k}'"

    # Root apex position is one of the expected values
    assert result["Root apex position"] in {"Labial", "Midway", "Palatal"}, (
        f"Unexpected apex position: {result['Root apex position']}"
    )

    # Angle is physically plausible (0–180°)
    assert 0.0 <= metrics["u1_pp_angle_deg"] <= 180.0, (
        f"U1-PP angle out of range: {metrics['u1_pp_angle_deg']}"
    )

    # Distances are non-negative
    assert metrics["lb_apex_dist_mm"] >= 0.0
    assert metrics["pb_apex_dist_mm"] >= 0.0

    # All classification values are non-empty strings
    for k, v in result.items():
        assert isinstance(v, str) and v.strip(), f"Empty or non-string value for '{k}'"

    print("    All assertions PASSED ✓")

    # ── Step 5: Boundary-condition tests ──────────────────────────────────────
    print("\n[5] Boundary-condition tests …")

    zones = [
        ("sub-105",   90.0,  "Labial"),
        ("band 110",  110.0, "Midway"),
        ("super-115", 120.0, "Palatal"),
    ]
    for label, angle, pos in zones:
        lb_d = 1.0
        pb_d = {"Labial": 1.5, "Midway": 1.05, "Palatal": 0.5}[pos]
        r = classify_treatment(angle, lb_d, pb_d)
        assert r["Root apex position"] == pos, (
            f"Boundary test '{label}': expected apex='{pos}', got '{r['Root apex position']}'"
        )
        print(f"    [{label}] angle={angle}°  apex={r['Root apex position']}  "
              f"→ {r['Preferred biomechanics'][:40]}…  ✓")

    print("\n" + "=" * 70)
    print("  All tests passed — biomechanics.py is working correctly.")
    print("=" * 70)
