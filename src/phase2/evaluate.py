"""
Phase 2 — Landmark Detection Evaluation Metrics
================================================
Implements MRE (Mean Radial Error) and SDR (Successful Detection Rate)
as used in the cephalometric landmark detection literature.

Inputs
------
predictions : List[Dict[str, Tuple[float, float]]]
    One dict per image.  Key = landmark name, value = (x, y) in pixels.
ground_truths : List[Dict[str, Tuple[float, float]]]
    Same format as predictions.  Must have the same length.

Usage
-----
    python src/phase2/evaluate.py
"""

import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 10 project landmarks (hardcoded per GEMINI.md)
LANDMARK_KEYS: List[str] = [
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

DEFAULT_MM_PER_PIXEL: float = 0.0984   # dataset mean from calibration.csv
DEFAULT_SDR_THRESHOLDS: List[float] = [2.0, 2.5, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _radial_error_px(
    pred: Tuple[float, float],
    gt: Tuple[float, float],
) -> float:
    """Euclidean distance in pixels between a predicted and GT landmark."""
    dx = pred[0] - gt[0]
    dy = pred[1] - gt[1]
    return math.sqrt(dx * dx + dy * dy)


def _collect_distances(
    predictions: List[Dict[str, Tuple[float, float]]],
    ground_truths: List[Dict[str, Tuple[float, float]]],
    mm_per_pixel: float,
) -> List[float]:
    """Return a flat list of per-landmark distances (in mm) across all images.

    Only landmarks that appear in BOTH the prediction dict and the ground-truth
    dict for a given image are included.  Missing landmarks are silently skipped
    rather than raising an error, which makes the metrics robust to partially-
    annotated images.

    Raises
    ------
    ValueError
        If predictions and ground_truths have different lengths.
    ZeroDivisionError
        (Impossible here, but we guard mm_per_pixel > 0 to be safe.)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions has {len(predictions)} entries but "
            f"ground_truths has {len(ground_truths)} — lengths must match."
        )
    if mm_per_pixel <= 0.0:
        raise ValueError(f"mm_per_pixel must be positive, got {mm_per_pixel}.")

    distances: List[float] = []
    for pred_img, gt_img in zip(predictions, ground_truths):
        common_keys = set(pred_img.keys()) & set(gt_img.keys())
        for key in sorted(common_keys):           # sorted for determinism
            dist_px = _radial_error_px(pred_img[key], gt_img[key])
            distances.append(dist_px * mm_per_pixel)
    return distances


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_mre(
    predictions: List[Dict[str, Tuple[float, float]]],
    ground_truths: List[Dict[str, Tuple[float, float]]],
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
) -> float:
    """Compute Mean Radial Error (MRE) in millimetres.

    Parameters
    ----------
    predictions : list of dicts
        One dict per image; maps landmark name → (x, y) pixel coordinate.
    ground_truths : list of dicts
        Same structure as predictions.
    mm_per_pixel : float
        Calibration factor (mm per pixel) for the image set.  Use the
        per-image value from calibration.csv when images differ.

    Returns
    -------
    float
        MRE in mm (average Euclidean error across all landmarks and images).

    Raises
    ------
    ValueError
        If no common landmark pairs are found (nothing to average).
    """
    distances = _collect_distances(predictions, ground_truths, mm_per_pixel)
    if not distances:
        raise ValueError(
            "No common landmarks found between predictions and ground_truths. "
            "Cannot compute MRE."
        )
    return sum(distances) / len(distances)


def calculate_sdr(
    predictions: List[Dict[str, Tuple[float, float]]],
    ground_truths: List[Dict[str, Tuple[float, float]]],
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
    thresholds: Optional[List[float]] = None,
) -> Dict[float, float]:
    """Compute Successful Detection Rate (SDR) for multiple distance thresholds.

    A landmark is considered successfully detected if its radial error is
    strictly less than or equal to the threshold (≤ threshold_mm).

    Parameters
    ----------
    predictions : list of dicts
        One dict per image; maps landmark name → (x, y) pixel coordinate.
    ground_truths : list of dicts
        Same structure as predictions.
    mm_per_pixel : float
        Calibration factor (mm per pixel).
    thresholds : list of float, optional
        Detection thresholds in mm.  Defaults to [2.0, 2.5, 3.0, 4.0].

    Returns
    -------
    dict
        Maps each threshold (float) → SDR percentage (0.0–100.0).
        Example: {2.0: 85.5, 2.5: 92.0, 3.0: 96.0, 4.0: 100.0}

    Raises
    ------
    ValueError
        If no common landmark pairs are found.
    """
    if thresholds is None:
        thresholds = DEFAULT_SDR_THRESHOLDS

    distances = _collect_distances(predictions, ground_truths, mm_per_pixel)
    if not distances:
        raise ValueError(
            "No common landmarks found between predictions and ground_truths. "
            "Cannot compute SDR."
        )

    n = len(distances)
    sdr: Dict[float, float] = {}
    for thr in thresholds:
        count_within = sum(1 for d in distances if d <= thr)
        sdr[thr] = round(100.0 * count_within / n, 2)
    return sdr


# ---------------------------------------------------------------------------
# Built-in tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Phase 2 — Evaluation Metrics  |  Built-in Self-Test")
    print("=" * 70)

    # ── Step 1: Build mock ground-truth and predictions ───────────────────────
    # Ground-truth: two images, all 10 landmarks with plausible pixel coords.
    # We keep both images identical here for simplicity; the metrics code
    # handles arbitrary per-image coordinates.

    _GT_IMG1: Dict[str, Tuple[float, float]] = {
        "Upper_tip":        (310.0, 480.0),
        "Upper_apex":       (295.0, 340.0),
        "ANS":              (250.0, 410.0),
        "PNS":              (520.0, 415.0),
        "LB":               (302.0, 340.0),
        "PB":               (280.0, 340.0),
        "Palatal_crest":    (278.0, 390.0),
        "Labial_crest":     (316.0, 395.0),
        "Labial_midroot":  (308.0, 415.0),
        "Palatal_midroot": (284.0, 412.0),
    }
    _GT_IMG2: Dict[str, Tuple[float, float]] = {
        "Upper_tip":        (308.0, 482.0),
        "Upper_apex":       (293.0, 338.0),
        "ANS":              (248.0, 412.0),
        "PNS":              (518.0, 417.0),
        "LB":               (300.0, 338.0),
        "PB":               (278.0, 338.0),
        "Palatal_crest":    (276.0, 392.0),
        "Labial_crest":     (314.0, 397.0),
        "Labial_midroot":  (306.0, 417.0),
        "Palatal_midroot": (282.0, 414.0),
    }
    ground_truths: List[Dict[str, Tuple[float, float]]] = [_GT_IMG1, _GT_IMG2]

    # Predictions: simulate AI output by adding a fixed pixel offset.
    # Using (+6, +8) → Euclidean offset = 10 px → ~0.984 mm at 0.0984 mm/px.
    OFFSET_X, OFFSET_Y = 6.0, 8.0
    predictions: List[Dict[str, Tuple[float, float]]] = [
        {k: (x + OFFSET_X, y + OFFSET_Y) for k, (x, y) in img.items()}
        for img in ground_truths
    ]

    print(f"\n[1] Mock data: {len(ground_truths)} images × {len(LANDMARK_KEYS)} landmarks")
    print(f"    Simulated offset: ({OFFSET_X}, {OFFSET_Y}) px  "
          f"→ {math.sqrt(OFFSET_X**2 + OFFSET_Y**2):.4f} px Euclidean  "
          f"→ {math.sqrt(OFFSET_X**2 + OFFSET_Y**2) * DEFAULT_MM_PER_PIXEL:.4f} mm expected MRE")

    # ── Step 2: Calculate MRE ─────────────────────────────────────────────────
    mre = calculate_mre(predictions, ground_truths, mm_per_pixel=DEFAULT_MM_PER_PIXEL)
    print(f"\n[2] MRE: {mre:.4f} mm")

    # ── Step 3: Calculate SDR ─────────────────────────────────────────────────
    sdr = calculate_sdr(predictions, ground_truths, mm_per_pixel=DEFAULT_MM_PER_PIXEL)
    print("\n[3] SDR (Successful Detection Rate):")
    for thr in sorted(sdr.keys()):
        bar_len = int(sdr[thr] / 2)            # 50-char max bar
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"    ≤ {thr:.1f} mm  [{bar}]  {sdr[thr]:6.2f} %")

    # ── Step 4: Assertions ─────────────────────────────────────────────────────
    print("\n[4] Running assertions …")

    # MRE must be a positive float
    assert isinstance(mre, float),  f"MRE type error: got {type(mre)}"
    assert mre > 0.0,               f"MRE must be positive, got {mre}"

    # MRE should be close to the expected theoretical value
    expected_mre = math.sqrt(OFFSET_X**2 + OFFSET_Y**2) * DEFAULT_MM_PER_PIXEL
    assert abs(mre - expected_mre) < 1e-9, (
        f"MRE {mre:.6f} differs from expected {expected_mre:.6f}"
    )
    print(f"    MRE value check passed  (expected ≈ {expected_mre:.4f} mm)  ✓")

    # SDR must have all 4 default threshold keys
    for thr in DEFAULT_SDR_THRESHOLDS:
        assert thr in sdr, f"Missing SDR threshold key: {thr}"
    print(f"    SDR has all {len(DEFAULT_SDR_THRESHOLDS)} threshold keys  ✓")

    # SDR values must be 0–100
    for thr, pct in sdr.items():
        assert 0.0 <= pct <= 100.0, f"SDR[{thr}] out of range: {pct}"
    print("    All SDR values in [0, 100]  ✓")

    # Because all predictions have the same fixed offset (< 2.0 mm at 0.0984 mm/px),
    # SDR at 2.0 mm should be 100 %.
    assert sdr[2.0] == 100.0, f"Expected SDR[2.0]=100.0, got {sdr[2.0]}"
    print("    SDR[2.0 mm] == 100.0 %  ✓  (fixed 0.984 mm offset < 2.0 mm threshold)")

    # SDR values are non-decreasing as threshold increases
    sorted_thrs = sorted(sdr.keys())
    for i in range(len(sorted_thrs) - 1):
        assert sdr[sorted_thrs[i]] <= sdr[sorted_thrs[i + 1]], (
            f"SDR is not monotone: SDR[{sorted_thrs[i]}]={sdr[sorted_thrs[i]]} "
            f"> SDR[{sorted_thrs[i+1]}]={sdr[sorted_thrs[i+1]]}"
        )
    print("    SDR is monotonically non-decreasing  ✓")

    # ── Step 5: Edge-case — missing landmarks ─────────────────────────────────
    print("\n[5] Edge-case: partial overlap (one landmark missing from predictions) …")
    partial_pred = [{"Upper_tip": (311.0, 481.0)}]   # only 1 of 10 landmarks
    partial_gt   = [{"Upper_tip": (310.0, 480.0), "ANS": (250.0, 410.0)}]
    mre_partial  = calculate_mre(partial_pred, partial_gt)
    sdr_partial  = calculate_sdr(partial_pred, partial_gt)
    assert mre_partial > 0.0
    assert isinstance(sdr_partial, dict)
    print(f"    Partial MRE = {mre_partial:.4f} mm  (only 'Upper_tip' matched)  ✓")

    print("\n" + "=" * 70)
    print("  All tests passed — evaluate.py is working correctly.")
    print("=" * 70)
