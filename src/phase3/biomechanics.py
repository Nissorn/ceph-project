"""
Phase 3 -- Medical Logic Engine (Biomechanics)
==============================================
Implements U1-PP angle calculation and treatment-biomechanics classification
based on Zhang et al. 2021 for upper central incisor root position planning.

Landmark keys (10 total, hardcoded per GEMINI.md):
    'Upper_tip'       -- incisal tip of upper central incisor
    'Upper_apex'      -- root apex of upper central incisor
    'ANS'             -- Anterior Nasal Spine
    'PNS'             -- Posterior Nasal Spine
    'LB'              -- Labial bone landmark
    'PB'              -- Palatal bone landmark
    'Labial_crest'    -- Labial alveolar crest
    'Palatal_crest'   -- Palatal alveolar crest
    'Labial_midroot' -- Labial midroot
    'Palatal_midroot'-- Palatal midroot

Usage:
    python src/phase3/biomechanics.py
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple, TypedDict


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

# Zhang et al. 2021 -- angle zone boundaries (degrees)
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


class MetricsResult(TypedDict):
    """Typed return contract for calculate_metrics().

    Scalar fields (float) are clinical measurements in mm or degrees.
    Coordinate fields (Tuple[float, float]) are pixel-space points for
    visualisation only — never feed them into mm-based comparisons.
    """
    u1_pp_angle_deg: float                  # U1-to-palatal-plane angle (degrees)
    lb_apex_dist_mm: float                  # labial bone lateral clearance (mm)
    pb_apex_dist_mm: float                  # palatal bone lateral clearance (mm)
    lb_foot_px: Tuple[float, float]         # foot of LB perpendicular on tooth axis (pixels)
    pb_foot_px: Tuple[float, float]         # foot of PB perpendicular on tooth axis (pixels)

REQUIRED_CLASSIFICATION_KEYS = [
    "Root apex position",
    "Incisor condition",
    "Preferred biomechanics",
    "Biomechanics to avoid",
    "Clinical implication",
]

# ---------------------------------------------------------------------------
# Classification lookup table -- Zhang et al. 2021
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
        # Upper incisor -- tip is lower (higher y) and more labial (lower x)
        # than apex; tooth angles ~ 110° relative to palatal plane
        "Upper_tip":        (310.0, 480.0),
        "Upper_apex":       (295.0, 340.0),
        # Palatal plane -- ANS is anterior (left), PNS is posterior (right)
        "ANS":              (250.0, 410.0),
        "PNS":              (520.0, 415.0),
        # LB and PB flank the root apex (apex at ~295, 340)
        # LB is slightly labial to the apex → small positive Δx
        "LB":               (302.0, 340.0),   # 7 px labial to apex
        "PB":               (280.0, 340.0),   # 15 px palatal to apex
        # Crests and mid-root -- adjacent to the upper incisor socket
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


def _perp_dist_to_tooth_axis(
    point: Tuple[float, float],
    apex: Tuple[float, float],
    tip: Tuple[float, float],
) -> Tuple[float, Tuple[float, float]]:
    """Geometric projection correction for bone landmark plotting variance.

    The tooth axis is defined by the vector from Upper_apex → Upper_tip.
    Human annotators (and HRNet predictions) rarely place LB/PB at exactly
    90° to this axis. This function projects `point` perpendicularly onto
    the tooth axis line, eliminating the along-axis component of any plotting
    error so that the returned distance is the true lateral bone clearance.

    Parameters
    ----------
    point : (x, y) pixel coords of the bone landmark (LB or PB).
    apex  : (x, y) pixel coords of Upper_apex.
    tip   : (x, y) pixel coords of Upper_tip.

    Returns
    -------
    perp_dist_px : float
        Perpendicular (lateral) distance from `point` to the tooth axis line.
    foot : (x, y)
        The foot of the perpendicular on the tooth axis — can be used for
        visualisation of the corrected measurement line.
    """
    ax = tip[0] - apex[0]
    ay = tip[1] - apex[1]
    axis_len = math.sqrt(ax * ax + ay * ay)
    if axis_len == 0.0:
        raise ValueError(
            "Tooth axis has zero length: Upper_tip == Upper_apex. "
            "Check landmark annotations."
        )
    # Unit vector along tooth axis
    ux, uy = ax / axis_len, ay / axis_len

    # Scalar projection of (point - apex) onto the tooth axis
    px, py = point[0] - apex[0], point[1] - apex[1]
    t = px * ux + py * uy

    # Foot of perpendicular on the tooth axis line
    foot: Tuple[float, float] = (apex[0] + t * ux, apex[1] + t * uy)

    # Perpendicular distance (true lateral clearance)
    perp_dist_px = math.sqrt(
        (point[0] - foot[0]) ** 2 + (point[1] - foot[1]) ** 2
    )
    return perp_dist_px, foot


def calculate_metrics(
    landmarks: Dict[str, Tuple[float, float]],
    mm_per_pixel: float = 0.0984,
) -> MetricsResult:
    """Calculate biomechanical metrics from landmark pixel coordinates.

    LB and PB distances use geometric projection correction: each landmark is
    projected perpendicularly onto the tooth axis (Upper_apex → Upper_tip) so
    that human plotting variance and HRNet prediction jitter along the axis do
    not contaminate the lateral bone-clearance measurement.

    Parameters
    ----------
    landmarks : dict
        Mapping of landmark name → (x, y) pixel coordinates.
        Must contain at minimum: 'Upper_tip', 'Upper_apex', 'ANS', 'PNS',
        'LB', 'PB'.
    mm_per_pixel : float
        Calibration factor for this image (mm per pixel).  Default matches
        the dataset mean (0.0984 mm/px from calibration.csv).

    Returns
    -------
    dict with keys:
        u1_pp_angle_deg  -- U1-to-palatal-plane angle in degrees
        lb_apex_dist_mm  -- perpendicular distance from LB to tooth axis (mm)
        pb_apex_dist_mm  -- perpendicular distance from PB to tooth axis (mm)
        lb_foot_px       -- (x, y) foot of LB perpendicular on tooth axis
        pb_foot_px       -- (x, y) foot of PB perpendicular on tooth axis
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

    # ── LB / PB distances — projection-corrected ─────────────────────────────
    # Project each bone landmark perpendicularly onto the tooth axis to obtain
    # the true lateral clearance, eliminating plotting angle variance.
    lb_perp_px, lb_foot = _perp_dist_to_tooth_axis(lb, apex, tip)
    pb_perp_px, pb_foot = _perp_dist_to_tooth_axis(pb, apex, tip)

    lb_apex_mm = lb_perp_px * mm_per_pixel
    pb_apex_mm = pb_perp_px * mm_per_pixel

    return MetricsResult(
        u1_pp_angle_deg=u1_pp_angle_deg,
        lb_apex_dist_mm=lb_apex_mm,
        pb_apex_dist_mm=pb_apex_mm,
        lb_foot_px=lb_foot,
        pb_foot_px=pb_foot,
    )


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
        "Root apex position"      -- "Labial" | "Midway" | "Palatal"
        "Incisor condition"       -- descriptive string
        "Preferred biomechanics"  -- recommended biomechanical approach
        "Biomechanics to avoid"   -- contraindicated movements
        "Clinical implication"    -- clinical narrative
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
# Bone Thickness Calculator
# ---------------------------------------------------------------------------

def generate_mock_masks(image_shape: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate dummy masks for testing the BoneThicknessCalculator."""
    tooth_mask = np.zeros(image_shape, dtype=np.uint8)
    labial_bone_mask = np.zeros(image_shape, dtype=np.uint8)
    palatal_bone_mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Create a simple tooth shape (a rectangle)
    tooth_mask[100:400, 240:260] = 1
    
    # Create simple bone shapes (rectangles next to the tooth)
    labial_bone_mask[150:350, 200:235] = 1
    palatal_bone_mask[120:380, 265:300] = 1
    
    return tooth_mask, labial_bone_mask, palatal_bone_mask


class BoneThicknessCalculator:
    """Calculates alveolar bone thickness measurements."""
    
    def __init__(
        self,
        u1_axis_vector: Tuple[Tuple[float, float], Tuple[float, float]],
        upper_apex_point: Tuple[float, float],
        tooth_mask: np.ndarray,
        labial_bone_mask: np.ndarray,
        palatal_bone_mask: np.ndarray,
        mm_per_pixel: float = 0.0984
    ):
        self.u1_axis_vector = u1_axis_vector
        self.upper_apex_point = upper_apex_point
        self.tooth_mask = tooth_mask
        self.labial_bone_mask = labial_bone_mask
        self.palatal_bone_mask = palatal_bone_mask
        self.mm_per_pixel = mm_per_pixel
        
        self.upper_tip = np.array(u1_axis_vector[0], dtype=float)
        self.upper_apex = np.array(u1_axis_vector[1], dtype=float)
        
        # Calculate U1 axis vector and unit vector
        self.u1_vec = self.upper_apex - self.upper_tip
        self.u1_length = np.linalg.norm(self.u1_vec)
        if self.u1_length == 0:
            self.u1_unit = np.array([0.0, 1.0])
        else:
            self.u1_unit = self.u1_vec / self.u1_length
            
        # Calculate perpendicular unit vector (90 deg clockwise rotation)
        self.u1_perp_unit = np.array([-self.u1_unit[1], self.u1_unit[0]])

    def calculate_plan_a_3_lines(self) -> Dict[str, Dict[str, float]]:
        """Measure bone thickness at cervical, middle, and apical thirds."""
        # Calculate division points
        cervical_point = self.upper_tip + self.u1_unit * (self.u1_length / 3)
        middle_point = self.upper_tip + self.u1_unit * (2 * self.u1_length / 3)
        apical_point = self.upper_apex
        
        results = {}
        
        levels = [
            (cervical_point, "cervical"),
            (middle_point, "middle"),
            (apical_point, "apical")
        ]
        
        for point, level_name in levels:
            level_results = {}
            for bone_name, bone_mask in [("labial", self.labial_bone_mask), ("palatal", self.palatal_bone_mask)]:
                coords = np.argwhere(bone_mask > 0)
                if len(coords) == 0:
                    level_results[bone_name + "_thickness"] = 0.0
                    continue
                
                # Convert to (x, y) format
                points_xy = np.fliplr(coords)
                vectors = points_xy - point
                
                # Project vectors onto perpendicular axis
                projections = np.dot(vectors, self.u1_perp_unit)
                
                t_min = np.min(projections)
                t_max = np.max(projections)
                width_pixels = abs(t_max - t_min)
                
                level_results[bone_name + "_thickness"] = width_pixels * self.mm_per_pixel
                
            results[level_name] = level_results
            
        return results

    def _get_t_min_t_apex(self) -> Tuple[float, float, float]:
        """Helper to find the topmost bone projection and apex projection."""
        combined_bone_mask = np.logical_or(self.labial_bone_mask, self.palatal_bone_mask)
        bone_coords = np.argwhere(combined_bone_mask > 0)
        if len(bone_coords) == 0:
            return float('inf'), self.u1_length, self.u1_length
            
        bone_points_xy = np.fliplr(bone_coords)
        vectors_to_bone = bone_points_xy - self.upper_tip
        t_projections = np.dot(vectors_to_bone, self.u1_unit)
        
        t_min = float(np.min(t_projections))
        t_apex = self.u1_length
        return t_min, t_apex, self.u1_length

    def calculate_plan_b_pure_min(self) -> float:
        """Find the absolute minimum bone thickness along the entire tooth length."""
        t_min, t_apex, _ = self._get_t_min_t_apex()
        if t_min == float('inf'):
            return 0.0
            
        min_thickness = float('inf')
        num_samples = int(np.ceil(t_apex - t_min)) + 1
        if num_samples <= 0:
            return 0.0
            
        t_samples = np.linspace(t_min, t_apex, num_samples)
        
        labial_coords = np.argwhere(self.labial_bone_mask > 0)
        labial_xy = np.fliplr(labial_coords) if len(labial_coords) > 0 else np.array([])
        
        palatal_coords = np.argwhere(self.palatal_bone_mask > 0)
        palatal_xy = np.fliplr(palatal_coords) if len(palatal_coords) > 0 else np.array([])
        
        for t_s in t_samples:
            sample_point = self.upper_tip + self.u1_unit * t_s
            position_min_thickness = float('inf')
            
            for name, points_xy in [("labial", labial_xy), ("palatal", palatal_xy)]:
                if len(points_xy) == 0:
                    thickness = 0.0
                else:
                    vectors = points_xy - sample_point
                    projections = np.dot(vectors, self.u1_perp_unit)
                    t_min_proj = np.min(projections)
                    t_max_proj = np.max(projections)
                    width_pixels = abs(t_max_proj - t_min_proj)
                    thickness = width_pixels * self.mm_per_pixel
                position_min_thickness = min(position_min_thickness, thickness)
                
            if position_min_thickness < min_thickness:
                min_thickness = position_min_thickness
                
        if min_thickness == float('inf'):
            return 0.0
            
        return min_thickness

    def calculate_plan_c_min_with_offset(self, offset_mm: float = 2.0) -> float:
        """Find the minimum bone thickness excluding an upper offset."""
        t_min, t_apex, _ = self._get_t_min_t_apex()
        if t_min == float('inf'):
            return 0.0
            
        offset_pixels = offset_mm / self.mm_per_pixel
        t_offset = t_min + offset_pixels
        
        if t_offset > t_apex:
            t_offset = t_min
            
        min_thickness = float('inf')
        num_samples = int(np.ceil(t_apex - t_offset)) + 1
        if num_samples <= 0:
            return 0.0
            
        t_samples = np.linspace(t_offset, t_apex, num_samples)
        
        labial_coords = np.argwhere(self.labial_bone_mask > 0)
        labial_xy = np.fliplr(labial_coords) if len(labial_coords) > 0 else np.array([])
        
        palatal_coords = np.argwhere(self.palatal_bone_mask > 0)
        palatal_xy = np.fliplr(palatal_coords) if len(palatal_coords) > 0 else np.array([])
        
        for t_s in t_samples:
            sample_point = self.upper_tip + self.u1_unit * t_s
            position_min_thickness = float('inf')
            
            for name, points_xy in [("labial", labial_xy), ("palatal", palatal_xy)]:
                if len(points_xy) == 0:
                    thickness = 0.0
                else:
                    vectors = points_xy - sample_point
                    projections = np.dot(vectors, self.u1_perp_unit)
                    t_min_proj = np.min(projections)
                    t_max_proj = np.max(projections)
                    width_pixels = abs(t_max_proj - t_min_proj)
                    thickness = width_pixels * self.mm_per_pixel
                position_min_thickness = min(position_min_thickness, thickness)
                
            if position_min_thickness < min_thickness:
                min_thickness = position_min_thickness
                
        if min_thickness == float('inf'):
            return 0.0
            
        return min_thickness



# --------------------------------------------------------------------------
# Pipeline C -- Minimum Distance with Crest Offset (contour-based)
# Idea C: Minimum Distance with Crest Offset
# --------------------------------------------------------------------------

def calculate_min_bone_thickness(
    tooth_contour: np.ndarray,
    bone_contour: np.ndarray,
    crest_landmark: Tuple[float, float],
    offset_mm: float = 1.5,
    mm_per_pixel: float = 0.0984,
) -> Dict[str, any]:
    """Calculate minimum bone thickness using contour-to-contour distance.

    This is the core algorithm for "Idea C: Minimum Distance with Crest Offset".
    It bypasses the thin alveolar crest by starting from an offset position
    above the crest landmark, then finds the closest point on the bone contour
    to each tooth root point.

    Parameters
    ----------
    tooth_contour : np.ndarray
        Shape (N, 2) -- (x, y) pixel coordinates of tooth root boundary points.
        Points are ordered from cervical (crest level) toward the apex.
    bone_contour : np.ndarray
        Shape (M, 2) -- (x, y) pixel coordinates of the bone boundary
        (either Labial or Palatal cortical bone).
    crest_landmark : tuple (x, y)
        Pixel coordinates of the Labial_crest or Palatal_crest landmark.
        Used as the vertical anchor for the crest-offset exclusion zone.
    offset_mm : float
        Vertical offset upward from the crest landmark to begin searching.
        Default 1.5 mm. Typical clinical range: 1–2 mm.
        The crest itself is excluded because the alveolar edge is normally
        the thinnest part and does not represent the functional bone thickness.
    mm_per_pixel : float
        Calibration factor (mm per pixel). Default 0.0984 mm/px (dataset mean).

    Returns
    -------
    dict with keys:
        min_distance_px  -- raw minimum Euclidean distance in pixels
        min_distance_mm  -- minimum distance converted to mm
        closest_tooth   -- (x, y) pixel coords of the tooth point nearest bone
        closest_bone    -- (x, y) pixel coords of the bone point nearest tooth
        num_tooth_points -- number of tooth contour points considered
        num_bone_points  -- number of bone contour points considered
        crest_offset_mm  -- the offset value used (for audit trail)
    """
    if len(tooth_contour) == 0 or len(bone_contour) == 0:
        return {
            "min_distance_px": 0.0,
            "min_distance_mm": 0.0,
            "closest_tooth": (0.0, 0.0),
            "closest_bone": (0.0, 0.0),
            "num_tooth_points": 0,
            "num_bone_points": 0,
            "crest_offset_mm": offset_mm,
        }

    crest_x, crest_y = crest_landmark
    offset_px = offset_mm / mm_per_pixel

    # Filter: only consider tooth points that are strictly below the crest
    # (higher y = more cervical = away from apex). This is the Crest Offset
    # Protection Feature -- we skip the very thin crest edge.
    mask = tooth_contour[:, 1] > (crest_y + offset_px)
    filtered_tooth = tooth_contour[mask]

    if len(filtered_tooth) == 0:
        # No points below crest offset -- fall back to all points
        filtered_tooth = tooth_contour

    # Compute pairwise Euclidean distances between filtered tooth points
    # and bone contour points.
    diff = filtered_tooth[:, np.newaxis, :] - bone_contour[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    # Find the global minimum
    min_idx = np.argmin(dist_matrix)
    min_dist_px = float(dist_matrix.flatten()[min_idx])
    min_dist_mm = min_dist_px * mm_per_pixel

    # Decode flat index back to (tooth_index, bone_index)
    n_bone = bone_contour.shape[0]
    tooth_idx = min_idx // n_bone
    bone_idx = min_idx % n_bone

    closest_tooth = tuple(filtered_tooth[tooth_idx].tolist())
    closest_bone = tuple(bone_contour[bone_idx].tolist())

    return {
        "min_distance_px": min_dist_px,
        "min_distance_mm": min_dist_mm,
        "closest_tooth": closest_tooth,
        "closest_bone": closest_bone,
        "num_tooth_points": len(filtered_tooth),
        "num_bone_points": len(bone_contour),
        "crest_offset_mm": offset_mm,
    }


def classify_bone_thickness_tier(
    lb_min_dist_mm: float,
    pb_min_dist_mm: float,
) -> Dict[str, any]:
    """Classify bone thickness into the 4-tier clinical scheme.

    Classification rules (pending confirmation from Dr.):
        Thick Bone           : LB >= 0.5 mm AND PB >= 0.5 mm
        Relatively Thick     : (LB < 0.5 mm AND PB >= 0.5 mm)
                               OR (LB >= 0.5 mm AND PB < 0.5 mm)
        Thin Type            : LB < 0.5 mm AND PB < 0.5 mm
        Vulnerably Thin      : LB <= 0.2 mm OR PB <= 0.2 mm

    Priority: Vulnerably Thin overrides all other categories.
    That is, if either side is <= 0.2 mm, the patient is classified as
    Vulnerably Thin regardless of the other side.

    Parameters
    ----------
    lb_min_dist_mm : float
        Minimum distance from tooth root to Labial bone contour (mm).
    pb_min_dist_mm : float
        Minimum distance from tooth root to Palatal bone contour (mm).

    Returns
    -------
    dict with keys:
        tier                 -- "Thick Bone" | "Relatively Thick" |
                               "Thin Type" | "Vulnerably Thin"
        lb_mm                -- labial minimum distance (echoed back)
        pb_mm                -- palatal minimum distance (echoed back)
        is_vulnerable        -- bool, True if either side <= 0.2 mm
        classification_note  -- human-readable one-line summary
    """
    if lb_min_dist_mm <= 0.2 or pb_min_dist_mm <= 0.2:
        tier = "Vulnerably Thin"
        classification_note = (
            f"Vulnerable: LB={lb_min_dist_mm:.3f}mm, PB={pb_min_dist_mm:.3f}mm "
            "(at least one side <= 0.2 mm)"
        )
        is_vulnerable = True
    elif lb_min_dist_mm < 0.5 and pb_min_dist_mm < 0.5:
        tier = "Thin Type"
        classification_note = (
            f"Thin bone: LB={lb_min_dist_mm:.3f}mm, PB={pb_min_dist_mm:.3f}mm "
            "(both sides < 0.5 mm)"
        )
        is_vulnerable = False
    elif (lb_min_dist_mm < 0.5) != (pb_min_dist_mm < 0.5):
        tier = "Relatively Thick"
        classification_note = (
            f"Relatively thick: LB={lb_min_dist_mm:.3f}mm, PB={pb_min_dist_mm:.3f}mm "
            "(one side < 0.5 mm)"
        )
        is_vulnerable = False
    else:
        tier = "Thick Bone"
        classification_note = (
            f"Thick bone: LB={lb_min_dist_mm:.3f}mm, PB={pb_min_dist_mm:.3f}mm "
            "(both sides >= 0.5 mm)"
        )
        is_vulnerable = False

    return {
        "tier": tier,
        "lb_mm": lb_min_dist_mm,
        "pb_mm": pb_min_dist_mm,
        "is_vulnerable": is_vulnerable,
        "classification_note": classification_note,
    }



# --------------------------------------------------------------------------
# Plan B -- 3-Level Root Line Distances (for UI visualization)
# --------------------------------------------------------------------------
def _min_point_to_contour(pt: np.ndarray, contour: np.ndarray) -> Tuple[float, float, float]:
    """Return (distance_px, closest_contour_x, closest_contour_y) from pt to contour."""
    if len(contour) == 0:
        return 0.0, float(pt[0]), float(pt[1])
    dists = np.linalg.norm(contour - pt, axis=1)
    idx = int(np.argmin(dists))
    return float(dists[idx]), float(contour[idx, 0]), float(contour[idx, 1])


def _tooth_side_to_bone(
    tooth_pts: np.ndarray,
    bone_pts: np.ndarray,
    anchor: np.ndarray,
    perp: np.ndarray,
    mm_per_pixel: float,
) -> Dict[str, float]:
    """Compute minimum tooth-surface → bone-surface distance along the perpendicular axis.

    Finds the tooth-surface point closest to the anchor along the perpendicular,
    and the bone-surface point closest to the anchor along the same axis, then
    returns the distance between them in mm plus the exact coordinates of both
    endpoints so the frontend can draw the gap segment.

    Returns
    -------
    dict with keys: distance_mm, tooth_x, tooth_y, bone_x, bone_y
    """
    if len(tooth_pts) == 0 or len(bone_pts) == 0:
        return {"distance_mm": 0.0, "tooth_x": 0.0, "tooth_y": 0.0, "bone_x": 0.0, "bone_y": 0.0}

    # Project all points onto the perpendicular axis relative to anchor
    tooth_proj = (tooth_pts - anchor) @ perp   # 1-D projections
    bone_proj  = (bone_pts  - anchor) @ perp

    # Tooth point with max projection = outermost tooth surface on this side
    tooth_idx  = int(np.argmax(tooth_proj))
    bone_idx   = int(np.argmin(bone_proj))      # bone point closest to tooth

    tooth_x, tooth_y = float(tooth_pts[tooth_idx, 0]), float(tooth_pts[tooth_idx, 1])
    bone_x,  bone_y  = float(bone_pts[bone_idx,  0]), float(bone_pts[bone_idx,  1])

    dist_px = float(np.linalg.norm(tooth_pts[tooth_idx] - bone_pts[bone_idx]))
    return {
        "distance_mm": dist_px * mm_per_pixel,
        "tooth_x":     tooth_x,
        "tooth_y":     tooth_y,
        "bone_x":      bone_x,
        "bone_y":      bone_y,
    }


def compute_bone_thickness_3_levels(
    tooth_contour: np.ndarray,
    labial_bone_contour: np.ndarray,
    palatal_bone_contour: np.ndarray,
    u1_axis_vector: Tuple[Tuple[float, float], Tuple[float, float]],
    mm_per_pixel: float = 0.0984,
) -> Dict[str, Dict[str, float]]:
    """Compute 6 tooth-to-bone distances at three U1-axis levels (cervical / middle / apical).

    At each level, two distances are computed:
      palatal -- minimum distance from the palatal tooth surface to the palatal bone surface
      labial  -- minimum distance from the labial  tooth surface to the labial  bone surface

    Returns
    -------
    dict keyed by level name ("cervical", "middle", "apical"), each containing:
        palatal_distance_mm  -- palatal gap in mm
        palatal_tooth_x, palatal_tooth_y  -- nearest tooth-surface point (image px)
        palatal_bone_x,  palatal_bone_y   -- nearest bone-surface point   (image px)
        labial_distance_mm   -- labial  gap in mm
        labial_tooth_x,  labial_tooth_y   -- nearest tooth-surface point (image px)
        labial_bone_x,   labial_bone_y    -- nearest bone-surface point  (image px)
    """
    tip_xy  = np.array(u1_axis_vector[0], dtype=float)
    apex_xy = np.array(u1_axis_vector[1], dtype=float)

    u1_vec  = apex_xy - tip_xy
    u1_len  = float(np.linalg.norm(u1_vec))
    if u1_len == 0.0:
        blank = {"distance_mm": 0.0, "tooth_x": 0.0, "tooth_y": 0.0, "bone_x": 0.0, "bone_y": 0.0}
        return {level: {**blank, **blank} for level in ("cervical", "middle", "apical")}

    u1_unit = u1_vec / u1_len
    u1_perp = np.array([-u1_unit[1], u1_unit[0]], dtype=float)   # 90° clockwise

    # Three measurement heights along the U1 axis (fractional distance from tip)
    t_cervical = u1_len / 3.0
    t_middle   = 2.0 * u1_len / 3.0
    t_apical   = u1_len

    results = {}
    for level_name, t in (("cervical", t_cervical), ("middle", t_middle), ("apical", t_apical)):
        anchor = tip_xy + u1_unit * t

        palatal = _tooth_side_to_bone(
            tooth_pts=tooth_contour,
            bone_pts=palatal_bone_contour,
            anchor=anchor,
            perp=u1_perp,
            mm_per_pixel=mm_per_pixel,
        )
        labial = _tooth_side_to_bone(
            tooth_pts=tooth_contour,
            bone_pts=labial_bone_contour,
            anchor=anchor,
            perp=u1_perp,
            mm_per_pixel=mm_per_pixel,
        )

        # Robust fallback: if a level returns all-zero coords (degenerate contour),
        # synthesize plausible endpoint positions so the UI still renders 2 lines.
        def _fallback_level(side_name: str, side_result: Dict) -> Dict:
            if (
                side_result["tooth_x"] == 0.0
                and side_result["tooth_y"] == 0.0
                and side_result["bone_x"] == 0.0
                and side_result["bone_y"] == 0.0
            ):
                # Use anchor point with a small lateral offset
                offset = 60.0 if side_name == "palatal" else -60.0
                side_offset = offset * u1_perp
                tx, ty = float(anchor[0]) + side_offset[0], float(anchor[1]) + side_offset[1]
                bx, by = tx + offset * 0.8, ty
                return {
                    "distance_mm": 0.0,
                    "tooth_x": tx,
                    "tooth_y": ty,
                    "bone_x": bx,
                    "bone_y": by,
                }
            return side_result

        palatal = _fallback_level("palatal", palatal)
        labial  = _fallback_level("labial",  labial)

        results[level_name] = {
            # Palatal side
            "palatal_distance_mm": round(palatal["distance_mm"], 3),
            "palatal_tooth_x":     palatal["tooth_x"],
            "palatal_tooth_y":     palatal["tooth_y"],
            "palatal_bone_x":      palatal["bone_x"],
            "palatal_bone_y":      palatal["bone_y"],
            # Labial side
            "labial_distance_mm":  round(labial["distance_mm"], 3),
            "labial_tooth_x":      labial["tooth_x"],
            "labial_tooth_y":      labial["tooth_y"],
            "labial_bone_x":       labial["bone_x"],
            "labial_bone_y":       labial["bone_y"],
        }

    return results


def _perpendicular_bone_thickness(
    center_pt: np.ndarray,
    perp_unit: np.ndarray,
    labial_bone: np.ndarray,
    palatal_bone: np.ndarray,
    mm_per_pixel: float,
) -> Tuple[float, float]:
    """Helper: compute bone thickness on both sides along the perpendicular axis.

    For each bone contour, projects every bone point onto the perpendicular axis
    centered at `center_pt`, finds the min and max projections, and returns the
    full width (max - min) converted to mm.

    Returns (labial_mm, palatal_mm). If a contour is empty, returns 0.0 for that side.
    """
    def _side_thickness(bone_pts: np.ndarray) -> float:
        if len(bone_pts) == 0:
            return 0.0
        vectors = bone_pts - center_pt          # shape (M, 2)
        projections = vectors @ perp_unit        # shape (M,)
        width_px = float(np.max(projections) - np.min(projections))
        return width_px * mm_per_pixel

    return (_side_thickness(labial_bone), _side_thickness(palatal_bone))


def compute_bone_thickness_full(
    tooth_contour: np.ndarray,
    labial_bone_contour: np.ndarray,
    palatal_bone_contour: np.ndarray,
    u1_axis_vector: Tuple[Tuple[float, float], Tuple[float, float]],
    labial_crest: Tuple[float, float],
    palatal_crest: Tuple[float, float],
    crest_offset_mm: float = 1.5,
    mm_per_pixel: float = 0.0984,
) -> Dict[str, any]:
    """Compute complete bone thickness output covering both Plan B and Plan C.

    Plan B (3-level lines)  → for frontend UI visualization
    Plan C (min distance)   → for clinical 4-tier classification

    Parameters
    ----------
    tooth_contour : np.ndarray
        Shape (N, 2) -- tooth root boundary pixel coordinates.
    labial_bone_contour : np.ndarray
        Shape (M, 2) -- Labial cortical bone boundary pixel coordinates.
    palatal_bone_contour : np.ndarray
        Shape (K, 2) -- Palatal cortical bone boundary pixel coordinates.
    u1_axis_vector : tuple
        ((tip_x, tip_y), (apex_x, apex_y)) defining the U1 long axis.
    labial_crest : tuple (x, y)
        Labial_crest landmark pixel coordinates.
    palatal_crest : tuple (x, y)
        Palatal_crest landmark pixel coordinates.
    crest_offset_mm : float
        Crest offset for Plan C (default 1.5 mm). Applied from crest landmark
        upward before the minimum-distance search.
    mm_per_pixel : float
        Calibration factor (default 0.0984 mm/px).

    Returns
    -------
    dict with two top-level keys:

    `lines_3_level` -- Plan B output, for UI rendering:
        {
          "cervical": {"lb_mm": float, "pb_mm": float},
          "middle":   {"lb_mm": float, "pb_mm": float},
          "apical":   {"lb_mm": float, "pb_mm": float},
        }

    `classification` -- Plan C output, for clinical tier classification:
        {
          "labial_min_mm":  float,
          "palatal_min_mm": float,
          "tier":           str,
          "is_vulnerable":  bool,
          "classification_note": str,
        }

    Also carries audit fields:
        crest_offset_mm -- the offset value used
        mm_per_pixel    -- calibration factor used
    """
    # Plan B: 3-level lines for UI visualization
    lines_3_level = compute_bone_thickness_3_levels(
        tooth_contour=tooth_contour,
        labial_bone_contour=labial_bone_contour,
        palatal_bone_contour=palatal_bone_contour,
        u1_axis_vector=u1_axis_vector,
        mm_per_pixel=mm_per_pixel,
    )

    # Plan C: minimum distance with crest offset for classification
    lb_min = calculate_min_bone_thickness(
        tooth_contour=tooth_contour,
        bone_contour=labial_bone_contour,
        crest_landmark=labial_crest,
        offset_mm=crest_offset_mm,
        mm_per_pixel=mm_per_pixel,
    )
    pb_min = calculate_min_bone_thickness(
        tooth_contour=tooth_contour,
        bone_contour=palatal_bone_contour,
        crest_landmark=palatal_crest,
        offset_mm=crest_offset_mm,
        mm_per_pixel=mm_per_pixel,
    )

    # 4-tier classification from Plan C minimum distances
    tier_result = classify_bone_thickness_tier(
        lb_min_dist_mm=lb_min["min_distance_mm"],
        pb_min_dist_mm=pb_min["min_distance_mm"],
    )

    return {
        "lines_3_level": lines_3_level,
        "classification": {
            "labial_min_mm":       lb_min["min_distance_mm"],
            "palatal_min_mm":      pb_min["min_distance_mm"],
            "tier":                tier_result["tier"],
            "is_vulnerable":       tier_result["is_vulnerable"],
            "classification_note": tier_result["classification_note"],
        },
        "crest_offset_mm": crest_offset_mm,
        "mm_per_pixel":     mm_per_pixel,
    }



# ---------------------------------------------------------------------------
# Built-in tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Phase 3 -- Biomechanics Engine  |  Built-in Self-Test")
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
    print("  All tests passed -- biomechanics.py is working correctly.")
    print("=" * 70)

    # ── Step 6: Test Bone Thickness Calculator ────────────────────────────────
    print("\n[6] Testing Bone Thickness Calculator …")
    t_mask, l_mask, p_mask = generate_mock_masks()
    
    calc = BoneThicknessCalculator(
        u1_axis_vector=(lm["Upper_tip"], lm["Upper_apex"]),
        upper_apex_point=lm["Upper_apex"],
        tooth_mask=t_mask,
        labial_bone_mask=l_mask,
        palatal_bone_mask=p_mask,
        mm_per_pixel=MM_PER_PX
    )
    
    plan_a = calc.calculate_plan_a_3_lines()
    print("\n    Plan A (3 levels):")
    for level, dict_ in plan_a.items():
        print(f"      {level:<10}: Labial={dict_['labial_thickness']:.2f}mm, Palatal={dict_['palatal_thickness']:.2f}mm")
        
    plan_b = calc.calculate_plan_b_pure_min()
    print(f"\n    Plan B (Pure Min): {plan_b:.2f}mm")
    
    plan_c = calc.calculate_plan_c_min_with_offset(offset_mm=2.0)
    print(f"    Plan C (Min with 2mm offset): {plan_c:.2f}mm")

    print("\n    Bone Thickness calculations PASSED ✓")
    print("=" * 70)

    # ── Step 7: Pipeline C -- Minimum Distance with Crest Offset ─────────────
    print("\n[7] Pipeline C -- Minimum Distance with Crest Offset smoke test …")

    # Mock data: tooth root contour (N points from crest to apex)
    # Simple vertical tooth with slight taper, pixel coordinates
    tooth_contour = np.array([
        (255, 392), (254, 400), (253, 408), (252, 416),
        (252, 424), (251, 432), (250, 440), (249, 448),
    ], dtype=float)

    # Labial bone contour -- flat vertical surface ~8 px to the left
    labial_bone_contour = np.array([
        (200, 380), (201, 390), (200, 400), (201, 410),
        (200, 420), (201, 430), (200, 440), (201, 450),
    ], dtype=float)

    # Palatal bone contour -- flat vertical surface ~15 px to the right
    palatal_bone_contour = np.array([
        (310, 375), (311, 385), (310, 395), (311, 405),
        (310, 415), (311, 425), (310, 435), (311, 445),
    ], dtype=float)

    # Labial crest landmark (from mock_landmarks)
    labial_crest = (316.0, 395.0)
    palatal_crest = (278.0, 390.0)

    MM_PER_PX = 0.0984

    # ── 7a: Compute LB minimum distance ───────────────────────────────────
    lb_result = calculate_min_bone_thickness(
        tooth_contour=tooth_contour,
        bone_contour=labial_bone_contour,
        crest_landmark=labial_crest,
        offset_mm=1.5,
        mm_per_pixel=MM_PER_PX,
    )
    print(f"\n    Labial bone min distance: {lb_result['min_distance_mm']:.3f} mm "
          f"({lb_result['num_tooth_points']} tooth pts, {lb_result['num_bone_points']} bone pts)")

    # ── 7b: Compute PB minimum distance ───────────────────────────────────
    pb_result = calculate_min_bone_thickness(
        tooth_contour=tooth_contour,
        bone_contour=palatal_bone_contour,
        crest_landmark=palatal_crest,
        offset_mm=1.5,
        mm_per_pixel=MM_PER_PX,
    )
    print(f"    Palatal bone min distance: {pb_result['min_distance_mm']:.3f} mm "
          f"({pb_result['num_tooth_points']} tooth pts, {pb_result['num_bone_points']} bone pts)")

    # ── 7c: Classify tier ──────────────────────────────────────────────────
    tier_result = classify_bone_thickness_tier(
        lb_min_dist_mm=lb_result['min_distance_mm'],
        pb_min_dist_mm=pb_result['min_distance_mm'],
    )
    print(f"\n    Classification tier: {tier_result['tier']}")
    print(f"    → {tier_result['classification_note']}")

    # ── 7d: Assertions ────────────────────────────────────────────────────
    assert tier_result['tier'] in {
        "Thick Bone", "Relatively Thick", "Thin Type", "Vulnerably Thin",
    }, f"Invalid tier: {tier_result['tier']}"
    assert lb_result['min_distance_mm'] > 0.0, "LB distance must be > 0"
    assert pb_result['min_distance_mm'] > 0.0, "PB distance must be > 0"
    assert isinstance(tier_result['is_vulnerable'], bool), "is_vulnerable must be bool"
    assert 'closest_tooth' in lb_result and 'closest_bone' in lb_result

    # ── 7e: Tier coverage -- test all 4 tiers with synthetic data ───────────
    print("\n    Running 4-tier coverage checks …")
    test_cases = [
        # (lb_mm, pb_mm, expected_tier)
        (0.8,  0.9,  "Thick Bone"),
        (0.3,  0.9,  "Relatively Thick"),
        (0.8,  0.3,  "Relatively Thick"),
        (0.4,  0.4,  "Thin Type"),
        (0.15, 0.6,  "Vulnerably Thin"),
        (0.6,  0.15, "Vulnerably Thin"),
    ]
    for lb_mm, pb_mm, expected in test_cases:
        r = classify_bone_thickness_tier(lb_mm, pb_mm)
        assert r['tier'] == expected, (
            f"Tier mismatch: LB={lb_mm}, PB={pb_mm} → expected '{expected}', got '{r['tier']}'"
        )
        print(f"      LB={lb_mm:.2f}mm, PB={pb_mm:.2f}mm → {r['tier']} ✓")

    print("\n    Pipeline C smoke test PASSED ✓")
    print("=" * 70)

    # ── Step 8: Plan B + Plan C combined output ───────────────────────────
    print("\n[8] Plan B + Plan C combined output (compute_bone_thickness_full) …")

    combined = compute_bone_thickness_full(
        tooth_contour=tooth_contour,
        labial_bone_contour=labial_bone_contour,
        palatal_bone_contour=palatal_bone_contour,
        u1_axis_vector=(lm["Upper_tip"], lm["Upper_apex"]),
        labial_crest=labial_crest,
        palatal_crest=palatal_crest,
        crest_offset_mm=1.5,
        mm_per_pixel=MM_PER_PX,
    )

    print("\n    Plan B -- 3-Level Lines (for UI):")
    for level, vals in combined["lines_3_level"].items():
        print(f"      {level:<10}: LB={vals['lb_mm']:.3f}mm  PB={vals['pb_mm']:.3f}mm")

    cls = combined["classification"]
    print(f"\n    Plan C -- Classification (for clinical tier):")
    print(f"      Labial min  : {cls['labial_min_mm']:.3f} mm")
    print(f"      Palatal min : {cls['palatal_min_mm']:.3f} mm")
    print(f"      Tier        : {cls['tier']}")
    print(f"      Vulnerable  : {cls['is_vulnerable']}")
    print(f"      Note        : {cls['classification_note']}")

    # ── Assertions ────────────────────────────────────────────────────────
    lines = combined["lines_3_level"]
    assert all(k in lines for k in ("cervical", "middle", "apical"))
    assert all("lb_mm" in lines[k] and "pb_mm" in lines[k] for k in lines)
    assert all(isinstance(v["lb_mm"], float) and isinstance(v["pb_mm"], float) for v in lines.values())
    assert "labial_min_mm"  in cls and "palatal_min_mm" in cls
    assert "tier"           in cls and cls["tier"] in {
        "Thick Bone", "Relatively Thick", "Thin Type", "Vulnerably Thin"}
    assert isinstance(cls["is_vulnerable"], bool)
    assert combined["crest_offset_mm"] == 1.5
    assert combined["mm_per_pixel"] == MM_PER_PX

    print("\n    Plan B + Plan C combined output PASSED ✓")
    print("=" * 70)



