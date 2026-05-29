from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class Point(BaseModel):
    x: float
    y: float
    confidence: Optional[float] = None


class BiomechanicalConstraints(BaseModel):
    """Phase 3 output — current biomechanical state and safety limits."""
    u1_pp_angle_deg: Optional[float] = None
    angle_zone: Optional[str] = None           # "<105" | "105-115" | ">115"
    root_apex_position: Optional[str] = None   # "Labial" | "Midway" | "Palatal"
    lb_apex_dist_mm: Optional[float] = None
    pb_apex_dist_mm: Optional[float] = None
    incisor_condition: Optional[str] = None
    preferred_biomechanics: Optional[str] = None
    biomechanics_to_avoid: Optional[str] = None
    clinical_implication: Optional[str] = None


class TreatmentRecommendation(BaseModel):
    """Phase 2c output — forward-looking treatment planning recommendation."""
    recommended_class: Optional[str] = None       # e.g. "Controlled_tipping"
    confidence: Optional[float] = None            # 0–1 sigmoid probability
    probabilities: Optional[Dict[str, float]] = None   # per-class probs
    insufficient_classes: Optional[List[str]] = None   # below min_support threshold
    # "pending_training" until Phase 2c model weights exist; "active" once deployed
    model_status: str = "pending_training"


class LandmarkPoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float
    snapped: bool = False


class SegmentationClass(BaseModel):
    polygon: List[List[float]]   # [[x, y], ...]
    pixel_count: int


class SnappingDiag(BaseModel):
    dx: float
    dy: float
    dist_px: float
    note: Optional[str] = None


class MaskOverlapDiagnostic(BaseModel):
    overlap_before: Optional[int] = None
    overlap_after: Optional[int] = None
    pixels_corrected: Optional[int] = None
    note: Optional[str] = None


class Metrics(BaseModel):
    u1_pp_angle_deg: float
    labial_crest_mm: float
    labial_crest_severity: str
    labial_midroot_mm: float
    labial_midroot_severity: str
    labial_apex_mm: float
    labial_apex_severity: str
    palatal_crest_mm: float
    palatal_crest_severity: str
    palatal_midroot_mm: float
    palatal_midroot_severity: str
    palatal_apex_mm: float
    palatal_apex_severity: str
    bone_thickness_type: str
    bone_thickness_interpretation: str
    root_apex_position_type: str
    general_retraction_strategy: str
    preferred_biomechanics: str
    biomechanics_to_avoid: str
    clinical_implication: str


class MeasurementLines(BaseModel):
    labial_crest_line: List[List[float]]
    labial_midroot_line: List[List[float]]
    labial_apex_line: List[List[float]]
    palatal_crest_line: List[List[float]]
    palatal_midroot_line: List[List[float]]
    palatal_apex_line: List[List[float]]


class DebugInfo(BaseModel):
    orig_width: int
    orig_height: int
    scale_x: float
    scale_y: float
    device: str


class SegmentationData(BaseModel):
    Upper_incisor: SegmentationClass
    Labial_bone: SegmentationClass
    Palatal_bone: SegmentationClass


class AnalysisResultData(BaseModel):
    image_id: Optional[str] = None
    landmarks: Optional[List[LandmarkPoint]] = None
    raw_landmarks: Optional[List[LandmarkPoint]] = None
    segmentation: Optional[SegmentationData] = None
    snapping: Optional[Dict[str, Any]] = None
    mask_overlap_diagnostic: Optional[MaskOverlapDiagnostic] = None
    metrics: Metrics
    measurement_lines: Optional[MeasurementLines] = None
    _debug: Optional[DebugInfo] = None


class AnalysisRequest(BaseModel):
    image_id: str
    landmarks: Optional[Dict[str, Point]] = None


class AnalysisResponse(BaseModel):
    status: str
    data: AnalysisResultData
