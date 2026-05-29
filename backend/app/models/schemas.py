from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class Point(BaseModel):
    x: float
    y: float


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


class AnalysisResponse(BaseModel):
    status: str
    data: AnalysisResultData