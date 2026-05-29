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


class BoneThickness(BaseModel):
    labial_min_mm: float
    mandibular_min_mm: float


class Classification(BaseModel):
    interpretation: str
    root_apex_position: Optional[str] = None
    preferred_biomechanics: Optional[str] = None
    avoid: Optional[str] = None
    implication: Optional[str] = None


class AnalysisResultData(BaseModel):
    image_id: str
    landmarks: List[LandmarkPoint]
    raw_landmarks: List[LandmarkPoint]
    segmentation: SegmentationData
    snapping: Dict[str, Any]
    mask_overlap_diagnostic: MaskOverlapDiagnostic
    metrics: Metrics
    bone_thickness: Optional[BoneThickness] = None
    classification: Optional[Classification] = None
    _debug: Optional[DebugInfo] = None


class AnalysisResponse(BaseModel):
    status: str
    data: AnalysisResultData