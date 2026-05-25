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
    overlap_before: int
    overlap_after: int
    pixels_corrected: int


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


class AnalysisResultData(BaseModel):
    image_id: str
    landmarks: List[LandmarkPoint]
    raw_landmarks: List[LandmarkPoint]
    segmentation: SegmentationData
    snapping: Dict[str, Any]
    mask_overlap_diagnostic: MaskOverlapDiagnostic
    metrics: Metrics
    _debug: Optional[DebugInfo] = None


class AnalysisResponse(BaseModel):
    status: str
    data: AnalysisResultData