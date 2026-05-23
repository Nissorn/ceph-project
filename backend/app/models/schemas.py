from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class LandmarkPoint(BaseModel):
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


class AnalysisRequest(BaseModel):
    image_id: str
    landmarks: Optional[Dict[str, LandmarkPoint]] = None


class AnalysisResponse(BaseModel):
    status: str
    image_id: Optional[str] = None
    landmarks: Optional[Dict[str, LandmarkPoint]] = None
    polygons: Optional[List[Dict[str, Any]]] = None
    bone_thickness: Optional[Dict[str, Any]] = None
    biomechanical_constraints: Optional[BiomechanicalConstraints] = None
    treatment_recommendation: Optional[TreatmentRecommendation] = None
