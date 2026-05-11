from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Point(BaseModel):
    x: float
    y: float

class AnalysisRequest(BaseModel):
    # Depending on how the frontend sends data, this could be paths or base64 images
    # For now we use some dummy inputs
    image_id: str
    landmarks: Optional[Dict[str, Point]] = None

class AnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]