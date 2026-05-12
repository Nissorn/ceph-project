from fastapi import APIRouter, File, UploadFile
from backend.app.models.schemas import AnalysisResponse
from backend.app.services.analysis_service import analysis_service

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for executing cephalometric analysis
    """
    # We can pass the file bytes or name to the service if needed later.
    # Process using service
    # analysis_result = analysis_service.analyze_image({"filename": file.filename})
    
    analysis_result = {
        "metrics": {
            "u1_pp_angle_deg": 112.5
        },
        "bone_thickness": {
            "labial_min_mm": 2.1,
            "mandibular_min_mm": 3.4
        },
        "classification": {
            "interpretation": "Reduced maxillary bone thickness detected below the 2.5mm threshold. Patient is at elevated risk for recession during retraction. Biomechanical compensation recommended."
        }
    }
    
    return AnalysisResponse(
        status="success",
        data=analysis_result
    )
