from fastapi import APIRouter
from backend.app.models.schemas import AnalysisRequest, AnalysisResponse
from backend.app.services.analysis_service import analysis_service

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(request: AnalysisRequest):
    """
    Endpoint for executing cephalometric analysis
    """
    # Convert Pydantic model to dict
    request_data = request.model_dump()
    
    # Process using service
    analysis_result = analysis_service.analyze_image(request_data)
    
    return AnalysisResponse(
        status="success",
        data=analysis_result
    )