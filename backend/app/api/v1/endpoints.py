from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import AnalysisResponse
from app.services.analysis_service import analysis_service

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(file: UploadFile = File(...)):
    """
    Upload a cephalogram JPEG/PNG image for full Phase 2A (HRNet-W32)
    landmark detection + Phase 2B (DeepLabV3Plus) segmentation +
    geometric snapping pipeline.

    Returns landmarks (10 pts), segmentation polygons, and biomechanical
    metrics compatible with the Astro + Konva frontend canvas.
    """
    # Read the uploaded file into memory
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    # Extract image_id from the uploaded filename
    from pathlib import Path
    image_id = Path(file.filename).stem if file.filename else None

    # Run the full production pipeline
    try:
        result = analysis_service().analyze_image(contents, image_id=image_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {e}")

    # Wrap in AnalysisResponse
    return AnalysisResponse(status=result["status"], data=result)