from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.models.schemas import AnalysisResponse
from backend.app.services.inference_service import InferenceService

# Lazy singleton — model loaded once when first request hits the endpoint.
_inference_service: InferenceService | None = None


def _get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(
    file: UploadFile = File(...),
    use_tta: bool = True,
):
    """
    Run cephalometric landmark detection on an uploaded X-ray image.

    Parameters
    ----------
    file : UploadFile
        JPEG / PNG / TIFF / BMP X-ray image.
    use_tta : bool
        If True (default), applies 5-variant TTA (orig + rot± + brig±) and
        averages coordinates. Set False for single-pass inference (~20% faster).

    Returns
    -------
    AnalysisResponse
        status: "success" | "error"
        data.landmarks: {name: {x, y, confidence}, ...} for all 10 keypoints
        data.use_tta: bool
    """
    # Validate MIME type
    ALLOWED_CONTENT_TYPES = {
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/bmp",
        "image/webp",
    }
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Supported: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # Read image bytes
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file uploaded.")

    # Run inference
    try:
        svc = _get_inference_service()
        result = svc.predict(image_bytes, use_tta=use_tta)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not loaded: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    return AnalysisResponse(
        status="success",
        data={
            "landmarks": result["landmarks"],
            "use_tta": result["use_tta"],
        },
    )
