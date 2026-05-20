from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Tuple
from backend.app.models.schemas import AnalysisResponse
from backend.app.services.inference_service import InferenceService

import sys
from pathlib import Path

# Add project root to sys.path so we can import src.phase3.biomechanics
_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # …/ceph-v2-auto
sys.path.insert(0, str(_ROOT))
from src.phase3.biomechanics import calculate_metrics

# Lazy singleton — model loaded once when first request hits the endpoint.
_inference_service: InferenceService | None = None


def _get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


router = APIRouter()

# Default calibration — per-image calibration via calibration.csv is applied
# at the image-loader layer; this default is a safe fallback.
DEFAULT_MM_PER_PIXEL = 0.0984


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(
    file: UploadFile = File(...),
    use_tta: bool = True,
):
    """
    Run cephalometric landmark detection on an uploaded X-ray image
    and compute clinical biomechanics metrics.

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
        data.metrics: {u1_pp_angle_deg, lb_apex_dist_mm, pb_apex_dist_mm}
        data.bone_thickness: {labial_min_mm, mandibular_min_mm}
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

    landmarks = result["landmarks"]

    # Transform InferenceService output (dict of {x,y,conf} objects)
    # into the flat coordinate format expected by calculate_metrics().
    landmark_coords: Dict[str, Tuple[float, float]] = {}
    for name, pt in landmarks.items():
        landmark_coords[name] = (float(pt["x"]), float(pt["y"]))

    # ── Compute clinical metrics from detected landmarks ──────────────────────
    try:
        metrics = calculate_metrics(landmark_coords, mm_per_pixel=DEFAULT_MM_PER_PIXEL)
    except Exception:
        # Surface-level landmark errors should not abort the whole response
        metrics = {}

    # Map biomechanics output to the field names the frontend expects:
    #   lb_apex_dist_mm  → labial_min_mm    (Maxillary Bone = labial side)
    #   pb_apex_dist_mm  → mandibular_min_mm (Mandibular Bone = palatal side)
    bone_thickness = {}
    if "lb_apex_dist_mm" in metrics:
        bone_thickness["labial_min_mm"] = round(metrics["lb_apex_dist_mm"], 3)
    if "pb_apex_dist_mm" in metrics:
        bone_thickness["mandibular_min_mm"] = round(metrics["pb_apex_dist_mm"], 3)

    response_data = {
        "landmarks": landmarks,
        "use_tta": result["use_tta"],
        "metrics": {k: round(v, 3) for k, v in metrics.items()},
        "bone_thickness": bone_thickness,
    }

    return AnalysisResponse(
        status="success",
        data=response_data,
    )
