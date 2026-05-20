from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Tuple, List, Any
from backend.app.models.schemas import AnalysisResponse
from backend.app.services.inference_service import InferenceService

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path so we can import src.phase3.biomechanics
_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # …/ceph-v2-auto
sys.path.insert(0, str(_ROOT))
from src.phase3.biomechanics import calculate_metrics, compute_bone_thickness_full

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
    bone_thickness: Dict[str, Any] = {}
    if "lb_apex_dist_mm" in metrics:
        bone_thickness["labial_min_mm"] = round(metrics["lb_apex_dist_mm"], 3)
    if "pb_apex_dist_mm" in metrics:
        bone_thickness["mandibular_min_mm"] = round(metrics["pb_apex_dist_mm"], 3)

    # Pre-initialize response_data so the try block can write into it.
    # Fields are filled in below; classification may be added by Plan B/C.
    response_data: Dict[str, Any] = {
        "landmarks": landmarks,
        "use_tta": result["use_tta"],
        "metrics": {k: round(v, 3) for k, v in metrics.items()},
        "bone_thickness": bone_thickness,
    }

    # ── Plan B: compute 3-level bone thickness lines for frontend rendering ──
    # Build contour arrays from detected landmarks (all in original image px).
    # We approximate each contour as a small box around the relevant landmarks.
    try:
        tip    = np.array(landmark_coords["Upper_tip"],       dtype=float)
        apex   = np.array(landmark_coords["Upper_apex"],      dtype=float)
        lb_crest  = np.array(landmark_coords["Labial_crest"],  dtype=float)
        pb_crest  = np.array(landmark_coords["Palatal_crest"], dtype=float)
        lb_pt  = np.array(landmark_coords["LB"],               dtype=float)
        pb_pt  = np.array(landmark_coords["PB"],              dtype=float)

        u1_axis_vector: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (float(tip[0]),  float(tip[1])),
            (float(apex[0]), float(apex[1])),
        )

        # Approximate tooth contour: bounding box around the 6 tooth landmarks
        tooth_pts = np.array([
            landmark_coords["Upper_tip"],
            landmark_coords["Upper_apex"],
            landmark_coords["Labial_midroot"],
            landmark_coords["Labial_crest"],
            landmark_coords["Palatal_midroot"],
            landmark_coords["Palatal_crest"],
        ], dtype=float)
        # Build a loose contour: shrink the bounding box by 5px on each side
        tx_min, ty_min = tooth_pts.min(axis=0) - 5
        tx_max, ty_max = tooth_pts.max(axis=0) + 5
        tooth_contour = np.array([
            [tx_min, ty_min], [tx_max, ty_min], [tx_max, ty_max], [tx_min, ty_max],
        ], dtype=float)

        # Labial bone contour: mirrored offset from LB toward labial side
        # Vector from tooth center to LB, then extend 30px further
        tooth_center = tooth_pts.mean(axis=0)
        lb_vec = lb_pt - tooth_center
        lb_unit = lb_vec / np.linalg.norm(lb_vec)
        labial_bone_contour = np.array([
            lb_pt + lb_unit * 25,
            lb_pt + lb_unit * 25 + np.array([-lb_unit[1], lb_unit[0]]) * 15,
            lb_pt - lb_unit * 5 + np.array([-lb_unit[1], lb_unit[0]]) * 15,
            lb_pt - lb_unit * 5,
        ], dtype=float)

        # Palatal bone contour: mirrored offset from PB toward palatal side
        pb_vec = pb_pt - tooth_center
        pb_unit = pb_vec / np.linalg.norm(pb_vec)
        palatal_bone_contour = np.array([
            pb_pt + pb_unit * 25,
            pb_pt + pb_unit * 25 + np.array([-pb_unit[1], pb_unit[0]]) * 15,
            pb_pt - pb_unit * 5 + np.array([-pb_unit[1], pb_unit[0]]) * 15,
            pb_pt - pb_unit * 5,
        ], dtype=float)

        # Compute full Plan B + Plan C output
        full_bt = compute_bone_thickness_full(
            tooth_contour=tooth_contour,
            labial_bone_contour=labial_bone_contour,
            palatal_bone_contour=palatal_bone_contour,
            u1_axis_vector=u1_axis_vector,
            labial_crest=(float(lb_crest[0]), float(lb_crest[1])),
            palatal_crest=(float(pb_crest[0]), float(pb_crest[1])),
            mm_per_pixel=DEFAULT_MM_PER_PIXEL,
        )

        # Inject Plan B (lines_3_level) into bone_thickness — includes line coords
        bone_thickness["lines_3_level"] = full_bt["lines_3_level"]

        # Inject Plan C (classification) — keyed at top-level alongside metrics
        response_data["classification"] = full_bt["classification"]

    except Exception as exc:
        # Surface-level errors (missing landmarks, degenerate contours) — skip
        # but still return metrics and the legacy bone_thickness fields.
        print(f"[analyze] Plan B/C computation skipped: {exc}")

    return AnalysisResponse(
        status="success",
        data=response_data,
    )
