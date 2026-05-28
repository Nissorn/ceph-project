"""
scripts/test_backend_endpoint.py
================================
In-process integration test for the /api/v1/analyze FastAPI endpoint.

Uses FastAPI.testclient to POST a synthetic 512×512 JPEG cephalogram
and validates:
  - HTTP 200 OK response
  - Presence of required JSON fields (u1_pp_angle_deg, landmarks, segmentation)

Run locally (NOT inside Docker) from the project root:
    python scripts/test_backend_endpoint.py

Requires checkpoints at:
    data/processed/checkpoints/fold1_best.pth
    models/exp0128_DeepLabV3Plus_resnet34_20260524_043501/best_model.pt
"""

from __future__ import annotations

import sys, warnings, io
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
# Put the backend directory on sys.path so:
#   "from backend.app.main import app"  →  ROOT/backend/app/main.py
#   backend/app/main.py does:  "from app.api.v1.endpoints import ..."
#   → ROOT/backend/app/api/v1/endpoints.py  ✓
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# ── verify checkpoints exist before starting ──────────────────────────────────
CKPT_LANDMARK = ROOT / "data" / "processed" / "checkpoints" / "fold1_best.pth"
CKPT_SEG      = ROOT / "models" / "exp0128_DeepLabV3Plus_resnet34_20260524_043501" / "best_model.pt"
missing = [p for p in (CKPT_LANDMARK, CKPT_SEG) if not p.exists()]
if missing:
    print("[test_backend_endpoint] WARNING: checkpoint files not found:")
    for p in missing:
        print(f"    {p}")
    print("[test_backend_endpoint] The /analyze call will fail at model-loading.")
    print("[test_backend_endpoint] Proceeding anyway to test routing + import resolution.\n")

# ── bootstrap the FastAPI app ────────────────────────────────────────────────
# Project root (ceph-project/) is on sys.path.  The ./backend/app/ directory
# resolves as the "backend.app" package.  backend.app.main does:
#   from app.api.v1.endpoints import ...   (because app/ lives at /app/app/)
# which is found because ./backend/app/ is named "app" relative to the container
# root /app.  Locally, ROOT/backend/app/ is named "backend" on disk but the
# import "backend.app" resolves it correctly.
#
# Docker: context=./backend, WORKDIR /app, COPY . ., CMD ["uvicorn", "app.main:app", ...]
#   → container /app/app/ mirrors local ROOT/backend/app/
from backend.app.main import app          # noqa: E402

client = TestClient(app)


# ── synthetic 512×512 JPEG factory ────────────────────────────────────────────
def _synthetic_cephalogram(height: int = 512, width: int = 512) -> bytes:
    """
    Return a JPEG-encoded byte stream of a synthetic greyscale cephalogram.
    Renders a faint dental arch outline so the model has something to detect.
    """
    img = np.full((height, width, 3), 220, dtype=np.uint8)  # light background

    # Draw a tooth-like oval in the upper-centre region
    centre_x, centre_y = width // 2, height // 3
    cv2.ellipse(img, (centre_x, centre_y), (60, 90), 0, 0, 360, (160, 140, 120), -1)

    # Add a palatal bone trace (horizontal band below the tooth)
    cv2.rectangle(img,
                  (width // 4, height // 2),
                  (3 * width // 4, height // 2 + 30),
                  (130, 110, 90), -1)

    # Encode to JPEG
    ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


# ── field audit helpers ───────────────────────────────────────────────────────
REQUIRED_LANDMARK_NAMES = {
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
}
REQUIRED_SEG_CLASSES = {"Upper_incisor", "Labial_bone", "Palatal_bone"}


def _audit_response(payload: dict) -> tuple[bool, list[str]]:
    errors = []
    data = payload.get("data", {})

    status = payload.get("status")
    if status not in ("ok", "success", "degraded"):
        errors.append(f"Unexpected status: {status}")

    expected_disclaimer = "Estimation model based on lateral 2D cephalometric imaging. Must not be considered a replacement for CBCT evaluation."
    actual_disclaimer = payload.get("disclaimer")
    if actual_disclaimer != expected_disclaimer:
        errors.append(f"Expected disclaimer {expected_disclaimer!r}, got {actual_disclaimer!r}")

    # 2. landmarks (10 pts)
    lmks = data.get("landmarks")
    if lmks is not None or status != "degraded":
        if not isinstance(lmks, list) or len(lmks) != 10:
            errors.append(f"Expected 10 landmarks, got {lmks}")
        else:
            names = {lp["name"] for lp in lmks}
            missing = REQUIRED_LANDMARK_NAMES - names
            if missing:
                errors.append(f"Missing landmark names: {missing}")

    # 3. segmentation (3 classes, each with polygon + pixel_count)
    seg = data.get("segmentation")
    if seg is not None or status != "degraded":
        if not isinstance(seg, dict):
            errors.append(f"Expected dict for segmentation, got {seg}")
        else:
            for cls_name in REQUIRED_SEG_CLASSES:
                cls_data = seg.get(cls_name, {})
                if "polygon" not in cls_data:
                    errors.append(f"segmentation.{cls_name} missing 'polygon'")
                if "pixel_count" not in cls_data:
                    errors.append(f"segmentation.{cls_name} missing 'pixel_count'")

    # 4. metrics check
    metrics = data.get("metrics", {})
    required_metric_fields = [
        "u1_pp_angle_deg", "labial_crest_mm", "labial_midroot_mm", "labial_apex_mm",
        "palatal_crest_mm", "palatal_midroot_mm", "palatal_apex_mm",
        "labial_crest_severity", "labial_midroot_severity", "labial_apex_severity",
        "palatal_crest_severity", "palatal_midroot_severity", "palatal_apex_severity",
        "bone_thickness_type", "bone_thickness_interpretation", "root_apex_position_type",
        "general_retraction_strategy", "preferred_biomechanics", "biomechanics_to_avoid", "clinical_implication"
    ]
    for field in required_metric_fields:
        if field not in metrics:
            errors.append(f"metrics missing '{field}'")
        else:
            val = metrics[field]
            if field.endswith("_mm") or field.endswith("_deg"):
                if not isinstance(val, (int, float)):
                    errors.append(f"metrics.{field} is not numeric: {val!r}")
            else:
                if not isinstance(val, str) or not val:
                    errors.append(f"metrics.{field} is not a non-empty string: {val!r}")

    # 5. measurement_lines check
    m_lines = data.get("measurement_lines")
    if m_lines is not None or status != "degraded":
        if not isinstance(m_lines, dict):
            errors.append(f"Expected dict for measurement_lines, got {m_lines}")
        else:
            required_lines = [
                "labial_crest_line", "labial_midroot_line", "labial_apex_line",
                "palatal_crest_line", "palatal_midroot_line", "palatal_apex_line"
            ]
            for line_name in required_lines:
                if line_name not in m_lines:
                    errors.append(f"measurement_lines missing '{line_name}'")
                else:
                    line = m_lines[line_name]
                    if not isinstance(line, list) or len(line) != 2:
                        errors.append(f"measurement_lines.{line_name} is not a coordinate pair of length 2: {line!r}")
                    else:
                        for pt in line:
                            if not isinstance(pt, list) or len(pt) != 2 or not all(isinstance(coord, (int, float)) for coord in pt):
                                errors.append(f"measurement_lines.{line_name} contains invalid coordinate point: {pt!r}")

    return (len(errors) == 0, errors)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("[test_backend_endpoint] Building synthetic 512×512 cephalogram ...")
    image_bytes = _synthetic_cephalogram()

    print("[test_backend_endpoint] POST /api/v1/analyze ...")
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("cephalogram.jpg", image_bytes, "image/jpeg")},
    )

    print(f"[test_backend_endpoint] HTTP {response.status_code}")

    # Pretty-print the JSON response (truncated for readability)
    payload = response.json()
    raw_json = str(payload)
    if len(raw_json) > 600:
        print(f"[test_backend_endpoint] Response body ({len(raw_json)} chars):")
        print(raw_json[:600] + " ... (truncated)")
    else:
        print(f"[test_backend_endpoint] Response body:\n{raw_json}")

    # ── field audit ──────────────────────────────────────────────────────────
    ok, errors = _audit_response(payload)
    if not ok:
        print("[test_backend_endpoint] AUDIT FAILED:")
        for err in errors:
            print(f"  - {err}")
        print("\n=== VALIDATION RESULT: FAIL ===")
        sys.exit(1)
    else:
        angle = payload["data"]["metrics"]["u1_pp_angle_deg"]
        landmarks = payload["data"].get("landmarks")
        lm_count = len(landmarks) if landmarks is not None else 0
        seg = payload["data"].get("segmentation")
        seg_classes = list(seg.keys()) if seg is not None else []
        print(f"[test_backend_endpoint] AUDIT PASSED")
        print(f"  status               : {payload.get('status')}")
        print(f"  landmarks returned : {lm_count}")
        print(f"  segmentation classes: {seg_classes}")
        print(f"  u1_pp_angle_deg      : {angle}°")
        print("\n=== VALIDATION RESULT: PASS ===")
        sys.exit(0)


if __name__ == "__main__":
    main()