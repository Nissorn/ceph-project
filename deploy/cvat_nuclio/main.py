import base64
import json
import io

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Label constants — must match data/annotations.xml exactly
# ---------------------------------------------------------------------------
POLYGON_CLASSES = ["Upper_incisor", "Labial_bone", "Palatal_bone"]

KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]


def init_context(context):
    """
    Called once when the Nuclio function starts. Load model weights here.

    Replace the comments below with real loading once weights are available:

        from src.phase2.model import build_hrnet
        from src.phase2b.segmentation import build_segmentation_model

        hrnet = build_hrnet(num_keypoints=10, pretrained=False)
        hrnet.load_state_dict(torch.load("models/hrnet_landmarks.pth", map_location="cuda"))
        hrnet.eval()
        context.user_data.hrnet_model = hrnet

        seg_model = build_segmentation_model()
        seg_model.load_state_dict(torch.load("models/unet_segmentation.pth", map_location="cuda"))
        seg_model.eval()
        context.user_data.seg_model = seg_model
    """
    context.user_data.hrnet_model = None   # replace with loaded HRNet
    context.user_data.seg_model = None     # replace with loaded U-Net
    context.logger.info("Cephalometric inference function initialised (models not yet loaded)")


def handler(context, event):
    """
    CVAT calls this for every image submitted to the auto-annotation function.

    Expected request body: JSON with {"image": "<base64-encoded-image>"}
    Response body: JSON array of annotation objects consumed by CVAT.
    """
    # ------------------------------------------------------------------
    # 1. Decode image
    # ------------------------------------------------------------------
    data = json.loads(event.body)
    image_bytes = base64.b64decode(data["image"])
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)          # H x W x 3, uint8
    h, w = image_np.shape[:2]

    # ------------------------------------------------------------------
    # 2. Pre-process (placeholder — match src/phase2/dataset.py transforms)
    # ------------------------------------------------------------------
    # tensor = preprocess(image_np)  # normalise, resize to 512×512, etc.

    # ------------------------------------------------------------------
    # 3. Landmark inference (HRNet)
    # ------------------------------------------------------------------
    # with torch.no_grad():
    #     heatmaps = context.user_data.hrnet_model(tensor)   # [1, 10, H, W]
    #     keypoints_xy = heatmaps_to_coords(heatmaps, original_size=(w, h))
    #
    # keypoints_xy is shape [10, 2] with (x, y) in original image pixels.

    # Mock: place all keypoints at image centre
    keypoints_xy = np.tile([w / 2, h / 2], (len(KEYPOINT_NAMES), 1))

    # ------------------------------------------------------------------
    # 4. Segmentation inference (U-Net)
    # ------------------------------------------------------------------
    # with torch.no_grad():
    #     logits = context.user_data.seg_model(tensor)       # [1, 3, H, W]
    #     masks  = (logits.sigmoid() > 0.5).cpu().numpy()[0] # [3, H, W] bool
    #
    # Convert each binary mask channel to a polygon contour with cv2:
    #   contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #   points = contours[0].flatten().tolist()

    # Mock: small rectangle in image centre for each polygon class
    cx, cy, r = w // 2, h // 2, min(w, h) // 8
    mock_polygon = [cx - r, cy - r, cx + r, cy - r, cx + r, cy + r, cx - r, cy + r]

    # ------------------------------------------------------------------
    # 5. Build CVAT response
    # ------------------------------------------------------------------
    results = []

    # Segmentation polygons
    for label in POLYGON_CLASSES:
        results.append({
            "confidence": "0.00",   # replace with real sigmoid score
            "label": label,
            "points": mock_polygon,
            "type": "polygon",
        })

    # Landmark keypoints
    for name, (x, y) in zip(KEYPOINT_NAMES, keypoints_xy):
        results.append({
            "confidence": "0.00",   # replace with heatmap peak confidence
            "label": name,
            "points": [float(x), float(y)],
            "type": "points",
        })

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
