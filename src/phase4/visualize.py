"""Generate tracing overlay image: T1 landmarks (blue) + T2 landmarks (red) on T2 image."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


T1_COLOR = (255, 100, 100)   # blue-ish (BGR: 100, 100, 255 for OpenCV)
T2_COLOR = (100, 100, 255)   # red-ish
T1_COLOR_BGR = (100, 100, 255)
T2_COLOR_BGR = (100, 255, 100)
ARROW_COLOR = (50, 200, 50)
LOW_CONF_COLOR = (0, 0, 255)  # red dot warning

KEYPOINT_RADIUS = 8
ARROW_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def draw_tracing_overlay(
    image_t2_path: str,
    keypoints_t1: np.ndarray,
    keypoints_t2: np.ndarray,
    valid_t1: np.ndarray,
    valid_t2: np.ndarray,
    keypoint_names: list[str],
    low_confidence_landmarks: Optional[list[str]] = None,
    scale_to_width: int = 1024,
) -> np.ndarray:
    """
    Draw T1 (blue) and T2 (red) landmarks on the T2 image with connecting arrows.

    Args:
        image_t2_path: path to T2 JPG
        keypoints_t1, keypoints_t2: [N, 2] in original image pixel space
        valid_t1, valid_t2: [N] bool
        low_confidence_landmarks: names to mark with warning color

    Returns:
        overlay: BGR uint8 numpy array
    """
    img = cv2.imread(image_t2_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_t2_path}")

    orig_h, orig_w = img.shape[:2]
    if scale_to_width and orig_w > scale_to_width:
        scale = scale_to_width / orig_w
        new_w = scale_to_width
        new_h = int(orig_h * scale)
        img = cv2.resize(img, (new_w, new_h))
        keypoints_t1 = keypoints_t1 * scale
        keypoints_t2 = keypoints_t2 * scale

    low_conf_set = set(low_confidence_landmarks or [])
    overlay = img.copy()

    for i, name in enumerate(keypoint_names):
        is_low_conf = name in low_conf_set

        if valid_t1[i]:
            pt1 = tuple(keypoints_t1[i].astype(int))
            color = LOW_CONF_COLOR if is_low_conf else T1_COLOR_BGR
            cv2.circle(overlay, pt1, KEYPOINT_RADIUS, color, -1)
            cv2.putText(overlay, f"T1:{name[:6]}", (pt1[0] + 5, pt1[1] - 5),
                        FONT, FONT_SCALE, color, FONT_THICKNESS)

        if valid_t2[i]:
            pt2 = tuple(keypoints_t2[i].astype(int))
            color = LOW_CONF_COLOR if is_low_conf else T2_COLOR_BGR
            cv2.circle(overlay, pt2, KEYPOINT_RADIUS, color, -1)
            cv2.putText(overlay, f"T2:{name[:6]}", (pt2[0] + 5, pt2[1] + 15),
                        FONT, FONT_SCALE, color, FONT_THICKNESS)

        if valid_t1[i] and valid_t2[i]:
            pt1 = tuple(keypoints_t1[i].astype(int))
            pt2 = tuple(keypoints_t2[i].astype(int))
            cv2.arrowedLine(overlay, pt1, pt2, ARROW_COLOR, ARROW_THICKNESS, tipLength=0.3)

    return overlay


def save_overlay(overlay: np.ndarray, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, overlay)
