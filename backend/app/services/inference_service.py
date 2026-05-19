"""
inference_service.py — Real-time inference engine for cephalometric landmark detection
======================================================================================

Extracts the complete preprocessing → model → TTA → decode pipeline from predict_all.py
and adapts it for synchronous FastAPI use.

GUARDRAIL: NO horizontal flip — cephalograms are anatomically directional.

Device priority: MPS (Apple Silicon) → CUDA → CPU (auto-detected at startup).
"""

from __future__ import annotations

import sys
from pathlib import Path
from io import BytesIO
from typing import Any

# ── project paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent   # /home/iddi/ceph-v2-auto
sys.path.insert(0, str(ROOT))

# ── stdlib ─────────────────────────────────────────────────────────────────────
import os
from io import BytesIO
from pathlib import Path
from typing import Any

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from src.phase2.model import CephalometricModel

# ── constants (mirrored from predict_all.py — do not change) ───────────────────
KEYPOINT_NAMES = [
    "Upper_tip", "Upper_apex", "Labial_midroot", "Labial_crest",
    "Palatal_midroot", "Palatal_crest", "ANS", "PNS", "LB", "PB",
]
INPUT_SIZE = (512, 512)    # (H, W) — model expects
HEATMAP_SIZE = (256, 256) # (H, W) — model outputs
NUM_KEYPOINTS = 10

# ── TTA variants (5 total: orig + rot± + brig±) ─────────────────────────────
# GUARDRAIL: NO horizontal flip — cephalograms are anatomically directional.
# NOTE: scale+ / scale- are EXCLUDED — geometric inverse produces ~100 px errors.


def _tta_identity(img: np.ndarray) -> np.ndarray:
    return img.copy()


def _inv_identity(coords_hm: np.ndarray, orig_size) -> np.ndarray:
    orig_h, orig_w = orig_size
    inp_w, inp_h = INPUT_SIZE[1], INPUT_SIZE[0]
    inp_coords = coords_hm * 2.0
    scale_x = orig_w / inp_w
    scale_y = orig_h / inp_h
    orig_coords = np.empty_like(inp_coords)
    orig_coords[:, 0] = inp_coords[:, 0] * scale_x
    orig_coords[:, 1] = inp_coords[:, 1] * scale_y
    return orig_coords


def _tta_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale=1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )


def _inv_rotate(angle_deg: float):
    def inv(coords_hm: np.ndarray, orig_size) -> np.ndarray:
        orig_h, orig_w = orig_size
        inp_w, inp_h = INPUT_SIZE[1], INPUT_SIZE[0]
        inp_coords = coords_hm * 2.0
        cx, cy = inp_w / 2, inp_h / 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, scale=1.0)
        x, y = inp_coords[:, 0], inp_coords[:, 1]
        inv_coords = np.empty_like(inp_coords)
        inv_coords[:, 0] = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        inv_coords[:, 1] = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        scale_x = orig_w / inp_w
        scale_y = orig_h / inp_h
        orig_coords = np.empty_like(inv_coords)
        orig_coords[:, 0] = inv_coords[:, 0] * scale_x
        orig_coords[:, 1] = inv_coords[:, 1] * scale_y
        return orig_coords
    return inv


def _tta_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


# NOTE: scale variants excluded — geometric inverse produces ~100 px errors.
TTA_VARIANTS = [
    ("orig",   _tta_identity,                                          _inv_identity),
    ("rot+",   lambda img: _tta_rotate(img,  2.0),                    _inv_rotate(2.0)),
    ("rot-",   lambda img: _tta_rotate(img, -2.0),                    _inv_rotate(-2.0)),
    ("brig+",  lambda img: _tta_brightness(img, 1.10),               _inv_identity),
    ("brig-",  lambda img: _tta_brightness(img, 0.90),               _inv_identity),
]
NUM_TTA = len(TTA_VARIANTS)  # 5

# ── decode helpers ─────────────────────────────────────────────────────────────


def _hard_argmax_decode(
    heatmaps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hard-argmax decode: integer (col=row=) peak for each keypoint channel.

    Args:
        heatmaps: [B, K, H_hm, W_hm] raw logits (no sigmoid inside here)

    Returns:
        coords:     [B, K, 2] float32 — (x=col, y=row) in heatmap pixel space
        confidence: [B, K] float32     — sigmoid(peak_activation)
    """
    B, K, H, W = heatmaps.shape
    conf = torch.sigmoid(heatmaps)
    flat_conf = conf.view(B * K, H * W)
    confidence, flat_idx = flat_conf.max(dim=-1)
    confidence = confidence.view(B, K)
    rows = (flat_idx // W).float()
    cols = (flat_idx % W).float()
    coords = torch.stack([cols, rows], dim=-1).view(B, K, 2)
    return coords, confidence


def _scale_coords_to_original(
    coords_hm: torch.Tensor,
    orig_size: tuple[int, int],
) -> torch.Tensor:
    """
    Map coordinates: heatmap [256×256] → input [512×512] → original image space.
    Each scale factor applied independently — valid for any input image size.
    """
    orig_h, orig_w = orig_size
    inp_h, inp_w = INPUT_SIZE

    scale_x_hm2inp = inp_w / HEATMAP_SIZE[1]
    scale_y_hm2inp = inp_h / HEATMAP_SIZE[0]

    coords_inp = coords_hm.clone()
    coords_inp[..., 0] *= scale_x_hm2inp
    coords_inp[..., 1] *= scale_y_hm2inp

    scale_x_inp2orig = orig_w / inp_w
    scale_y_inp2orig = orig_h / inp_h

    coords_orig = coords_inp.clone()
    coords_orig[..., 0] *= scale_x_inp2orig
    coords_orig[..., 1] *= scale_y_inp2orig
    return coords_orig


# ── InferenceService ───────────────────────────────────────────────────────────


class InferenceService:
    """
    Loads the HRNet-W32 checkpoint once at startup and runs inference on request.

    Exposes:
        predict(image_bytes: bytes) -> dict
            Runs 5-variant TTA and returns landmark coordinates + confidences.

    Device: MPS (Apple Silicon) → CUDA → CPU, auto-detected.
    Checkpoint: outputs/checkpoints/fold1_best.pth
    """

    CHECKPOINT_PATH = Path(
        os.environ.get(
            "MODEL_CHECKPOINT_DIR",
            str(ROOT / "outputs" / "checkpoints"),
        )
    ) / "fold1_best.pth"

    def __init__(self):
        self._device = self._detect_device()
        self._model: torch.nn.Module | None = None
        self._load_model()

    # ── device ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    # ── model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if not self.CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.CHECKPOINT_PATH}"
            )
        model = CephalometricModel(num_keypoints=NUM_KEYPOINTS, pretrained=False)
        ckpt = torch.load(
            self.CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model = model.to(self._device)
        model.eval()
        self._model = model
        print(f"[InferenceService] Model loaded on {self._device}")

    # ── image preprocessing ───────────────────────────────────────────────────

    @staticmethod
    def _bytes_to_rgb_ndarray(image_bytes: bytes) -> np.ndarray:
        """
        Decode image bytes to RGB ndarray (H, W, 3) uint8.
        Handles JPEG, PNG, BMP, TIFF, WebP via PIL → OpenCV.
        """
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil_img)  # PIL already returns RGB
        return rgb

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """
        Resize to INPUT_SIZE and normalize to [0, 1] float32 tensor.

        NOTE: Simple /255 normalization only.
        DO NOT use ImageNet mean/std — model was trained with /255.
        """
        img_resized = cv2.resize(
            rgb,
            (INPUT_SIZE[1], INPUT_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(tensor).unsqueeze(0)  # [1, 3, 512, 512]

    # ── single-pass inference ───────────────────────────────────────────────────

    @torch.no_grad()
    def _predict_single(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run single-pass (no TTA) inference on the image.

        Returns:
            coords_orig: [K, 2] in original image pixel coordinates
            confs:       [K] confidence scores [0, 1]
        """
        assert self._model is not None
        tensor = self._preprocess(rgb).to(self._device)
        orig_h, orig_w = rgb.shape[:2]

        out = self._model(tensor)
        heatmaps = out[0] if isinstance(out, tuple) else out
        if heatmaps.shape[-2:] != HEATMAP_SIZE:
            heatmaps = F.interpolate(
                heatmaps,
                size=HEATMAP_SIZE,
                mode="bilinear",
                align_corners=False,
            )
        coords_hm, confs = _hard_argmax_decode(heatmaps.cpu())
        coords_orig = _scale_coords_to_original(coords_hm, (orig_h, orig_w))
        return coords_orig[0].numpy(), confs[0].numpy()

    # ── TTA inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _predict_tta(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run 5-variant TTA (orig + rot± + brig±) and average coordinates.

        GUARDRAIL: NO horizontal flip — cephalograms are anatomically directional.
        Scale variants excluded — geometric inverse produces ~100 px errors.
        """
        assert self._model is not None
        orig_h, orig_w = rgb.shape[:2]

        all_coords: list[np.ndarray] = []
        all_confs: list[np.ndarray] = []

        for tta_name, transform_fn, inverse_fn in TTA_VARIANTS:
            img_aug = transform_fn(rgb)
            tensor = self._preprocess(img_aug).to(self._device)

            out = self._model(tensor)
            heatmaps = out[0] if isinstance(out, tuple) else out
            if heatmaps.shape[-2:] != HEATMAP_SIZE:
                heatmaps = F.interpolate(
                    heatmaps,
                    size=HEATMAP_SIZE,
                    mode="bilinear",
                    align_corners=False,
                )

            coords_hm, confs_tensor = _hard_argmax_decode(heatmaps.cpu())
            coords_hm_np = coords_hm[0].numpy()  # [K, 2] in heatmap [256,256] space
            conf_np = confs_tensor[0].numpy()    # [K]

            coords_orig_np = inverse_fn(coords_hm_np, (orig_h, orig_w))
            all_coords.append(coords_orig_np)
            all_confs.append(conf_np)

        avg_coords = np.mean(all_coords, axis=0)
        avg_conf = np.mean(all_confs, axis=0)
        return avg_coords, avg_conf

    # ── public API ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image_bytes: bytes, use_tta: bool = True) -> dict[str, Any]:
        """
        Main entry point from the FastAPI endpoint.

        Args:
            image_bytes: raw uploaded image (JPEG, PNG, etc.)
            use_tta:     if True (default), use 5-variant TTA; else single-pass

        Returns:
            dict with keys:
                landmarks: {name: {x, y, confidence}, ...} for all 10 keypoints
                use_tta:   bool
        """
        rgb = self._bytes_to_rgb_ndarray(image_bytes)

        if use_tta:
            xy, conf = self._predict_tta(rgb)
        else:
            xy, conf = self._predict_single(rgb)

        landmarks: dict[str, Any] = {}
        for k, name in enumerate(KEYPOINT_NAMES):
            landmarks[name] = {
                "x": round(float(xy[k, 0]), 1),
                "y": round(float(xy[k, 1]), 1),
                "confidence": round(float(conf[k]), 3),
            }

        return {
            "landmarks": landmarks,
            "use_tta": use_tta,
        }