"""
Singdent Cephalometric Analysis — MVP Dashboard
================================================
Streamlit web application for demonstrating the cephalometric landmark
detection and biomechanics classification pipeline to clinical stakeholders.

Run with:
    streamlit run app.py
"""

import time
from io import BytesIO
from typing import Dict, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.phase3.biomechanics import (
    calculate_metrics,
    classify_treatment,
    mock_landmarks,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Singdent | Cephalometric AI",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Singdent Cephalometric Landmark Detection — Research Prototype",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS — medical dark theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0d1b2a 0%, #1b2e45 100%);
    }

    /* ── Hero banner ─────────────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 60%, #0d4f7a 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #e0f2fe;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #7dd3fc;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38bdf8;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 999px;
        margin-bottom: 0.75rem;
    }

    /* ── Section cards ───────────────────────────────────────────────────── */
    .report-card {
        background: linear-gradient(135deg, #0f1f30 0%, #1a2f45 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .report-card h4 {
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 0 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    }

    /* ── Landmark pill ───────────────────────────────────────────────────── */
    .lm-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.72rem;
        color: #bae6fd;
        margin: 3px;
    }
    .lm-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    /* ── Classification result ───────────────────────────────────────────── */
    .class-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f0fdf4;
    }
    .avoid-label {
        color: #fca5a5;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .prefer-label {
        color: #86efac;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .implication-label {
        color: #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.6;
        font-style: italic;
    }

    /* ── Metric override ─────────────────────────────────────────────────── */
    [data-testid="stMetricLabel"] { color: #7dd3fc !important; }
    [data-testid="stMetricValue"] { color: #e0f2fe !important; font-weight: 600; }

    /* ── Divider ─────────────────────────────────────────────────────────── */
    hr { border-color: rgba(56, 189, 248, 0.12) !important; }

    /* ── Upload area ─────────────────────────────────────────────────────── */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed rgba(56, 189, 248, 0.35) !important;
        border-radius: 12px !important;
        background: rgba(13, 27, 42, 0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants — landmark render palette
# ---------------------------------------------------------------------------

# BGR colours for OpenCV drawing
_LANDMARK_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "Upper_tip":        (255, 220,  50),   # gold
    "Upper_apex":       (255, 140,   0),   # orange
    "ANS":              ( 50, 220, 255),   # cyan
    "PNS":              ( 50, 180, 255),   # sky
    "LB":               ( 80, 255, 120),   # green
    "PB":               (180, 255,  80),   # lime
    "Palatal_crest":    (220,  80, 255),   # violet
    "Labial_crest":     (255,  80, 200),   # pink
    "Labial_midroot":  (255, 120, 120),   # salmon
    "Palatal_midroot": (120, 120, 255),   # lavender
}

# Reference frame the mock landmarks were authored for (~800 × ~830 px region)
_REF_WIDTH  = 830.0
_REF_HEIGHT = 830.0

# Approximate centre of the landmark cloud in the reference frame
# (used to re-centre after scaling)
_REF_CX = 390.0
_REF_CY = 415.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_landmarks(
    raw: Dict[str, Tuple[float, float]],
    img_w: int,
    img_h: int,
) -> Dict[str, Tuple[float, float]]:
    """Scale mock landmark pixel coordinates to fit the uploaded image.

    Strategy
    --------
    1. Compute a uniform scale factor (min of x and y ratios) so the whole
       landmark cloud fits inside the image.
    2. Shift the scaled cloud to the image centre.
    """
    scale = min(img_w / _REF_WIDTH, img_h / _REF_HEIGHT) * 0.85   # 85% of available space
    cx, cy = img_w / 2.0, img_h / 2.0

    scaled: Dict[str, Tuple[float, float]] = {}
    for name, (x, y) in raw.items():
        sx = (x - _REF_CX) * scale + cx
        sy = (y - _REF_CY) * scale + cy
        scaled[name] = (sx, sy)
    return scaled


def _draw_landmarks(
    img_rgb: np.ndarray,
    landmarks: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Draw landmarks + key lines onto a copy of img_rgb (RGB uint8)."""
    vis = img_rgb.copy()

    # ── Palatal plane (ANS → PNS) ─────────────────────────────────────────
    if "ANS" in landmarks and "PNS" in landmarks:
        p1 = (int(landmarks["ANS"][0]),  int(landmarks["ANS"][1]))
        p2 = (int(landmarks["PNS"][0]),  int(landmarks["PNS"][1]))
        cv2.line(vis, p1, p2, (50, 210, 255), 2, cv2.LINE_AA)

    # ── U1 long axis (Upper_apex → Upper_tip) ─────────────────────────────
    if "Upper_tip" in landmarks and "Upper_apex" in landmarks:
        p1 = (int(landmarks["Upper_apex"][0]), int(landmarks["Upper_apex"][1]))
        p2 = (int(landmarks["Upper_tip"][0]),  int(landmarks["Upper_tip"][1]))
        cv2.line(vis, p1, p2, (255, 210, 50), 2, cv2.LINE_AA)

    # ── LB → PB connector (shows apex corridor) ───────────────────────────
    if "LB" in landmarks and "PB" in landmarks:
        p1 = (int(landmarks["LB"][0]), int(landmarks["LB"][1]))
        p2 = (int(landmarks["PB"][0]), int(landmarks["PB"][1]))
        cv2.line(vis, p1, p2, (150, 255, 150), 1, cv2.LINE_AA)

    # ── Landmark dots + labels ────────────────────────────────────────────
    for name, (x, y) in landmarks.items():
        colour = _LANDMARK_COLOURS.get(name, (255, 255, 255))
        # BGR for cv2 — colours dict stores BGR already
        cx, cy_ = int(x), int(y)
        cv2.circle(vis, (cx, cy_), 7, colour, -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy_), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Short label (first word only to reduce clutter)
        short = name.split()[0]
        cv2.putText(
            vis, short,
            (cx + 10, cy_ - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            colour, 1, cv2.LINE_AA,
        )

    return vis


def _apex_position_colour(pos: str) -> str:
    return {"Labial": "🔴", "Midway": "🟢", "Palatal": "🟡"}.get(pos, "⚪")


def _angle_zone_label(angle: float) -> str:
    if angle < 105:
        return f"⬇ {angle:.1f}° — Retroclined  (< 105°)"
    elif angle <= 115:
        return f"✅ {angle:.1f}° — Normal  (105–115°)"
    else:
        return f"⬆ {angle:.1f}° — Proclined  (> 115°)"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0;">
          <div style="font-size:2.5rem;">🦷</div>
          <div style="color:#7dd3fc; font-weight:700; font-size:1rem; letter-spacing:0.5px;">
            SINGDENT
          </div>
          <div style="color:#475569; font-size:0.72rem; letter-spacing:1px;
               text-transform:uppercase; margin-top:2px;">
            Cephalometric AI · Research
          </div>
        </div>
        <hr style="border-color:rgba(56,189,248,0.15); margin: 0.5rem 0 1.5rem 0;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Upload a Cephalometric X-Ray**")
    uploaded_file = st.file_uploader(
        label="Drag & drop or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model / calibration info
    st.markdown("**Calibration**")
    mm_per_pixel = st.number_input(
        "mm / pixel",
        value=0.0984,
        min_value=0.05,
        max_value=0.30,
        step=0.001,
        format="%.4f",
        help="From calibration.csv — dataset mean is 0.0984 mm/px",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#475569; font-size:0.72rem; line-height:1.7;">
          <b style="color:#7dd3fc;">Model:</b> HRNet-W32 (scaffold)<br>
          <b style="color:#7dd3fc;">Keypoints:</b> 10 landmarks<br>
          <b style="color:#7dd3fc;">Logic:</b> Zhang et al. 2021<br>
          <b style="color:#7dd3fc;">Status:</b> ⚠ Mock inference
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
      <div class="hero-badge">Research Prototype · v0.1</div>
      <p class="hero-title">🦷 Cephalometric Analysis Dashboard</p>
      <p class="hero-subtitle">
        Upload a lateral cephalometric radiograph to detect 10 anatomical landmarks
        and receive an automated biomechanical treatment classification
        based on Zhang et al. 2021.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

if uploaded_file is None:
    # ── Empty state ────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown(
            """
            <div style="
                height: 380px;
                border: 2px dashed rgba(56,189,248,0.25);
                border-radius: 16px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: #475569;
                background: rgba(13,27,42,0.4);
            ">
              <div style="font-size:3rem; margin-bottom:0.5rem;">🩻</div>
              <div style="font-size:0.9rem; font-weight:500;">
                No image uploaded yet
              </div>
              <div style="font-size:0.75rem; margin-top:0.25rem;">
                Use the sidebar to upload a cephalometric X-ray
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            """
            <div class="report-card">
              <h4>How it works</h4>
              <ol style="color:#cbd5e1; font-size:0.85rem; line-height:2;">
                <li>Upload a lateral cephalometric radiograph (JPG / PNG)</li>
                <li>The AI model detects <strong style="color:#7dd3fc;">10 landmarks</strong>
                    including incisors, ANS, PNS, LB and PB</li>
                <li>U1-PP angle and root apex distances are calculated</li>
                <li>A biomechanical treatment classification is generated
                    per <strong style="color:#7dd3fc;">Zhang et al. 2021</strong></li>
              </ol>
            </div>
            <div class="report-card">
              <h4>Landmark legend</h4>
              <div style="display:flex; flex-wrap:wrap; gap:4px;">
            """,
            unsafe_allow_html=True,
        )
        for name, (b, g, r) in _LANDMARK_COLOURS.items():
            hex_col = f"#{r:02x}{g:02x}{b:02x}"
            st.markdown(
                f'<span class="lm-pill">'
                f'<span class="lm-dot" style="background:{hex_col};"></span>'
                f'{name}'
                f'</span>',
                unsafe_allow_html=True,
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

else:
    # ── Image processing ───────────────────────────────────────────────────

    # Load the uploaded image
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)          # RGB uint8
    img_h, img_w = img_array.shape[:2]

    # ── Mock inference with spinner ────────────────────────────────────────
    with st.spinner("🤖  AI is analysing cephalometric landmarks…"):
        time.sleep(2)
        raw_landmarks = mock_landmarks()

    st.success("✅  Landmark detection complete  —  10 / 10 keypoints found")

    # ── Scale landmarks to the uploaded image ──────────────────────────────
    scaled_landmarks = _scale_landmarks(raw_landmarks, img_w, img_h)

    # ── Draw on image ──────────────────────────────────────────────────────
    annotated = _draw_landmarks(img_array, scaled_landmarks)

    # ── Calculate metrics (scaled coords match the real image + mm/px) ────
    metrics = calculate_metrics(scaled_landmarks, mm_per_pixel=mm_per_pixel)
    classification = classify_treatment(
        u1_pp_angle  = metrics["u1_pp_angle_deg"],
        lb_apex_dist = metrics["lb_apex_dist_mm"],
        pb_apex_dist = metrics["pb_apex_dist_mm"],
    )

    # ── Layout: image left, report right ──────────────────────────────────
    col_img, col_report = st.columns([1.1, 1], gap="large")

    with col_img:
        st.markdown("#### 🩻 Annotated Radiograph")
        st.image(
            annotated,
            caption=f"Detected 10 landmarks  ·  {img_w} × {img_h} px  "
                    f"·  {mm_per_pixel:.4f} mm/px",
            use_container_width=True,
        )

        # Landmark coordinate table
        with st.expander("📍 Raw landmark coordinates (mock)", expanded=False):
            lm_rows = []
            for name in raw_landmarks:
                x, y = raw_landmarks[name]
                sx, sy = scaled_landmarks[name]
                lm_rows.append(
                    f"| {name} | {x:.1f} | {y:.1f} | {sx:.1f} | {sy:.1f} |"
                )
            table_md = (
                "| Landmark | Mock X | Mock Y | Scaled X | Scaled Y |\n"
                "|---|---|---|---|---|\n"
                + "\n".join(lm_rows)
            )
            st.markdown(table_md)

    with col_report:
        st.markdown("#### 📋 Clinical Report")

        # ── Metrics ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="report-card"><h4>Measured Metrics</h4>',
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "U1-PP Angle",
            f"{metrics['u1_pp_angle_deg']:.1f}°",
            help="Angle between upper incisor long axis and palatal plane",
        )
        m2.metric(
            "LB-Apex",
            f"{metrics['lb_apex_dist_mm']:.2f} mm",
            help="Distance from labial bone landmark to root apex",
        )
        m3.metric(
            "PB-Apex",
            f"{metrics['pb_apex_dist_mm']:.2f} mm",
            help="Distance from palatal bone landmark to root apex",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Root apex position ────────────────────────────────────────────
        apex_pos = classification["Root apex position"]
        apex_icon = _apex_position_colour(apex_pos)
        st.markdown(
            f'<div class="report-card"><h4>Root Apex Position</h4>'
            f'<span class="class-label">{apex_icon}  {apex_pos}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Incisor condition + angle zone ────────────────────────────────
        st.markdown(
            f'<div class="report-card"><h4>Incisor Condition</h4>'
            f'<div style="color:#e2e8f0; font-size:0.9rem; margin-bottom:0.5rem;">'
            f'{classification["Incisor condition"]}</div>'
            f'<div style="color:#94a3b8; font-size:0.8rem;">'
            f'{_angle_zone_label(metrics["u1_pp_angle_deg"])}'
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Preferred biomechanics ────────────────────────────────────────
        st.markdown(
            f'<div class="report-card"><h4>Preferred Biomechanics</h4>'
            f'<div class="prefer-label">✅  {classification["Preferred biomechanics"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Biomechanics to avoid ─────────────────────────────────────────
        st.markdown(
            f'<div class="report-card"><h4>Biomechanics to Avoid</h4>'
            f'<div class="avoid-label">⚠  {classification["Biomechanics to avoid"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Clinical implication ──────────────────────────────────────────
        st.markdown(
            f'<div class="report-card"><h4>Clinical Implication</h4>'
            f'<div class="implication-label">{classification["Clinical implication"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Disclaimer ────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="
                background: rgba(234,179,8,0.08);
                border: 1px solid rgba(234,179,8,0.3);
                border-radius: 8px;
                padding: 0.75rem 1rem;
                font-size: 0.72rem;
                color: #a3a3a3;
                margin-top: 0.5rem;
            ">
            ⚠ <strong style="color:#fbbf24;">Research prototype.</strong>
            Landmark positions are mock data — not from a trained model.
            All clinical decisions must be validated by a qualified orthodontist.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Debug / raw JSON ───────────────────────────────────────────────────
    with st.expander("🔧 Raw classification output (debug)", expanded=False):
        st.json({
            "metrics": metrics,
            "classification": classification,
            "image_size": {"width": img_w, "height": img_h},
            "mm_per_pixel": mm_per_pixel,
        })
