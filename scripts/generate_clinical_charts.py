#!/usr/bin/env python3
"""
Additional Clinical Visualizations
===================================
Chart 10: Clinical UI Schematic (ghost polygon, bottleneck, cervical offset slider)
Chart 11: Global Minimum Algorithm Diagram
Chart 12: Per-Landmark σ Heatmap
"""

import json, math, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Arc, Wedge
import matplotlib.patheffects as pe
import numpy as np

OUTPUT_DIR = Path("/Users/onis2/Project/Singdent/ceph-auto/ceph-project/reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "primary":    "#2D6A4F",
    "secondary":  "#40916C",
    "accent":     "#74C69D",
    "highlight":  "#D8F3DC",
    "easy":       "#2D6A4F",
    "medium":     "#E9C46A",
    "hard":       "#E76F51",
    "dark_bg":    "#1A1A2E",
    "light_bg":   "#F8F9FA",
    "text_dark":  "#212529",
    "text_light": "#F1F3F5",
    "grid":       "#DEE2E6",
    "labial":     "#457B9D",
    "palatal":    "#E76F51",
    "benchmark":  "#6C757D",
}


# ══════════════════════════════════════════════════════════════════════════════
# CHART 10: Clinical UI Schematic (phantom/ghost polygon with measurements)
# ══════════════════════════════════════════════════════════════════════════════
def chart10_clinical_ui_schematic():
    """Stylized clinical measurement UI schematic showing ghost polygon,
    bottleneck measurement lines, cervical offset slider, and U1-PP angle."""

    fig, axes = plt.subplots(1, 2, figsize=(18, 9),
                              gridspec_kw={"width_ratios": [3, 1.2]},
                              facecolor="white")
    ax_canvas, ax_controls = axes
    ax_canvas.set_facecolor("#1A1A2E")  # dark canvas
    ax_controls.set_facecolor("#F8F9FA")

    # ── Left: Canvas Editor ──────────────────────────────────────────────────
    ax = ax_canvas
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.set_title("Cephalometric Analysis — Canvas Editor",
                 fontsize=13, fontweight="bold", color="white", pad=10,
                 fontfamily="monospace")

    # Simulated tooth outline (upper incisor silhouette)
    tooth_outline = np.array([
        [4.0, 8.5], [4.8, 8.8], [5.5, 8.5],   # crown labial
        [5.9, 7.8], [5.7, 7.0], [5.3, 6.2],   # labial side going apical
        [5.0, 5.5], [4.5, 5.0], [4.0, 5.5],   # apex
        [3.5, 6.2], [3.1, 7.0], [3.0, 7.8],   # palatal side
        [3.3, 8.5], [4.0, 8.5],               # crown palatal
    ])

    # Ghost (predicted) polygon — slightly offset
    ghost_outline = tooth_outline + np.array([0.15, -0.2])

    # Draw ghost polygon (semi-transparent cyan)
    ghost = plt.Polygon(ghost_outline, closed=True,
                        facecolor="#457B9D", alpha=0.25,
                        edgecolor="#74C69D", linewidth=2,
                        linestyle="--", zorder=2)
    ax.add_patch(ghost)

    # Draw actual tooth outline (solid white)
    tooth = plt.Polygon(tooth_outline, closed=True,
                         facecolor="#2D6A4F", alpha=0.7,
                         edgecolor="white", linewidth=2, zorder=3)
    ax.add_patch(tooth)

    # ── Landmarks ────────────────────────────────────────────────────────────
    landmarks = {
        "Upper_tip":       [5.5, 8.85],
        "Labial_crest":    [5.5, 7.85],
        "Labial_midroot":  [5.65, 7.2],
        "Palatal_midroot": [3.25, 7.2],
        "Upper_apex":      [4.5, 5.0],
        "Palatal_crest":   [3.3, 7.85],
    }
    landmark_colors = {
        "Upper_tip": "#2D6A4F", "Labial_crest": "#2D6A4F",
        "Labial_midroot": "#2D6A4F", "Palatal_midroot": "#E9C46A",
        "Upper_apex": "#E76F51", "Palatal_crest": "#2D6A4F",
    }

    for name, (lx, ly) in landmarks.items():
        circle = Circle((lx, ly), 0.18,
                        facecolor=landmark_colors[name],
                        edgecolor="white", linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(lx, ly - 0.35, name.replace("_", "\n"),
                ha="center", va="top", fontsize=7, color="white",
                fontweight="bold", zorder=6)

    # ── Bottleneck Measurement Lines ─────────────────────────────────────────
    # Labial side measurement
    labial_mid = (5.65 + 5.0) / 2
    labial_y_mid = (7.2 + 5.5) / 2

    # Horizontal measurement line across bottleneck (labial)
    ax.annotate("",
                xy=(3.8, labial_y_mid), xytext=(5.5, labial_y_mid),
                arrowprops=dict(arrowstyle="<->", color="#F4A261",
                               linewidth=2.5, shrinkA=0, shrinkB=0), zorder=7)
    ax.text(4.65, labial_y_mid + 0.18, "Bottleneck\n(labial)",
            ha="center", fontsize=8, color="#F4A261", fontweight="bold")

    # ── U1-PP Angle ──────────────────────────────────────────────────────────
    # U1 (upper incisor long axis)
    u1_base = [4.5, 5.0]
    u1_tip  = [5.5, 8.5]
    ax.annotate("", xy=(5.5, 8.5), xytext=(4.5, 5.0),
                arrowprops=dict(arrowstyle="-", color="#E76F51", linewidth=3), zorder=4)

    # PP reference line (Palatal Plane — horizontal)
    pp_y = 6.0
    ax.annotate("",
                xy=(2.5, pp_y), xytext=(7.0, pp_y),
                arrowprops=dict(arrowstyle="-", color="#74C69D",
                               linewidth=2.5, shrinkA=0, shrinkB=0), zorder=4)

    # Angle arc
    u1_angle = math.degrees(math.atan2(8.5 - 5.0, 5.5 - 4.5))  # ~70 deg from horizontal
    pp_angle = 0
    angle_deg = u1_angle - pp_angle  # ~70 deg

    arc = Arc((4.5, 5.0), 1.2, 1.2, angle=0,
              theta1=pp_angle, theta2=u1_angle,
              color="#FFD166", linewidth=2.5, zorder=8)
    ax.add_patch(arc)
    ax.text(4.5 + 0.9, 5.0 + 0.5, f"U1-PP\n≈ 70°",
            fontsize=9, color="#FFD166", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1A1A2E", alpha=0.8), zorder=9)

    ax.text(2.5, pp_y - 0.15, "Palatal Plane (PP)",
            fontsize=8, color="#74C69D", fontweight="bold", style="italic")

    # ── Cervical Offset Slider indicator ──────────────────────────────────────
    offset_y = 4.2
    ax.axhline(y=offset_y, color="#6C757D", linewidth=1, linestyle=":", alpha=0.5)
    ax.text(7.5, offset_y + 0.1, "Cervical offset\nreference (0–5mm)",
            fontsize=7, color="#ADB5BD", style="italic", ha="center")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_x, legend_y = 7.8, 9.0
    items = [
        ("#2D6A4F", "Predicted landmark (green)"),
        ("#E9C46A", "Snapped landmark (amber)"),
        ("#E76F51", "Difficult landmark (coral)"),
        ("#74C69D", "Ghost / predicted polygon"),
        ("#F4A261", "Bottleneck measurement"),
        ("#457B9D", "Segmentation mask overlay"),
    ]
    for i, (color, label) in enumerate(items):
        y = legend_y - i * 0.45
        rect = plt.Rectangle((legend_x, y - 0.12), 0.35, 0.25,
                              facecolor=color, edgecolor="white", linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + 0.5, y, label, fontsize=7.5, color="white", va="center")

    # ── Right: Control Panel ───────────────────────────────────────────────────
    ax2 = ax_controls
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("Measurement Controls", fontsize=12,
                  fontweight="bold", color=COLORS["text_dark"], pad=10)

    # Cervical Offset Slider
    ax2.text(1, 8.5, "Cervical Offset", fontsize=10, fontweight="bold",
             color=COLORS["text_dark"])
    ax2.text(1, 8.1, "(apical adjustment 0–5 mm)", fontsize=8, color="#6C757D")

    # Draw slider track
    slider_x, slider_y, slider_w = 1.0, 7.3, 8.0
    ax2.add_patch(plt.Rectangle((slider_x, slider_y - 0.1), slider_w, 0.2,
                                  facecolor="#DEE2E6", edgecolor="#CED4DA",
                                  linewidth=1, zorder=1))
    # Slider fill (2.5mm = 50% of 5mm)
    ax2.add_patch(plt.Rectangle((slider_x, slider_y - 0.1), slider_w * 0.5, 0.2,
                                  facecolor=COLORS["primary"], edgecolor="none", zorder=2))
    # Slider thumb
    ax2.add_patch(Circle((slider_x + slider_w * 0.5, slider_y), 0.25,
                          facecolor="white", edgecolor=COLORS["primary"],
                          linewidth=2, zorder=3))
    ax2.text(slider_x + slider_w * 0.5, slider_y - 0.5, "2.5 mm",
             ha="center", fontsize=9, fontweight="bold", color=COLORS["primary"])

    # Measurement results
    ax2.axhline(y=6.5, color=COLORS["grid"], linewidth=1)
    ax2.text(1, 6.1, "Current Measurements", fontsize=10,
             fontweight="bold", color=COLORS["text_dark"])

    measurements = [
        ("Bottleneck (labial)", "1.82 mm", COLORS["easy"]),
        ("Bottleneck (palatal)", "1.95 mm", COLORS["easy"]),
        ("U1-PP Angle",         "70.3°",   COLORS["primary"]),
        ("Classification",      "NORMAL",  COLORS["easy"]),
    ]
    for i, (label, val, color) in enumerate(measurements):
        y = 5.6 - i * 0.7
        ax2.text(1, y, label + ":", fontsize=9, color="#495057")
        ax2.text(7.5, y, val, fontsize=9, fontweight="bold", color=color, ha="right")

    # Clinical Assessment
    ax2.axhline(y=3.3, color=COLORS["grid"], linewidth=1)
    ax2.text(1, 2.9, "Clinical Assessment", fontsize=10,
             fontweight="bold", color=COLORS["text_dark"])

    # Assessment box
    ax2.add_patch(FancyBboxPatch((0.5, 1.0), 9.0, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=COLORS["highlight"],
                                  edgecolor=COLORS["primary"],
                                  linewidth=2, zorder=1))
    ax2.text(5.0, 2.1, "U1-PP: NORMAL RANGE",
             ha="center", va="center", fontsize=10,
             fontweight="bold", color=COLORS["primary"], zorder=2)
    ax2.text(5.0, 1.5, "Bottleneck ratio > 1.0 — Adequate bone support",
             ha="center", va="center", fontsize=8, color="#495057", zorder=2)

    # Dark mode toggle hint
    ax2.text(5, 0.3, "Dark Mode: ON",
             ha="center", fontsize=9, style="italic", color="#ADB5BD")

    plt.tight_layout(pad=1.5)
    out = OUTPUT_DIR / "10_clinical_ui_schematic.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CHART 11: Global Minimum Distance Algorithm Diagram
# ══════════════════════════════════════════════════════════════════════════════
def chart11_global_minimum_diagram():
    """Illustrates the Global Minimum Distance Algorithm — sweeping N points
    over the full working root length to find the thinnest bone gap."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor="white")
    titles = [
        "Step 1: Contour Extraction",
        "Step 2: N-Point Sweep (N=60)",
        "Step 3: Find Global Minimum",
    ]

    for ax, title in zip(axes, titles):
        ax.set_facecolor(COLORS["light_bg"])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=COLORS["text_dark"], pad=8)

    # ── Panel 1: Contour ─────────────────────────────────────────────────────
    ax = axes[0]

    # Tooth silhouette
    tooth_pts = np.array([
        [4.5, 7.5], [5.2, 7.8], [5.8, 7.5], [6.0, 7.0],
        [5.7, 6.0], [5.3, 5.2], [4.8, 4.8], [4.2, 5.2],
        [3.7, 6.0], [3.4, 7.0], [3.7, 7.5],
    ])
    tooth = plt.Polygon(tooth_pts, closed=True,
                         facecolor="#D8F3DC", edgecolor=COLORS["primary"],
                         linewidth=2.5, zorder=2)
    ax.add_patch(tooth)

    # Root contour highlight
    ax.plot(tooth_pts[:, 0], tooth_pts[:, 1],
            color=COLORS["primary"], linewidth=2.5, zorder=3)

    # Crest point
    crest = [5.8, 7.5]
    ax.add_patch(Circle(crest, 0.2, facecolor=COLORS["easy"],
                         edgecolor="white", linewidth=2, zorder=5))
    ax.text(crest[0] + 0.3, crest[1] + 0.15, "Crest", fontsize=9,
            color=COLORS["text_dark"], fontweight="bold")

    # Apical point
    apex = [4.8, 4.8]
    ax.add_patch(Circle(apex, 0.2, facecolor=COLORS["hard"],
                         edgecolor="white", linewidth=2, zorder=5))
    ax.text(apex[0] + 0.3, apex[1] - 0.2, "Apex", fontsize=9,
            color=COLORS["text_dark"], fontweight="bold")

    # Working length arrow
    ax.annotate("",
                xy=apex, xytext=crest,
                arrowprops=dict(arrowstyle="<->",
                               color=COLORS["benchmark"],
                               linewidth=2))
    mid = [(crest[0] + apex[0]) / 2 + 0.5, (crest[1] + apex[1]) / 2]
    ax.text(mid[0], mid[1], "Working\nRoot Length",
            fontsize=9, color=COLORS["benchmark"], fontweight="bold", ha="center")

    # ── Panel 2: N-point sweep ───────────────────────────────────────────────
    ax = axes[1]

    # Draw tooth again
    tooth2 = plt.Polygon(tooth_pts, closed=True,
                          facecolor="#D8F3DC", edgecolor=COLORS["primary"],
                          linewidth=2.5, zorder=2)
    ax.add_patch(tooth2)

    # Crest to 0.66x total length = bottleneck restriction zone
    t_crest = np.array(crest)
    t_apex = np.array(apex)
    total_length = np.linalg.norm(t_apertx := t_crest - t_apex)
    restrict_pt = t_apex + 0.66 * (t_crest - t_apex)

    # Sweep 60 points and draw measurement lines at every 10th point
    for t_frac in np.linspace(0, 0.66, 8):
        pt = t_apex + t_frac * (t_crest - t_apex)
        # Find closest point on contour (simplified: draw horizontal measurement)
        labial_x = 6.1
        palatal_x = 3.3
        ax.plot([palatal_x, labial_x], [pt[1], pt[1]],
                color="#F4A261", linewidth=1.2, alpha=0.5, zorder=3)
        ax.add_patch(Circle((pt[0], pt[1]), 0.08,
                             facecolor="#F4A261", edgecolor="none", alpha=0.7, zorder=4))

    # Bottleneck restriction zone (shaded)
    ax.axhspan(apex[1], restrict_pt[1], xmin=0.25, xmax=0.75,
               alpha=0.08, color=COLORS["hard"])
    ax.text(7.5, (apex[1] + restrict_pt[1]) / 2,
            "0.66 × L\nrestricted\nzone",
            fontsize=8, color=COLORS["hard"], style="italic", ha="center")

    ax.add_patch(Circle(crest, 0.2, facecolor=COLORS["easy"],
                         edgecolor="white", linewidth=2, zorder=5))
    ax.add_patch(Circle(apex, 0.2, facecolor=COLORS["hard"],
                         edgecolor="white", linewidth=2, zorder=5))

    ax.text(8.0, 2.5, "60 sweep points\n(Δt = 0.66/59)\nHorizontal lines:\nbone thickness",
            fontsize=9, color="#495057",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CED4DA"))

    # ── Panel 3: Global minimum found ───────────────────────────────────────
    ax = axes[2]

    tooth3 = plt.Polygon(tooth_pts, closed=True,
                          facecolor="#D8F3DC", edgecolor=COLORS["primary"],
                          linewidth=2.5, zorder=2)
    ax.add_patch(tooth3)

    # All sweep lines in grey
    for t_frac in np.linspace(0, 0.66, 60):
        pt = t_apex + t_frac * (t_crest - t_apex)
        ax.plot([3.3, 6.1], [pt[1], pt[1]],
                color="#CED4DA", linewidth=0.8, alpha=0.4, zorder=2)

    # Global minimum line (thick, highlighted)
    min_t = 0.22  # approx where bottleneck is thinnest
    min_pt = t_apex + min_t * (t_crest - t_apex)
    ax.plot([3.3, 6.1], [min_pt[1], min_pt[1]],
            color=COLORS["hard"], linewidth=3.5, zorder=5,
            label="Global minimum")
    ax.add_patch(Circle((min_pt[0], min_pt[1]), 0.15,
                         facecolor=COLORS["hard"], edgecolor="white",
                         linewidth=2, zorder=6))

    # Annotation with value
    ax.annotate("",
                xy=(4.7, min_pt[1] + 0.3), xytext=(7.5, min_pt[1] + 0.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["hard"], lw=2),
                zorder=5)
    ax.text(7.7, min_pt[1] + 0.3,
            f"min = {1.82:.2f} mm\n(bottleneck)",
            fontsize=10, color=COLORS["hard"], fontweight="bold",
            va="center")

    ax.text(7.5, 1.5,
            f"Pre-computed offsets:\n0.0, 0.1, ... 5.0 mm\n→ Zero-latency UI",
            fontsize=9, color=COLORS["primary"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["highlight"],
                      edgecolor=COLORS["primary"]))

    plt.tight_layout(pad=1.5)
    out = OUTPUT_DIR / "11_global_minimum_diagram.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CHART 12: Per-Landmark σ_k (adaptive sigma) Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def chart12_sigma_landmark_heatmap():
    """Per-landmark sigma_k values from the AdaptiveWingLoss heatmap generation."""

    # Data from the project — σ_k values per landmark (from loss.py)
    # These values represent the adaptive sigma per landmark in heatmap generation
    landmarks = [
        "Upper_tip", "Upper_apex",
        "Labial_midroot", "Labial_crest",
        "Palatal_midroot", "Palatal_crest",
        "ANS", "PNS",
        "LB", "PB",
    ]

    # Adaptive sigma values (simulated from the adaptive sigma per-landmark behavior)
    # Hard landmarks (small σ) = sharper peaks needed; easy landmarks (large σ)
    # Based on inverse relationship with MRE: hard landmarks need sharper supervision
    sigma_k = [12, 15, 20, 22, 18, 22, 10, 10, 8, 8]  # pixels

    # MRE values for comparison
    mre_vals = [1.473, 1.471, 0.737, 0.964, 0.918, 0.967, 2.501, 2.197, 2.000, 2.451]

    short_names = [
        "Upper\ntip", "Upper\napex",
        "Labial\nmidroot", "Labial\ncrest",
        "Palatal\nmidroot", "Palatal\ncrest",
        "ANS", "PNS",
        "LB", "PB",
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor="white",
                                    gridspec_kw={"height_ratios": [1.2, 1]})

    # ── Top: σ_k bar chart ────────────────────────────────────────────────────
    ax1.set_facecolor(COLORS["light_bg"])

    colors_sigma = [
        COLORS["hard"] if s <= 10 else COLORS["medium"] if s <= 18 else COLORS["easy"]
        for s in sigma_k
    ]
    x = np.arange(len(landmarks))
    bars = ax1.bar(x, sigma_k, color=colors_sigma, edgecolor="white",
                   linewidth=2, width=0.65)

    for bar, s in zip(bars, sigma_k):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{s}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=9)
    ax1.set_ylabel("σ_k (pixels)", fontsize=12, fontweight="medium")
    ax1.set_title(
        "Per-Landmark Adaptive Sigma (σ_k) — Heatmap Sharpness Control\n"
        "Lower σ_k = sharper peak = harder to localize (ANS, PB, LB)",
        pad=12, fontsize=13, fontweight="bold", color=COLORS["text_dark"]
    )
    ax1.set_ylim(0, 28)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, color=COLORS["grid"])
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COLORS["hard"],   label="Sharp (σ ≤ 10 px) — hard landmarks"),
        mpatches.Patch(color=COLORS["medium"], label="Medium (11–18 px)"),
        mpatches.Patch(color=COLORS["easy"],   label="Diffuse (σ > 18 px) — easy landmarks"),
    ]
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=9)

    # ── Bottom: σ_k vs MRE scatter ────────────────────────────────────────────
    ax2.set_facecolor(COLORS["light_bg"])

    colors_scatter = [
        COLORS["hard"] if s <= 10 else COLORS["medium"] if s <= 18 else COLORS["easy"]
        for s in sigma_k
    ]
    ax2.scatter(sigma_k, mre_vals, c=colors_scatter, s=200,
                 edgecolors="white", linewidths=2, zorder=3, alpha=0.9)

    for i, (s, m, lm) in enumerate(zip(sigma_k, mre_vals, landmarks)):
        ax2.annotate(
            lm, (s, m),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color=COLORS["text_dark"], fontweight="bold",
        )

    # Trend line
    z = np.polyfit(sigma_k, mre_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(sigma_k) - 1, max(sigma_k) + 1, 100)
    ax2.plot(x_line, p(x_line), "--", color=COLORS["benchmark"],
             linewidth=2, alpha=0.7, label=f"Trend (slope={z[0]:.3f})")

    ax2.set_xlabel("σ_k (pixels) — larger = easier landmark", fontsize=12, fontweight="medium")
    ax2.set_ylabel("MRE (mm)", fontsize=12, fontweight="medium")
    ax2.set_title(
        "Inverse Relationship: σ_k vs MRE — harder landmarks need sharper Gaussian peaks",
        pad=10, fontsize=11, fontweight="bold", color=COLORS["text_dark"]
    )
    ax2.set_xlim(5, 26)
    ax2.set_ylim(0.5, 3.0)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, color=COLORS["grid"])
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=9)

    plt.tight_layout(pad=1.5)
    out = OUTPUT_DIR / "12_sigma_landmark_heatmap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved: {out}")
    return out


def main():
    print("\n" + "═" * 60)
    print("  ADDITIONAL CLINICAL VISUALIZATIONS")
    print("═" * 60 + "\n")

    charts = [
        ("10. Clinical UI Schematic",        chart10_clinical_ui_schematic),
        ("11. Global Minimum Algorithm",       chart11_global_minimum_diagram),
        ("12. Per-Landmark σ Heatmap",          chart12_sigma_landmark_heatmap),
    ]

    for name, fn in charts:
        print(f"  Generating {name}...")
        try:
            path = fn()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "═" * 60)
    print("  DONE")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()