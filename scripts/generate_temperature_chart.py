#!/usr/bin/env python3
"""
Chart 13: Temperature Sensitivity — Clean Publication Figure
=============================================================
3-column × 2-row. Dark theme. Zero annotations.
Row 1: heatmaps (T=10 | T=0.1 | argmax)
Row 2: 1D cross-sections (same T progression, overlay in panel 3)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = "/Users/onis2/Project/Singdent/ceph-auto/ceph-project/reports/figures"
SIZE = 200
cx, cy = SIZE // 2, SIZE // 2  # 100, 100


def softmax_T(logits, T):
    scaled = logits / T
    exp = np.exp(scaled - np.max(scaled))
    return exp / exp.sum()


def gaussian_2d(size, cx, cy, sigma):
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    return np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / sigma ** 2)


def build_heatmap(T, cx=100, cy=100, sigma=8):
    raw = gaussian_2d(SIZE, cx, cy, sigma)
    flat = raw.flatten()
    probs = softmax_T(np.log(flat + 1e-10), T)
    return probs.reshape(SIZE, SIZE)


def argmax_map():
    arr = np.zeros((SIZE, SIZE))
    arr[cy, cx] = 1.0
    return arr


# Colormaps — cool blue for diffuse (T=10), warm coral for sharp (T=0.1)
CMAP_DIFFUSE = LinearSegmentedColormap.from_list("diffuse",
    ["#0D1117", "#1D3557", "#457B9D", "#A8DADC", "white"])
CMAP_SHARP  = LinearSegmentedColormap.from_list("sharp",
    ["#0D1117", "#6B1D1D", "#E76F51", "#FF6B35", "white"])
CMAP_ARGMAX = LinearSegmentedColormap.from_list("argmax",
    ["#0D1117", "#DC3545", "white"])


# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 10), facecolor="#0D1117")
fig.patch.set_facecolor("#0D1117")

gs = GridSpec(2, 3, figure=fig,
              left=0.0, right=1.0, top=1.0, bottom=0.0,
              hspace=0.0, wspace=0.0)

# ════════════════════════════════════════════════════════════════════════════
# ROW 1: Heatmaps — T=10 | T=0.1 | argmax
# ════════════════════════════════════════════════════════════════════════════
panel_specs = [
    (gs[0, 0], build_heatmap(T=10.0), CMAP_DIFFUSE, 55),   # T=10: wide FWHM ring
    (gs[0, 1], build_heatmap(T=0.1),  CMAP_SHARP,   5),    # T=0.1: tight ring
    (gs[0, 2], argmax_map(),          CMAP_ARGMAX,  1.2),  # argmax: dot
]

for sp, hm, cmap, fwhm_diam in panel_specs:
    ax = fig.add_subplot(sp)
    ax.set_facecolor("#0D1117")
    ax.imshow(hm, cmap=cmap, vmin=0, vmax=1,
              extent=[0, SIZE, 0, SIZE], origin="lower", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    # Crosshair at image center (true landmark location)
    ax.axhline(cy, color="white", lw=0.6, alpha=0.25)
    ax.axvline(cx, color="white", lw=0.6, alpha=0.25)
    ax.plot(cx, cy, "w+", markersize=12, markeredgewidth=2.5, zorder=5)

    # FWHM ring showing peak spread (wider = more center bias)
    ring = Circle((cx, cy), fwhm_diam / 2,
                  fill=False, edgecolor="white", linewidth=2, linestyle="--", alpha=0.75)
    ax.add_patch(ring)


# ════════════════════════════════════════════════════════════════════════════
# ROW 2: 1D Cross-Sections
# ════════════════════════════════════════════════════════════════════════════
x = np.arange(SIZE)
sigma_true = 8.0

base = np.exp(-0.5 * ((x - cx) ** 2) / sigma_true ** 2)
base = base / base.max()

p_T10  = (base ** (1.0 / 10.0))
p_T10  = p_T10 / p_T10.max()

p_T01  = (base ** (1.0 / 0.1))
p_T01  = p_T01 / p_T01.max()

p_arg  = np.zeros(SIZE)
p_arg[cx] = 1.0

row2_specs = [
    (gs[1, 0], p_T10,  "#457B9D", 0.25),   # T=10 fill
    (gs[1, 1], p_T01,  "#E76F51", 0.30),   # T=0.1 fill
    (gs[1, 2], None,    None,       None),   # overlay panel
]

for sp, prof, color, fill_alpha in row2_specs:
    ax = fig.add_subplot(sp)
    ax.set_facecolor("#0D1117")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    if prof is None:
        # Panel 3: overlay all three profiles
        ax.fill_between(x, p_T10, alpha=0.2, color="#457B9D")
        ax.plot(x, p_T10, color="#457B9D", linewidth=2.5)
        ax.fill_between(x, p_T01, alpha=0.2, color="#E76F51")
        ax.plot(x, p_T01, color="#E76F51", linewidth=2.5)
        ax.plot(x, p_arg,  color="white",   linewidth=3.5, linestyle="--")
        ax.axvline(cx, color="white", lw=0.6, alpha=0.2, linestyle="--")
        ax.set_xlim(0, SIZE)
        ax.set_ylim(0, 1.2)
        continue

    ax.fill_between(x, prof, alpha=fill_alpha, color=color)
    ax.plot(x, prof, color=color, linewidth=2.5)
    ax.axvline(cx, color="white", lw=0.6, alpha=0.2, linestyle="--")
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, 1.15)


plt.savefig(f"{OUTPUT_DIR}/13_temperature_sensitivity.png",
            dpi=180, bbox_inches="tight", facecolor="#0D1117")
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/13_temperature_sensitivity.png")