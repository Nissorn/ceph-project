import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def draw_box(ax, x, y, width, height, text, facecolor, edgecolor, fontsize=10):
    box = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True, zorder=3)
    return x + width/2, y + height/2, x + width, y + height/2, x + width/2, y

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("",
                xy=(x2, y2), xycoords='data',
                xytext=(x1, y1), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5, shrinkA=0, shrinkB=0),
                zorder=1)

def generate_pipeline_diagram(out_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Phase 1
    cx1, cy1, rx1, ry1, bx1, by1 = draw_box(ax, 1, 6, 2.5, 1, "Phase 1\nData Prep &\nCalibration", "#e0f7fa", "#00838f")
    # Phase 2
    cx2, cy2, rx2, ry2, bx2, by2 = draw_box(ax, 4.5, 6, 2.5, 1, "Phase 2\nPrediction\n(Keypoints & Seg)", "#e8f5e9", "#2e7d32")
    # Phase 3
    cx3, cy3, rx3, ry3, bx3, by3 = draw_box(ax, 8, 5, 3.5, 3, "Phase 3\nHeuristics & Superimposition", "#fff3e0", "#ef6c00")
    
    # Sub-boxes in Phase 3
    draw_box(ax, 8.2, 7.0, 3.1, 0.6, "Sliding Window", "#ffe0b2", "#e65100", 9)
    draw_box(ax, 8.2, 6.2, 3.1, 0.6, "Geometric Snapping", "#ffe0b2", "#e65100", 9)
    draw_box(ax, 8.2, 5.4, 3.1, 0.6, "Cervical Offset & Global Min", "#ffe0b2", "#e65100", 9)

    # Phase 4
    cx4, cy4, rx4, ry4, bx4, by4 = draw_box(ax, 12.5, 6, 1.2, 1, "Phase 4\nExport", "#f3e5f5", "#6a1b9a")

    draw_arrow(ax, rx1, ry1, 4.5, 6.5)
    draw_arrow(ax, rx2, ry2, 8, 6.5)
    draw_arrow(ax, 11.5, 6.5, 12.5, 6.5)

    plt.title("Pipeline Stage Diagram", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_architecture_diagram(out_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    cx1, cy1, rx1, ry1, bx1, by1 = draw_box(ax, 0.5, 1.5, 1.5, 1, "X-ray Input", "#eceff1", "#455a64")
    
    cx2a, cy2a, rx2a, ry2a, bx2a, by2a = draw_box(ax, 3, 2.2, 2.5, 1, "Phase2A:\nHRNet\n(Keypoint Heatmaps)", "#e3f2fd", "#1565c0")
    cx2b, cy2b, rx2b, ry2b, bx2b, by2b = draw_box(ax, 3, 0.8, 2.5, 1, "Phase2B:\nDeepLabV3+\n(Segmentation)", "#e3f2fd", "#1565c0")
    
    cx3, cy3, rx3, ry3, bx3, by3 = draw_box(ax, 6.5, 1.5, 2.5, 1, "Geometric Snapping\n(Integration)", "#fbe9e7", "#d84315")
    
    cx4, cy4, rx4, ry4, bx4, by4 = draw_box(ax, 10, 1.5, 2.5, 1, "Biomechanics\n(Measurements)", "#f0f4c3", "#9e9d24")

    # Arrows
    draw_arrow(ax, rx1, ry1, 3, 2.7) # X-ray to 2A
    draw_arrow(ax, rx1, ry1, 3, 1.3) # X-ray to 2B
    
    draw_arrow(ax, rx2a, ry2a, 6.5, 2.0) # 2A to Snapping
    draw_arrow(ax, rx2b, ry2b, 6.5, 1.6) # 2B to Snapping
    
    draw_arrow(ax, rx3, ry3, 10, 2.0) # Snapping to Biomechanics

    plt.title("Architecture Diagram - Pipeline Flow", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_temperature_sensitivity(out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate spatial locations (-10 to 10 pixels from center)
    x = np.linspace(-10, 10, 200)
    
    # Simulated heatmap logits (e.g. distance from center squared)
    logits = -0.5 * (x**2)
    
    def softmax(l, T):
        e = np.exp(l / T)
        return e / np.sum(e)
    
    y_10 = softmax(logits, 10.0)
    y_01 = softmax(logits, 0.1)
    
    ax.plot(x, y_10, label='T=10 (Diffuse)', color='#1976d2', linewidth=2.5)
    ax.plot(x, y_01, label='T=0.1 (Center Bias / Sharp)', color='#d32f2f', linewidth=2.5)
    
    ax.fill_between(x, y_10, alpha=0.2, color='#1976d2')
    ax.fill_between(x, y_01, alpha=0.2, color='#d32f2f')
    
    ax.set_title("Temperature Sensitivity for Heatmap Peak Extraction", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Spatial Offset from True Landmark (pixels)", fontsize=12)
    ax.set_ylabel("Probability Density (Softmax output)", fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Annotations
    ax.annotate("Sharp peak tightly\nbounds the global minimum", xy=(0, max(y_01)), xytext=(2, max(y_01)*0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=11)
                
    ax.annotate("Diffuse distribution\nspreads confidence", xy=(5, y_10[len(x)//2 + 50]), xytext=(6, y_10[len(x)//2 + 50] + 0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    out_dir = "/Users/onis2/Project/Singdent/ceph-auto/ceph-project/reports/figures"
    generate_pipeline_diagram(os.path.join(out_dir, "pipeline_stage_diagram.png"))
    generate_architecture_diagram(os.path.join(out_dir, "architecture_diagram.png"))
    generate_temperature_sensitivity(os.path.join(out_dir, "temperature_sensitivity.png"))
    print("Figures generated successfully.")
