# Ceph-Project — Product & Clinical System Specification
**Version:** 2.1.1  
**Context:** Local Mac M4 Inference Pipeline & Advanced Clinical Features  
**Document Purpose:** Master blueprint containing current technical specifications, upcoming overnight milestones, and session state tracking flags.

---

## 📋 Master Progress Tracker & Session Checklists

### 1. Completed & Fully Integrated Infrastructure
- [x] **Phase 1 Calibration & Splits:** 104/104 images fully calibrated and stratified into a 5-fold patient-level GroupKFold split configuration.
- [x] **Phase 2 Landmark Model Architecture:** HRNet-W32 model with localized Coordinate Attention + Uncertainty Heads fully wired and verified.
- [x] **Phase 3 Biomechanics Core Engine:** Zhang et al. 2021 clinical standard calculations successfully mapped into `src/phase3/biomechanics.py`.
- [x] **Local Multi-Container Deployment:** Full-stack Docker Compose arrangement running dynamically with FastAPI API backend mapped to host port `8123` and Astro Web interface exposed on port `4321`.
- [x] **Live Dual-Data Flow:** Complete transition from hardcoded static mock fallbacks to a dynamic live inference data stream rendering real coordinate properties.

### 2. High-Priority Overnight Tasks (Current Active Session Scope)
- [x] **PRIORITY 0: CRITICAL BUG FIX - Missing Measurement Lines**
  - [x] Fix state mapping in `DashboardApp.tsx` so `results.lines_3_level` is correctly passed to the canvas.
  - [x] Add mock fallback in `CephCanvasEditor.tsx` so 3-level lines always render even when backend API hasn't computed `bone_thickness.lines_3_level` yet (lines now show even on plain image upload before analysis).
  - [x] Fix `boneThickness[key]` → `bt[key]` variable bug inside the levels map (previously referenced the undefined outer prop after the mock fallback was introduced).
  - [x] Rewrite `compute_bone_thickness_3_levels` math: 6 independent tooth→bone gap distances (palatal+labial at cervical/middle/apical) with precise pixel start/end coordinates. Old bone-extent math (Bone→Bone passing through tooth) replaced with `_tooth_side_to_bone()` nearest-point logic.
  - [x] CephCanvasEditor TS interface updated to `Segment6` / `Lines3Level` schema; mockLines now contains separate tooth+bone coords per segment; Konva renders 6 independent gap lines (violet P:Xmm palatal, orange L:Xmm labial) with level badge.
- [ ] **Task 1: Interactive Canvas Enhancements**
  - [ ] Undo/Redo state history and "Confirm & Save" buttons exist in CephCanvasEditor.tsx toolbar code — but may need wiring verification against live API POST.
- [x] **Task 2: Medical Clinical Interface Theme Shift**
  - [x] `index.astro`: removed `class="dark"` from `<html>` tag — light mode is now the default.
  - [x] `ThemeToggle.tsx`: `isDark` initial state now defaults to `false` (was hardcoded `true`); localStorage respected as override only.
  - [x] `DashboardApp.tsx`: stripped `dark:` variants from all UI chrome; canvas container retained `bg-slate-900` for diagnostic contrast.
  - [x] `MetricCard.tsx`, `UploadZone.tsx`, `ThemeToggle.tsx`: all dark-mode Tailwind classes removed.
  - [x] Toolbar contrast fix: high-contrast `text-white` labels, larger checkboxes, disabled Undo has distinct border style, amber Confirm&Save pill, Dev and ✕ hide buttons.
- [ ] **Task 3: Phase 2b Bone Segmentation Scaffold**
  - [x] `scripts/run_phase2b_segmentation.py` scaffolded with `--mock` flag (dry-run).
  - [x] `PolygonToMaskConverter` via `cv2.fillPoly` confirmed implemented in `src/phase2b/segmentation_dataset.py`.
  - [x] Albumentations dual-target (`additional_targets={'mask': 'mask'}`) confirmed in `SegmentationDataset.__getitem__`.
  - [ ] **Training blocked**: 0/104 images have polygon annotations in CVAT; real training requires Dr. to export polygons.

---

## 📐 Functional Subsystem Specifications

### Phase 3 — Parallel Bone Thickness Architecture
Alveolar bone layer thickness computation runs two systems side-by-side to accommodate both front-facing display tools and programmatic diagnostic metrics:

#### 1. Plan B: Perpendicular 3-Level Projections (Visual Aid Modality)
- **Clinical Target:** Provides visual alignment lines for the Doctor at specific horizontal anchors across the upper central incisor tooth root structure.
- **Anatomical Level Division:** Splits the root vertically from the cervical margin line down to the absolute apex node into exact thirds: Cervical, Middle, and Apical.
- **Data Shape:** Returns discrete starting and ending vector coordinate attributes (`lb_x`, `lb_y` for Labial; `pb_x`, `pb_y` for Palatal) paired with converted millimeter metrics for clean stage display mapping inside the canvas view layer.

#### 2. Plan C: Closest-Point Minimum Search with Crest Offset Protection (Clinical Decision Modality)
- **Clinical Target:** Computes the true physical minimum thickness values across the functional surface of the root structure to protect anatomical integrity.
- **Crest Exclusion Protection Zone:** Imposes a mandatory 1.5mm vertical buffer offset parameter measured upwards from the Alveolar Crest baseline landmark points.
- **Mathematical Execution Rule:** $$\mathcal{T}_{\text{min}} = \min_{i, j} \| \mathbf{p}_{\text{tooth}, i} - \mathbf{p}_{\text{bone}, j} \|_2$$  
  Subject to the spatial protection coordinate filter rule:  
  $$y_{\text{tooth}, i} > (y_{\text{crest}} + \text{offset}_{\text{px}})$$

#### 3. Standardized 4-Tier Clinical Bone Condition Classification
| Clinical Condition Tier | Technical Rule Mapping | Clinical Classification Rule |
| :--- | :--- | :--- |
| **Thick Alveolar Type** | `LB >= 0.5mm` AND `PB >= 0.5mm` | Balanced thick profile; safest biomechanical target. |
| **Relatively Thick Type** | `(LB < 0.5mm ⊕ PB < 0.5mm)` | Single-sided thinning. Moderate risk vector. |
| **Thin Alveolar Type** | `LB < 0.5mm` AND `PB < 0.5mm` | Structural bone layer thinning on both facets. High caution required. |
| **Vulnerably Thin Type** | `LB <= 0.2mm` OR `PB <= 0.2mm` | Imminent cortical wall perforation risk. Immediate warning badge status. |

---

## 📡 Unified Backend JSON Payload Schema Matrix
```json
{
  "status": "ok",
  "landmarks": { ... },
  "bone_thickness": {
    "labial_min_mm": 3.86,
    "palatal_min_mm": 7.42,
    "classification": "Thick Alveolar Type",
    "lines_3_level": {
      "cervical": {
        "palatal_distance_mm": 1.9, "palatal_tooth_x": 1005.2, "palatal_tooth_y": 1345.1,
        "palatal_bone_x": 945.1, "palatal_bone_y": 1345.1,
        "labial_distance_mm": 2.4, "labial_tooth_x": 1120.4, "labial_tooth_y": 1338.3,
        "labial_bone_x": 1175.2, "labial_bone_y": 1338.3
      },
      "middle": { ... },
      "apical": { ... }
    }
  },
  "metrics": { ... }
}