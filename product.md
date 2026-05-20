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
  - [x] Fix state mapping in `DashboardApp.tsx` so `results.lines_3_level` is correctly passed to the canvas (normalizedResults pipeline was already flattening `bone_thickness.lines_3_level` to `lines_3_level`).
  - [x] Verify Konva `<Line>` components in `CephCanvasEditor.tsx` correctly interpret image-space coords via `toContent()` — confirmed: returns `[ix*fitScale+offX, iy*fitScale+offY]` with no NaN on valid data.
- [x] **Task 1: Interactive Canvas Enhancements**
  - [x] Add coordinate state history array stack (`historyStack`) within `CephCanvasEditor.tsx` to log manual marker movements.
  - [x] Render responsive UI toolbar "Undo" control button (↩) to pop the tracking stack.
  - [x] Implement "Confirm & Save Changes" (amber high-contrast pill button) that freezes canvas, marshals dragged coordinates, and POSTs to `/api/v1/analyze`.
- [x] **Task 2: Medical Clinical Interface Theme Shift**
  - [x] Purge dark theme layout modifiers (`bg-slate-900`, `bg-black`, `dark:`) from index.astro, DashboardApp.tsx, MetricCard.tsx, UploadZone.tsx, ThemeToggle.tsx.
  - [x] Inject clinical bright surface palettes (`bg-slate-50`, `bg-white`, `text-slate-800`/`text-slate-900`) globally.
  - [x] Lock isolated dark backdrops solely onto the X-ray view stage layer (`bg-slate-950` in canvas container, `border-slate-800` canvas border) — preserved for diagnostic contrast.
- [x] **Task 3: Phase 2b Bone Segmentation Scaffold**
  - [x] `PolygonToMaskConverter` via `cv2.fillPoly` already implemented in `src/phase2b/segmentation_dataset.py` (line 98).
  - [x] Albumentations dual-target (`additional_targets={'mask': 'mask'}`) wired in `SegmentationDataset.__getitem__` (line 105).
  - [x] `scripts/run_phase2b_segmentation.py` with `--mock` dry-run switch scaffolded; validates forward+backward pass with synthetic data.

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
      "cervical": { "palatal": {"x": 1065.2, "y": 1345.1, "mm": 1.9}, "labial": {"x": 1120.4, "y": 1338.3, "mm": 2.4} },
      "middle": { "palatal": {"x": 1062.1, "y": 1300.5, "mm": 1.5}, "labial": {"x": 1112.8, "y": 1295.2, "mm": 1.8} },
      "apical": { "palatal": {"x": 1059.8, "y": 1270.2, "mm": 0.9}, "labial": {"x": 1107.1, "y": 1266.0, "mm": 0.6} }
    }
  },
  "metrics": { ... }
}