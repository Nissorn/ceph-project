# Ceph-Project — Product & Clinical Specification

## Overview

Ceph-project is a clinical decision support system for orthodontic treatment planning,
focused on cephalometric X-ray landmark detection, root position classification, and
alveolar bone thickness assessment.

---

## Phase 3 — Biomechanics Engine

### Bone Thickness: Plan B and Plan C in Parallel

Two approaches run in parallel to serve different purposes:

| Method | Purpose | Output |
|---|---|---|
| **Plan B** | Frontend visualization — 3 horizontal measurement lines | `lines_3_level`: cervical / middle / apical distances for UI |
| **Plan C** | Clinical classification — biometric safety | `classification`: minimum distance + 4-tier tier |

---

### Plan B — 3-Level Root Line Distances (for UI Visualization)

**Purpose:** The Doctor requested display of 3 horizontal/perpendicular measurement
lines at cervical, middle, and apical root levels. Plan B provides these for frontend
rendering and reporting.

**Core Logic:** The U1 long axis (Upper_tip → Upper_apex) is divided into thirds:

```
Cervical  → 1/3 down from tip  (near Labial_crest / Palatal_crest)
Middle    → 2/3 down from tip  (at midroot level)
Apical    → at the root apex
```

At each level a perpendicular projection is cast through the tooth root and the
Labial and Palatal bone contours are searched for the width occupied by cortical bone.
The width (max projection − min projection) on each side is returned in mm.

**API Response Shape:**
```json
{
  "bone_thickness": {
    "lines_3_level": {
      "cervical": { "lb_mm": 0.636, "pb_mm": 0.636 },
      "middle":   { "lb_mm": 0.636, "pb_mm": 0.636 },
      "apical":   { "lb_mm": 0.636, "pb_mm": 0.636 }
    }
  }
}
```

---

### Plan C — Minimum Distance with Crest Offset (for Clinical Classification)

**Purpose:** Core mathematical logic for the 4-tier biometric safety classification.
Uses the minimum contour-to-contour distance with a crest offset to bypass the
anatomically thin alveolar crest edge.

**Crest Offset Protection Feature:** A mandatory 1–2 mm upward offset from the
Labial_crest / Palatal_crest landmark excludes the hyper-thin crest region from
the minimum-distance search. This prevents the crest from artificially biasing
the result downward.

**Algorithm:**
```
1. Input: tooth_contour (N×2), bone_contour (M×2), crest_landmark (x,y), offset_mm
2. Compute offset_px = offset_mm / mm_per_pixel
3. Filter: tooth_points where y > (crest_y + offset_px)
4. Compute pairwise Euclidean distance matrix: (N_filtered × M)
5. Find global minimum → min_dist_px
6. Convert to mm: min_dist_mm = min_dist_px × mm_per_pixel
7. Record closest tooth point and closest bone point for audit
```

**API Response Shape:**
```json
{
  "bone_thickness": {
    "classification": {
      "labial_min_mm":  4.727,
      "palatal_min_mm": 5.651,
      "tier":           "Thick Bone",
      "is_vulnerable":  false,
      "classification_note": "Thick bone: LB=4.727mm, PB=5.651mm (both sides >= 0.5 mm)"
    }
  }
}
```

**4-Tier Classification:**

| Tier | Rule | Clinical Implication |
|---|---|---|
| Thick Bone | LB ≥ 0.5 mm AND PB ≥ 0.5 mm | Normal bone support; standard biomechanics |
| Relatively Thick | (LB < 0.5 mm OR PB < 0.5 mm) but NOT both | One side thinner; caution required |
| Thin Type | LB < 0.5 mm AND PB < 0.5 mm | Both sides thin; limited torque possible |
| Vulnerably Thin | LB ≤ 0.2 mm OR PB ≤ 0.2 mm | High risk; requires careful monitoring |

> **Note:** Vulnerably Thin overrides all other categories (highest priority).
> **Pending Dr. Confirmation:** All threshold values (0.5 mm, 0.2 mm) pending formal annotation review.

---

### Complete API Response Shape (Both Plans)

```json
{
  "bone_thickness": {
    "lines_3_level": {
      "cervical": { "lb_mm": 0.636, "pb_mm": 0.636 },
      "middle":   { "lb_mm": 0.636, "pb_mm": 0.636 },
      "apical":   { "lb_mm": 0.636, "pb_mm": 0.636 }
    },
    "classification": {
      "labial_min_mm":  4.727,
      "palatal_min_mm": 5.651,
      "tier":           "Thick Bone",
      "is_vulnerable":  false,
      "classification_note": "Thick bone: LB=4.727mm, PB=5.651mm (both sides >= 0.5 mm)"
    },
    "crest_offset_mm": 1.5,
    "mm_per_pixel": 0.0984
  }
}
```

---

### Mock Validation Engine

Both Plan B and Plan C are currently scaffolded with a **mock validation engine**
using synthetic numpy arrays representing tooth and bone contours.
Real patient segmentation data (polygon annotations) from the Doctor has not yet
been received.

The mock data validates:
- Mathematical correctness of Plan B perpendicular-width calculation
- Mathematical correctness of Plan C contour-to-contour minimum distance
- Crest offset filtering logic (Plan C)
- All 4-tier classification coverage (Plan C)
- Combined output shape from `compute_bone_thickness_full()`

**Code location:** `src/phase3/biomechanics.py` — Steps 7 & 8 (smoke tests)

---

### Landmark Reference (10 Landmarks)

```
Upper_tip        — Incisal tip of upper central incisor
Upper_apex       — Root apex of upper central incisor
ANS              — Anterior Nasal Spine
PNS              — Posterior Nasal Spine
LB               — Labial bone landmark
PB               — Palatal bone landmark
Labial_crest     — Labial alveolar crest
Palatal_crest    — Palatal alveolar crest
Labial_midroot  — Labial midroot
Palatal_midroot — Palatal midroot
```

---

### Treatment Classification (Zhang et al. 2021)

Based on U1-PP angle and root apex position, the system classifies patients into
biomechanical categories and recommends preferred/avoided treatment approaches.

**Angle zones:**
- `< 105°` — Retroclined
- `105–115°` — Normal inclination
- `> 115°` — Proclined

**Root apex positions:** Labial / Midway / Palatal

---

## Project Status

- Phase 1: Landmark detection (heatmap-based CNN) — trained on synthetic + real data
- Phase 2: Model training pipeline — MRE evaluation, argmax keypoint extraction
- Phase 3: Superimposition + biomechanics — in progress (Plan B + Plan C parallel)
- Phase 4: Output format and visualization — pending real data

## Data Rules

- No horizontal flip augmentation (breaks lateral anatomy)
- Patient-level train/test splits only (no image-level leakage)
- `mm_per_pixel` from calibration.csv — never hardcoded
- No raw images, annotations, or model weights in git