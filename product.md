# Ceph-Project — Product & Clinical Specification

## Overview

Ceph-project is a clinical decision support system for orthodontic treatment planning,
focused on cephalometric X-ray landmark detection, root position classification, and
alveolar bone thickness assessment.

---

## Phase 3 — Biomechanics Engine

### Bone Thickness Pipeline Comparison

Three approaches were evaluated for computing alveolar bone thickness
(Labial and Palatal) relative to the upper central incisor root.

| Method | Description | Risk |
|---|---|---|
| Plan A | 3-level perpendiculars (cervical / middle / apical) | Lines may miss deformed bone |
| Plan B | Absolute minimum distance (no offset) | Hyper-thin crest edge biases result |
| **Plan C** | **Minimum distance with crest offset** | **Selected — most robust** |

---

### Idea C: Minimum Distance with Crest Offset

**Core Logic:** Extract the tooth root contour (from Crest to Apex) and compute the
exact Minimum Euclidean Distance from each filtered tooth root point to the Labial
and Palatal bone contours.

#### Crest Offset Protection Feature

A mandatory crest offset exclusion zone is applied before the minimum-distance search.
The alveolar crest (Labial_crest / Palatal_crest) is the anatomical region where the
cortical bone is normally the thinnest — this does not represent functional bone
thickness and must be excluded.

- **Offset:** 1–2 mm upward (cervical direction) from the crest landmark
- **Implementation:** Only tooth contour points with `y > (crest_y + offset_px)` are used
- **Fallback:** If all points are filtered out, all tooth points are considered

#### Algorithm (Pipeline C)

```
1. Input: tooth_contour (N×2), bone_contour (M×2), crest_landmark (x,y), offset_mm
2. Compute offset_px = offset_mm / mm_per_pixel
3. Filter: tooth_points where y > (crest_y + offset_px)
4. Compute pairwise Euclidean distance matrix: (N_filtered × M)
5. Find global minimum → min_dist_px
6. Convert to mm: min_dist_mm = min_dist_px × mm_per_pixel
7. Record closest tooth point and closest bone point for audit
```

#### 4-Tier Classification

| Tier | Rule | Clinical Implication |
|---|---|---|
| Thick Bone | LB ≥ 0.5 mm AND PB ≥ 0.5 mm | Normal bone support; standard biomechanics |
| Relatively Thick | (LB < 0.5 mm OR PB < 0.5 mm) but NOT both | One side is thinner; caution required |
| Thin Type | LB < 0.5 mm AND PB < 0.5 mm | Both sides thin; limited torque possible |
| Vulnerably Thin | LB ≤ 0.2 mm OR PB ≤ 0.2 mm | High risk; require careful monitoring |

> **Note:** Vulnerably Thin overrides all other categories (highest priority).

> **Pending Dr. Confirmation:** All threshold values (0.5 mm, 0.2 mm) are
> pending formal annotation review from the Doctor before clinical use.

#### Mock Validation Engine

Pipeline C is currently scaffolded with a **mock validation engine** using synthetic
numpy arrays representing tooth and bone contours. Real patient segmentation data
(polygon annotations) from the Doctor has not yet been received.

The mock data validates:
- Mathematical correctness of contour-to-contour distance calculation
- Crest offset filtering logic
- All 4-tier classification coverage
- No None / zero-value errors in the pipeline

**Mock data locations:** `src/phase3/biomechanics.py` — Step 7 (smoke test)

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
- Phase 3: Superimposition + biomechanics — in progress
- Phase 4: Output format and visualization — pending real data

## Data Rules

- No horizontal flip augmentation (breaks lateral anatomy)
- Patient-level train/test splits only (no image-level leakage)
- `mm_per_pixel` from calibration.csv — never hardcoded
- No raw images, annotations, or model weights in git