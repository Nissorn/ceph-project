# context.md — Cephalometric Landmark Detection System
# Singapodent Internship Project
# Last updated: 2026-05-03

---

## Background

**Internship at:** Singapodent — dental digital ecosystem company, Ho Chi Minh City, Vietnam  
**Collaborator:** Clinic orthodontist (Dr.) who provides annotations and clinical domain knowledge  
**Goal:** Build an AI system that analyzes lateral cephalogram X-ray images to automatically measure orthodontic tooth movement and classify the type of treatment applied

**Clinical problem this solves:**
- Currently orthodontists manually compare before/after X-rays (T1 and T2) to assess how a tooth moved
- This takes time and has inter-observer variability
- This system automates the measurement and outputs clinically meaningful numbers (degrees of tipping, mm of translation) with visual tracing overlay

---

## Dataset

### Primary Dataset (Proprietary — NEVER commit to git)
| Property | Value |
|----------|-------|
| Source | Clinic's own X-ray scanner |
| Image quality | High — scanner-grade JPG, ~1729×2048 px |
| Patients | 52 |
| Images | 104 total (1× T1 + 1× T2 per patient) |
| Format | JPG images + CVAT XML 1.1 annotations |
| Location | `data/raw/` (gitignored) |

### Why not use public datasets?
- Aariz dataset (2025): quality deemed insufficient by clinic Dr.
- ISBI 2015: different landmark set, different scanner characteristics
- Vietnamese clinic data has superior image quality and is the deployment target

### Annotation Format (CVAT XML 1.1)
Each image block contains up to 4 annotation types:

**1. Calibration_30mm (polyline, 2 points)**
```xml
<polyline label="Calibration_30mm" points="x1,y1;x2,y2"/>
```
Represents the 0mm–30mm span of a physical ruler embedded in the X-ray.
Used to compute `mm_per_pixel = 30.0 / euclidean_distance(p1, p2)`.
Ruler is nearly vertical in all images. Annotated for all 104 images.

**2. Incisor_Maxilla_Complex_Skeleton (skeleton, 8 keypoints)**
```xml
<skeleton label="Incisor_Maxilla_Complex_Skeleton">
  <points label="Upper_tip"       points="x,y"/>  <!-- index 0 -->
  <points label="Upper_apex"      points="x,y"/>  <!-- index 1 -->
  <points label="Labial_midroot"  points="x,y"/>  <!-- index 2 -->
  <points label="Labial_crest"    points="x,y"/>  <!-- index 3 -->
  <points label="Palatal_midroot" points="x,y"/>  <!-- index 4 -->
  <points label="Palatal_crest"   points="x,y"/>  <!-- index 5 -->
  <points label="ANS"             points="x,y"/>  <!-- index 6 -->
  <points label="PNS"             points="x,y"/>  <!-- index 7 -->
</skeleton>
```

**3. Segmentation polygons**
- `Upper_incisor` — tooth body
- `Labial_bone` — bone on labial (front) side
- `Palatal_bone` — bone on palatal (roof-of-mouth) side

**4. Tags**
- Treatment (T1 only): `Uncontrolled_tipping`, `Controlled_tipping`, `Translation`, `Root_torque`, `Extrusion`, `Intrusion`
- Quality: `Quality_Reject` (exclude from training), `Low_Visibility` (flag in output)

### Annotation Status
| Type | Status | Notes |
|------|--------|-------|
| Calibration_30mm | ✅ 104/104 | All done |
| Skeleton landmarks | 🔄 ~2/104 | Dr. annotating — main blocker |
| Segmentation polygons | 🔄 Partial | Not needed for Phase 2 |
| Treatment tags (T1) | 🔄 Partial | Needed for Phase 3 validation |

---

## Keypoint Specification

**10 logical keypoints — indices 0–9:**

| Index | Name | Source | Clinical meaning |
|-------|------|--------|-----------------|
| 0 | Upper_tip | CVAT annotation | Tip of upper central incisor (crown) |
| 1 | Upper_apex | CVAT annotation | Root apex of upper central incisor |
| 2 | Labial_midroot | CVAT annotation | Midpoint of root on labial (front) surface |
| 3 | Labial_crest | CVAT annotation | Alveolar bone crest on labial side |
| 4 | Palatal_midroot | CVAT annotation | Midpoint of root on palatal (back) surface |
| 5 | Palatal_crest | CVAT annotation | Alveolar bone crest on palatal side |
| 6 | ANS | CVAT annotation | Anterior Nasal Spine — stable reference point |
| 7 | PNS | CVAT annotation | Posterior Nasal Spine — stable reference point |
| 8 | LB | CVAT annotation (v2+) | Labial bone level — annotated directly in CVAT skeleton |
| 9 | PB | CVAT annotation (v2+) | Palatal bone level — annotated directly in CVAT skeleton |

**All 10 keypoints** are annotated via CVAT skeleton label `Incisor_Maxilla_Complex_Skeleton`.  
LB and PB appear in CVAT v2+ exports only — the v1 export (used in Session 1) only had 8 keypoints.  
`KEYPOINT_NAMES` in `src/data/cvat_parser.py` is fixed at 10 entries — never revert to 8.

**ANS and PNS are the superimposition reference plane** — they define the maxillary base plane that is used to register T1 and T2 images in Phase 3.

---

## 4-Phase Pipeline

### Phase 1 — Data Parsing & Calibration
**Status:** ✅ Calibration complete (2026-05-05). Landmark parsing ready — will activate automatically when Dr. exports annotations.  
**Blocker:** None for calibration. Landmark JSON awaits Dr. completing skeleton annotations.

```
Input:  data/annotations.xml          ← actual location (not data/raw/annotations/)

Output: data/processed/calibration.csv        — 104 rows, all 104 pass QC
        data/processed/calibration_clean.csv  — 104 rows (= calibration.csv; 0 rejected)
        data/processed/rejection_log.txt      — empty
```

**Calibration stats (2026-05-05, all 104 images):**
- mm/pixel min:  0.0974
- mm/pixel max:  0.0990
- mm/pixel mean: 0.0984 (≈ 1 pixel ≈ 0.1 mm)
- mm/pixel std:  0.0004 — scanner is extremely consistent; no outliers
- QC range used: [0.05, 0.30] — all 104 pass comfortably

**landmarks_clean.json structure:**
```json
{
  "images": [
    {
      "image_id": "Patient03_T1",
      "file_name": "Patient03_T1.jpg",
      "patient_id": "Patient03",
      "timepoint": "T1",
      "width": 1729,
      "height": 2048,
      "has_landmarks": true,
      "has_calibration": true,
      "keypoints": [
        {"name": "Upper_tip",      "x": 1225.15, "y": 1354.56, "visible": true},
        {"name": "Upper_apex",     "x": 1088.08, "y": 1147.76, "visible": true},
        {"name": "Labial_midroot", "x": 1151.42, "y": 1218.04, "visible": true},
        {"name": "Labial_crest",   "x": 1197.30, "y": 1243.68, "visible": true},
        {"name": "Palatal_midroot","x": 1093.16, "y": 1230.67, "visible": true},
        {"name": "Palatal_crest",  "x": 1119.06, "y": 1281.92, "visible": true},
        {"name": "ANS",            "x": 1121.81, "y": 1111.80, "visible": true},
        {"name": "PNS",            "x":  690.90, "y": 1139.05, "visible": true}
      ],
      "valid_mask": [1, 1, 1, 1, 1, 1, 1, 1],
      "treatment": ["Controlled_tipping"],
      "quality_flags": [],
      "polygons": {
        "Upper_incisor": [[x,y], ...],
        "Labial_bone":   [[x,y], ...],
        "Palatal_bone":  [[x,y], ...]
      }
    }
  ]
}
```

**calibration.csv columns:**
`image_id, file_name, patient_id, timepoint, pt1_x, pt1_y, pt2_x, pt2_y, distance_px, mm_per_pixel`

---

### Phase 2 — Landmark Detection
**Status:** Architecture decided, awaiting annotation data  
**Blocker:** Need ~20+ annotated images minimum before meaningful training  

**Model:** HRNet-W32 pretrained on COCO human pose (8 keypoints = our 8 landmarks)  
Framework: MMPose or timm + custom training loop  

**Augmentation pipeline (Albumentations):**
- Rotation: ±5° max
- Zoom: ±10%
- Brightness/Contrast: ±20%
- CLAHE (contrast enhancement for X-rays)
- ❌ NO horizontal flip — breaks lateral cephalogram anatomy

**Evaluation strategy: LOPO-CV (Leave-One-Patient-Out)**
- 52 folds total
- Each fold: train on 51 patients (102 images), test on 1 patient (2 images)
- T1 and T2 of the same patient ALWAYS stay in the same fold (no leakage)
- Reports: MRE ± std, SDR@2mm, SDR@2.5mm, SDR@3mm, SDR@4mm

**Confidence system:**
- Primary: heatmap peak value (0–1)
- Secondary: TTA (Test-Time Augmentation) standard deviation across 5 augmented predictions
- Visual output: green dot = confident, red dot = low confidence (warn Dr.)

**Reference benchmark (MICCAI CL-Detection2023):**
- Best team MRE: 1.518 mm on 38 landmarks, multi-center dataset
- Our task: 8 landmarks, single-center, higher image quality → expect better MRE
- Target: MRE < 2.0 mm for all 8 landmarks

---

### Phase 3 — Heuristic Treatment Classification
**Status:** Design complete, thresholds pending  
**Blocker:** Dr. must confirm tipping/translation thresholds  

**Algorithm:**
1. Load T1 and T2 landmark coordinates for same patient
2. Rigid registration: align coordinate systems using ANS-PNS line as reference plane
   - Translate so ANS = origin
   - Rotate so ANS-PNS vector is horizontal
   - Apply same transform to T2
3. Compute vectors (in registered space):
   - `delta_tip  = Upper_tip_T2  − Upper_tip_T1`   (2D vector in mm)
   - `delta_apex = Upper_apex_T2 − Upper_apex_T1`  (2D vector in mm)
4. Compute angle change of tooth long axis (Upper_apex → Upper_tip)
5. Apply classification rules:

```python
def classify(delta_tip, delta_apex, angle_change,
             tipping_threshold_deg,      # PENDING from Dr.
             translation_threshold_mm):  # PENDING from Dr.
    if abs(angle_change) > tipping_threshold_deg:
        if magnitude(delta_apex) < translation_threshold_mm:
            return "Uncontrolled_tipping"
        else:
            return "Controlled_tipping"
    elif magnitude(delta_tip) > translation_threshold_mm:
        return "Translation"
    elif ...
        return "Root_torque"
    ...
```

**Pending from Dr.:**
- [ ] Tipping threshold (degrees) — what angle change counts as tipping vs noise?
- [ ] Translation threshold (mm) — minimum displacement to call it translation?

---

### Phase 4 — Clinical Output
**Status:** Design complete  
**Blocker:** Depends on Phase 2 + 3  

**Outputs per patient pair (T1+T2):**
- `tipping_angle_deg`: float (positive = tipping, negative = uprighting)
- `translation_mm`: float (horizontal displacement of tooth centroid)
- `treatment_class`: string (from Phase 3 classification)
- `confidence`: dict per landmark (peak value + TTA std)
- `tracing_image`: overlay of T1 (blue) and T2 (red) landmarks on T2 image
- `report_dict`: JSON-serializable summary for downstream use

**Calibration usage:**
```python
mm_per_pixel = calibration_df.loc[image_id, "mm_per_pixel"]
translation_mm = translation_px * mm_per_pixel
```

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| HRNet-W32 over YOLOv8-Pose | HRNet maintains high-res features throughout — better for sub-pixel accuracy on medical images. YOLOv8 optimizes for object detection, unnecessary overhead for single-image landmark task. |
| LOPO-CV over 5-Fold | 52 patients is too few for hold-out test set. LOPO uses every case as test exactly once, giving 52 datapoints for MRE reporting. Scales automatically as more data arrives. |
| Heatmap confidence over MC Dropout | MC Dropout requires T=50 forward passes per image — impractical for clinical use. Heatmap peak value is a natural, free confidence signal. TTA adds lightweight uncertainty estimate. |
| No horizontal flip | Lateral cephalograms have strict anatomical orientation. Flipping swaps left/right facial anatomy — ANS/PNS/landmarks become anatomically incorrect. All other augmentations are safe. |
| Phase 3 as heuristics, not DL | 52 cases insufficient for deep learning classifier (overfitting guaranteed). Geometric rules derived from orthodontic knowledge are interpretable, clinically explainable, and require zero training data. |
| num_workers=0 in DataLoader | MPS backend on Apple Silicon does not support multiprocessing DataLoader workers. Setting >0 causes hangs. |

---

## Pending Questions for Dr.

- [ ] **Tipping threshold:** At what angle change (degrees) do we call it tipping vs noise?
- [ ] **Translation threshold:** Minimum displacement (mm) to classify as translation?
- [ ] **Torque definition:** How to distinguish root torque from controlled tipping geometrically?
- [ ] **Quality_Reject criteria:** Which specific image quality issues trigger rejection?

---

## References

1. **MICCAI CL-Detection2023 Challenge** — "Is the Problem Solved?" arxiv:2409.15834 (Sep 2024)
   - Best MRE: 1.518 mm on 38 landmarks, 600 images, multi-center
   - Top methods: HRNet + deep supervision + attention blocks
   - Dataset format: same COCO keypoint format as our export

2. **Rank-1 Method** — "Revisiting Cephalometric Landmark Detection from Human Pose Estimation" arxiv:2309.17143
   - Super-resolution heatmap head fixes quantization bias
   - Adapts human pose estimation best practices (UDP, unbiased data processing)

3. **HRNet+CBAM GitHub** — github.com/Xushuolin/HRNet-combine-CBAM-f-Cephalometric-Landmark-Detection
   - Ready-to-use MMPose config for cephalometric detection
   - Same COCO format input as our pipeline

---

## Session Log

### 2026-05-03 (Session 1)
- Confirmed dataset: 52 patients, 104 JPG images
- Confirmed keypoint order: Upper_tip(0) ... PNS(7)
- Confirmed annotation format: CVAT XML 1.1
- Confirmed calibration: polyline 2-point, 30mm, all 104 done
- Confirmed Calibration_30mm is nearly vertical (Δx small, Δy ~300px)
- Pipeline architecture finalized (4 phases)
- Model selection: HRNet-W32, LOPO-CV, heatmap + TTA confidence
- Identified key papers: CL-Detection2023 (arxiv:2409.15834), rank-1 paper (arxiv:2309.17143)
- Wrote parse_and_verify.py (Phase 1 prototype)
- Pending: Dr. to finish landmark annotations + confirm thresholds

### 2026-05-05 (Session 2)
- Built Phase 1 calibration pipeline: `src/data/cvat_parser.py`, `calibration.py`, `quality_filter.py`
- CLI: `scripts/run_phase1_calibration.py --cvat_xml data/annotations.xml --output_dir data/processed/`
- Ran on all 104 images: 104/104 pass QC, 0 rejected
- mm/pixel: mean 0.0984, std 0.0004 — scanner is extremely consistent
- Keypoint spec updated: added LB(8) and PB(9) as computed (not annotated) points
- `src/data/cvat_parser.py` will pick up skeleton landmarks automatically when Dr. exports them
- Outputs live at `data/processed/calibration.csv` and `data/processed/calibration_clean.csv`
