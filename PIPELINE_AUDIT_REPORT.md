# Pipeline Audit Report
**Singdent Cephalometric AI — Full Pipeline Stress Test**
_Executed: 2026-05-08 | Auditor: Lead QA Automation | Environment: Python 3.9.6 / .venv_

---

## Executive Summary

| Phase | Component | Status | Notes |
|---|---|---|---|
| Phase 1 | Data Parsing (`parse_annotations.py`) | ✅ PASS | After 1 bug fix |
| Phase 2 | Evaluation Logic (`evaluate.py`) | ✅ PASS | All assertions passed |
| Phase 3 | Clinical Integration (`biomechanics.py`) | ✅ PASS | JSON → engine verified |
| Phase 4 | Dashboard Import (`app.py`) | ✅ PASS | All imports resolve |

**Overall QA Verdict: MECHANICALLY ROBUST** — The full data pipeline is end-to-end functional. One critical pre-existing bug was found and fixed. No crashes, no unhandled exceptions, no missing imports.

---

## Bug Found and Fixed

### BUG-001: `SKELETON_LABEL` Mismatch in `cvat_parser.py` — CRITICAL

**File:** `src/data/cvat_parser.py` (line 32, pre-fix)

**Root cause:**
```python
# BEFORE (broken)
SKELETON_LABEL = "Incisor_Maxilla_Complex_Skeleton"

# AFTER (fixed)
SKELETON_LABEL = "Incisor_Maxilla_Complex"
```

**Impact:** The `_Skeleton` suffix does not appear anywhere in the CVAT XML export. Every `<skeleton>` element has `label="Incisor_Maxilla_Complex"`. Because the label never matched, the `elif tag == "skeleton" and label == SKELETON_LABEL:` branch was never entered for any image. All 10 keypoints were silently skipped for every record. The JSON output would have `"has_landmarks": false` and `"keypoints": []` (all zeros) for every image, causing a fatal crash at first training batch.

**Fix applied:** Changed `SKELETON_LABEL` to match the exact XML export label.

**Severity:** Critical (pipeline-blocking) — would have silently produced invalid training data with no error message.

**Rule to add to FAILURES.md:** Always verify that label constant strings in the parser exactly match the actual `label="..."` attribute values in the CVAT XML — a wrong constant silently no-ops rather than raising an exception.

---

## Architectural Observation (Not a Bug)

**Landmark naming convention gap between parser and engines:**

`cvat_parser.py` uses underscore names matching the CVAT schema (`Upper_tip`, `Upper_apex`, `Labial_midroot`, etc.). `biomechanics.py` and `evaluate.py` use a different convention (`Upper-tip`, `Upper-apex`, `Labial mid-root`, etc.).

This is not a crash bug — the parser and the evaluation engines operate on independent data structures — but any future integration layer that reads `landmarks_clean.json` and feeds it into Phase 2/3 must include an explicit name-mapping step. The Phase 3 integration test script demonstrated the required bridge:

```python
_NAME_MAP = {
    "Upper_tip": "Upper-tip",  "Upper_apex": "Upper-apex",
    "Labial_midroot": "Labial mid-root",  "Labial_crest": "Labial crest",
    "Palatal_midroot": "Palatal mid-root",  "Palatal_crest": "Palatal crest",
    "ANS": "ANS",  "PNS": "PNS",  "LB": "LB",  "PB": "PB",
}
```

**Recommendation:** Document this in `PIPELINE_PLAN.md` or standardise both sides to use one naming convention before the training loop connects Phase 1 JSON to Phase 2 dataset loading.

---

## Phase 1 — Data Parsing

**Command:** `python scripts/parse_annotations.py`

**Input:** `data/annotations.xml` — 2 images, dummy landmark plots (Patient01_T1.jpg, Patient01_T2.jpg)

**Output:** `data/processed/landmarks_clean.json`

```
INFO: Parsing CVAT XML: .../data/annotations.xml
INFO: Successfully parsed 2 image records.
INFO: Stats: 2 have calibration, 2 have landmarks, 2 have polygons
INFO: Saving cleanly parsed records to .../data/processed/landmarks_clean.json
INFO: Done!
```

**JSON integrity check:**

| Field | Image 0 (T1) | Image 1 (T2) |
|---|---|---|
| `has_calibration` | `true` | `true` |
| `has_landmarks` | `true` | `true` |
| `valid_mask` | `[1,1,1,1,1,1,1,1,1,1]` | `[1,1,1,1,1,1,1,1,1,1]` |
| Keypoints present | 10/10 visible | 10/10 visible |
| Polygons | Upper_incisor, Labial_bone, Palatal_bone | Upper_incisor, Palatal_bone |
| `treatment` | `["Uncontrolled_tipping"]` | `[]` |
| `quality_flags` | `[]` | `[]` |

**Note:** Image 1 (T2) has no `Labial_bone` polygon — this is correct; the XML does not annotate one for T2 at this stage. The parser handles missing polygons gracefully.

**Result: ✅ PASS**

---

## Phase 2 — Evaluation Logic

**Command:** `python src/phase2/evaluate.py`

```
======================================================================
  Phase 2 — Evaluation Metrics  |  Built-in Self-Test
======================================================================

[1] Mock data: 2 images × 10 landmarks
    Simulated offset: (6.0, 8.0) px  → 10.0000 px Euclidean  → 0.9840 mm expected MRE

[2] MRE: 0.9840 mm

[3] SDR (Successful Detection Rate):
    ≤ 2.0 mm  [██████████████████████████████████████████████████]  100.00 %
    ≤ 2.5 mm  [██████████████████████████████████████████████████]  100.00 %
    ≤ 3.0 mm  [██████████████████████████████████████████████████]  100.00 %
    ≤ 4.0 mm  [██████████████████████████████████████████████████]  100.00 %

[4] Running assertions …
    MRE value check passed  (expected ≈ 0.9840 mm)  ✓
    SDR has all 4 threshold keys  ✓
    All SDR values in [0, 100]  ✓
    SDR[2.0 mm] == 100.0 %  ✓  (fixed 0.984 mm offset < 2.0 mm threshold)
    SDR is monotonically non-decreasing  ✓

[5] Edge-case: partial overlap (one landmark missing from predictions) …
    Partial MRE = 0.1392 mm  (only 'Upper-tip' matched)  ✓

======================================================================
  All tests passed — evaluate.py is working correctly.
======================================================================
```

**Assertions verified:**
- MRE type is `float` and matches theoretical value to 9 decimal places ✓
- SDR dict has all 4 required threshold keys ✓
- All SDR percentages in range `[0.0, 100.0]` ✓
- SDR is monotonically non-decreasing ✓
- Partial-overlap edge case: missing landmarks silently skipped, no KeyError ✓

**Result: ✅ PASS — 5/5 assertions, 0 exceptions**

---

## Phase 3 — Clinical Logic Integration

**Method:** Loaded `data/processed/landmarks_clean.json`, extracted Image 0 (Patient01_T1) keypoints, applied name-mapping bridge, computed mm/px from embedded calibration polyline, passed to `calculate_metrics()` and `classify_treatment()`.

**Exact CLI output of the integration test:**

```
======================================================================
  Phase 3 Integration Test — JSON → Biomechanics Engine
======================================================================

Image     : Patient01_T1.jpg  (1729 × 2048 px)
Timepoint : T1  |  Patient: Patient01
Calibration: 305.16 px span → 0.0983 mm/px

Landmarks loaded from JSON (10/10):
  Upper-tip              (  1310.53,   1497.50) px
  Upper-apex             (  1198.84,   1317.36) px
  Labial mid-root        (  1292.61,   1386.46) px
  Labial crest           (  1308.60,   1441.23) px
  Palatal mid-root       (  1240.45,   1428.57) px
  Palatal crest          (  1272.15,   1458.01) px
  ANS                    (  1196.81,   1178.77) px
  PNS                    (   704.50,   1210.75) px
  LB                     (  1213.28,   1403.68) px
  PB                     (  1261.24,   1353.58) px

----------------------------------------------------------------------
calculate_metrics() output:
  U1-PP angle       :  118.08 °
  LB-Apex distance  :   8.604 mm
  PB-Apex distance  :   7.093 mm

----------------------------------------------------------------------
classify_treatment() output:
  Root apex position       : Palatal
  Incisor condition        : Proclined incisor with apex near palatal bone
  Preferred biomechanics   : Controlled tipping during retraction with apex control
  Biomechanics to avoid    : Retraction causing further palatal displacement of apex
  Clinical implication     : Retraction possible but avoid excessive palatal pressure

======================================================================
  INTEGRATION TEST PASSED — JSON data flows cleanly to Phase 3 engine.
======================================================================
```

**Commentary on output values:** As expected for dummy/random landmark plots, the clinical result is an edge-case scenario (>115° angle, Palatal apex). These values are physically unrealistic for actual orthodontic use — this is consistent with the random annotation source. The engine handled the edge case correctly and produced a valid Zhang et al. 2021 classification entry without any exception.

**Data flow verified:**
- `landmarks_clean.json` → name-mapped dict → `calculate_metrics()` → `classify_treatment()` ✓
- All 3 metric keys returned: `u1_pp_angle_deg`, `lb_apex_dist_mm`, `pb_apex_dist_mm` ✓
- All 5 classification keys returned: `Root apex position`, `Incisor condition`, `Preferred biomechanics`, `Biomechanics to avoid`, `Clinical implication` ✓
- `temp_test_phase3.py` deleted after run ✓

**Result: ✅ PASS — End-to-end JSON→Phase3 integration confirmed**

---

## Phase 4 — Dashboard Import

**Command:** `python -c "import app"` (inside `.venv`)

**Exit code:** 0 (clean exit, no exception)

**Output summary:** Streamlit emitted `ScriptRunContext` warnings — this is the documented, expected behavior when Streamlit UI commands (`st.set_page_config()`, `st.markdown()`, `st.sidebar`, etc.) are executed outside a `streamlit run` server context. Streamlit 1.35+ no-ops gracefully rather than raising an exception.

**No Python exceptions of any kind:**
- No `ImportError` ✓
- No `ModuleNotFoundError` ✓
- No `SyntaxError` ✓
- No `AttributeError` ✓
- No `NameError` ✓

**Third-party imports confirmed resolving:**
`streamlit`, `cv2` (opencv), `PIL` (pillow), `numpy`, `src.phase3.biomechanics`

**Result: ✅ PASS — All imports clean, no broken dependencies**

---

## Final QA Verdict

```
╔══════════════════════════════════════════════════════════════════════╗
║                   PIPELINE: MECHANICALLY ROBUST                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Phase 1 (Parsing)    ✅ PASS — 2/2 images, 10/10 keypoints each    ║
║  Phase 2 (Metrics)    ✅ PASS — 5/5 assertions, all SDR correct     ║
║  Phase 3 (Clinical)   ✅ PASS — JSON→engine integration proven      ║
║  Phase 4 (Dashboard)  ✅ PASS — all imports resolve clean           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Bugs fixed : 1  (SKELETON_LABEL mismatch — critical, silent drop)  ║
║  Bugs open  : 0                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

The pipeline is ready to process real annotated data from Dr. as soon as it arrives. The single pre-existing bug (`SKELETON_LABEL` constant) has been corrected and logged. The naming convention gap between `cvat_parser.py` and the Phase 2/3 engines is documented and requires a mapping bridge in the training loop connector, but does not affect any currently operational code path.

**Next action when Dr. delivers the annotated CVAT XML:**
1. Replace `data/annotations.xml`
2. `python scripts/parse_annotations.py`
3. `python scripts/run_phase2_train.py --debug --max-images 10`
4. Full LOPO training run

---
_Generated by Full Pipeline Stress Test · 2026-05-08_
