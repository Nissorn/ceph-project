# FAILURES.md — Lessons Learned

Read this before starting any task. Each entry is a mistake that already happened — do not repeat it.

---

## 2026-05-05 — Wrong annotation XML path

**Tried:** `data/raw/annotations/annotations.xml` (what config.yaml said)  
**Reality:** XML lives at `data/annotations.xml` (project root data/ folder)  
**Fix applied:** config.yaml `annotation_file` corrected to `data/annotations.xml`  
**Rule:** Actual data file locations may differ from config — always `ls data/` first.

---

## 2026-05-05 — src/phase1/ is legacy, do not use

**Tried:** Referencing `src/phase1/` for active Phase 1 code  
**Reality:** `src/phase1/` is an earlier scaffold that was superseded. Active data layer is `src/data/`  
**Rule:** Use `src/data/` for all parsing/calibration. `src/phase1/` can be deleted when convenient.

---

## 2026-05-05 — LB and PB are CVAT-annotated, not computed

**Tried:** Documenting LB(8) and PB(9) as "geometrically computed in Phase 3/4"  
**Reality:** LB and PB ARE annotated as skeleton keypoints in CVAT v2 export (same skeleton label).  
  The v1 export shared in Session 1 only had 8 keypoints — the latest CVAT export has 10.  
**Rule:** KEYPOINT_NAMES has 10 entries (0–9). Never revert to 8. Always expect LB and PB in skeleton.

---

## 2026-05-06 — dataset.py uses wrong key for filename

**Bug:** `src/phase2/dataset.py:88` — `rec["file_name"]` (underscore)  
**Reality:** `cvat_parser.py` stores the key as `rec["filename"]` (no underscore)  
**Impact:** KeyError crash at first training batch — never reached because annotations blocked, but will break immediately when training starts  
**Fix:** Change `rec["file_name"]` → `rec["filename"]` in dataset.py `__getitem__`  
**Rule:** Key names between parser output and dataset must match exactly — check the parser schema before writing any `rec[...]` access.

---

## 2026-05-05 — Calibration expected range was wrong

**Tried:** context.md said expected range was ~0.04–0.06 mm/pixel  
**Reality:** Actual values are 0.0974–0.0990 (≈ 0.1 mm/pixel)  
**Fix applied:** context.md and STATUS.md updated  
**Rule:** Valid QC range is [0.05, 0.30]. Actual dataset sits at ~0.099 — not 0.04.
