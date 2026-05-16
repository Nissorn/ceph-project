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

## 2026-05-07 — train.py read landmarks_clean.json as dict, not list

**Bug:** `src/phase2/train.py:113` — `landmarks_data["images"]` (treating the JSON as `{"images": [...]}`)  
**Reality:** `parse_annotations.py` saves records as a bare JSON array `[...]`, not a dict  
**Impact:** `TypeError: list indices must be integers or slices, not str` — crash before any training starts  
**Fix:** Changed `landmarks_data["images"]` → `landmarks_data` (treat loaded JSON as list directly)  
**Rule:** Always match the exact save format. `parse_annotations.py` calls `json.dump(records, f)` — that is a list.

---

## 2026-05-07 — run_pipeline.py same JSON dict assumption + wrong key name

**Bug 1:** `scripts/run_pipeline.py:36` — `json.load(f)["images"]` — same issue as train.py above  
**Bug 2:** `scripts/run_pipeline.py:84` — `rec_t2["file_name"]` (underscore) — schema uses `"filename"`  
**Impact:** Both crash before producing any output  
**Fix:** Line 36 → `json.load(f)`; Line 84 → `rec_t2["filename"]`  
**Rule:** Never assume a wrapper dict around the JSON array. Check the save call in parse_annotations.py.

---

## 2026-05-07 — requirements.txt missing iterstrat

**Bug:** `src/data/splits.py` imports `from iterstrat.ml_stratifiers import ...` but `iterstrat` was not in requirements.txt  
**Impact:** `pip install -r requirements.txt` does not install splits dependency — silent failure until splits.py is first called  
**Fix:** Added `iterstrat>=2.0.0` to requirements.txt  
**Rule:** When writing a new import, add the package to requirements.txt in the same commit.

---

## 2026-05-07 — Python 3.10+ union type syntax used in Python 3.9 venv

**Bug:** `metrics.py:52` and `train.py:107` used `X | None` union syntax (PEP 604, requires Python 3.10+)  
**Reality:** Project venv is Python 3.9.6 — raises `TypeError: unsupported operand type(s) for |`  
**Fix:** Added `from typing import Optional` and replaced `X | None` → `Optional[X]` in both files  
**Rule:** Always use `Optional[X]` from `typing` until venv is explicitly upgraded past Python 3.10.

---

## 2026-05-05 — Calibration expected range was wrong

**Tried:** context.md said expected range was ~0.04–0.06 mm/pixel  
**Reality:** Actual values are 0.0974–0.0990 (≈ 0.1 mm/pixel)  
**Fix applied:** context.md and STATUS.md updated  
**Rule:** Valid QC range is [0.05, 0.30]. Actual dataset sits at ~0.099 — not 0.04.

---

## 2026-05-08 — SKELETON_LABEL constant did not match XML label (silent keypoint drop)

**Bug:** `src/data/cvat_parser.py:32` — `SKELETON_LABEL = "Incisor_Maxilla_Complex_Skeleton"`
**Reality:** CVAT XML exports `label="Incisor_Maxilla_Complex"` (no `_Skeleton` suffix).
**Impact:** The `elif tag == "skeleton" and label == SKELETON_LABEL:` branch never matched. All 10 keypoints were silently dropped for every image. `has_landmarks` remained `false` in all JSON records. Would have caused a crash or trained on zeroed coordinates without any error message.
**Fix:** Changed constant to `"Incisor_Maxilla_Complex"`. Found and fixed during 2026-05-08 QA audit.
**Rule:** Always verify parser label constants against the actual `label="..."` attribute in the CVAT XML — a wrong constant silently no-ops rather than raising an exception.

---

## 2026-05-08 — cvat_parser.py uses Python 3.10+ union type syntax in Python 3.9 venv

**Bug:** `cvat_parser.py:72` — `calibration_pts list[tuple[float,float]] | None` union syntax (PEP 604, requires Python 3.10+)  

**Reality:** Project venv is Python 3.9.6 — raises `TypeError: unsupported operand type(s) for |`  

**Fix:** Added `from typing import Optional` and replaced `list[tuple[float,float]] | None` → `Optional[list[tuple[float,float]]]`  

**Rule:** Always use `Optional[X]` from `typing` until venv is explicitly upgraded past Python 3.10.

---

## 2026-05-15 — SoftArgmax2D temperature=0.1 collapses predictions to center

**Bug:** `src/phase2/heatmap.py` SoftArgmax2D — `temperature=0.1` in soft-argmax makes weight ratio peak/base = exp(0.105) ≈ 1.11 — essentially uniform. Soft-argmax returns spatial average (center of heatmap), not peak location.

**Root cause (two compounding issues):**
1. `temperature=0.1` in validation decode → soft-argmax nearly uniform → MRE ~10mm looked plausible but was completely wrong
2. `self.beta = nn.Parameter(torch.tensor(temperature))` in SoftArgmax2D made beta LEARNABLE — backprop pushed beta back toward 0.1 regardless of initialization

**Fix applied:**
1. Changed `SoftArgmax2D.beta` from `nn.Parameter` → `register_buffer` (non-learnable)
2. Changed validation/training decode temperature to 10.0 (from 0.1)
3. Early stopping now uses argmax MRE, not soft-argmax MRE

**Verification:** Run `python debug_coords.py` on server — argmax predictions should vary per image (not all 256.0,256.0)

---

## 2026-05-15 — Loss normalization crushed gradients by ~1000×

**Bug:** `src/phase2/loss.py:107` — `loss = mse / (n_valid * H * W)` dividing by 2,621,440 → gradients ~0.001× true value.

**Fix applied:** Changed to `/ n_valid` only — MSE per-sample averaged, not per-pixel.

---

## 2026-05-16 — 92 images fundamentally insufficient for HRNet-W32

**Reality:** HRNet-W32 has 28M+ parameters. 73 training images per fold cannot provide sufficient generalization. All regularization (weight decay, augmentation) only SLOWS overfitting — does not prevent it.

**Current baseline results (v6 with heavy regularization):**
- Fold 1: best argmax MRE = 31.15mm (early stopped at epoch 20)
- Fold 2: best argmax MRE = 36.51mm (still running, noisy improvement)

**User's plan:** Acquiring 300+ clinical images. With proper data volume, heatmap regression IS the correct approach.

**What NOT to do:** Do NOT switch to direct coordinate regression. Heatmap regression preserves spatial resolution and is the medical imaging industry standard for landmark detection. User explicitly rejected this direction.