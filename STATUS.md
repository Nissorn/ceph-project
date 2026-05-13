# STATUS.md — Current State Snapshot
_Updated: 2026-05-12 (Microservices MVP complete, E2E verified, project paused awaiting clinical data)_

---

## ⛔ CURRENT BLOCKER — READ FIRST
**Training is strictly paused. Do NOT write or execute new AI training code until CVAT skeleton + polygon annotations are provided by Dr. Viet/Thủy.**

---

## Milestone: Microservices MVP Complete (2026-05-12)
**E2E Verification passed on 2026-05-12. All three subsystems confirmed working.**

### Architecture — Fully Migrated to Microservices
- **Backend:** FastAPI (`backend/`) on port 8000 — receives `multipart/form-data` image uploads, returns structured JSON (metrics, bone thickness, classification)
- **Frontend:** Astro 6.3.1 + React 19 + Tailwind CSS 4 (`frontend/`) on port 4321 — polished medical dashboard with Glassmorphism, Light/Dark mode, Singapodent Navy/Orange brand

### AI Pipeline Status
- **Phase 2 — HRNet Landmark Detection:** Scaffold complete, 10-keypoint system, AdaptiveWingLoss — untrained (awaiting annotations)
- **Phase 2b — U-Net Segmentation (SMP ResNet-34):** Smoke-tested — both dataloader (tensor shape [3,512,512]) and forward/backward pass (24,436,659 params, all with gradients) confirmed PASS
- **Phase 3 — Biomechanics Engine:** All self-tests PASS — U1-PP angle, LB/PB distances, Zhang et al. classification, BoneThicknessCalculator Plans A/B/C

### Frontend State
- MVP UI is **complete and production-built** (`npm run build` passes, 1 page in ~1s)
- Displays: raw X-ray preview before analysis; **interactive `CephCanvasEditor` after analysis**
- `CephCanvasEditor` (`frontend/src/components/ui/CephCanvasEditor.tsx`) — react-konva canvas editor:
  - 10 draggable keypoint markers (Sella, Nasion, Orbitale, Porion, A/B-points, Pogonion, Menton, ANS, PNS) with orange crosshair + name labels
  - 3 editable polygons (Maxillary Bone, Mandibular Bone, Cranial Base) with filled regions + vertex handles
  - **Drag** any vertex to reposition it (polygon updates live)
  - **Shift+Click** anywhere on a polygon edge → inserts new vertex at the nearest point on that edge
  - **Dbl-Click** or **Alt+Click** a vertex → deletes it (min 3 vertices enforced)
  - Image fits the container with correct aspect ratio; handles window resize
  - Default landmark positions are normalized fractions — replaced by real model output once training is complete
- CVAT external server deployment **abandoned** — replaced by this built-in editor
- Dependencies added: `konva`, `react-konva`

### venv Note
The `.venv` at `ceph-project/.venv` does not have `pip` pre-installed. Bootstrap with:
```bash
.venv/bin/python -m ensurepip && .venv/bin/python -m pip install -r requirements.txt
```
`segmentation-models-pytorch` is now installed in `.venv` (installed 2026-05-12).

---

## What is done
- **[P9 DONE]** Medical Logic Engine (Biomechanics) — `src/phase3/biomechanics.py`
  - U1-PP angle, LB-Apex distance, PB-Apex distance calculation
  - Zhang et al. 2021 classification (Labial/Midway/Palatal × <105/105-115/>115°)
  - Mock landmark generator + comprehensive built-in self-tests (all passing)
- **[P10 DONE]** Evaluation Metrics — `src/phase2/evaluate.py`
  - `calculate_mre()` — Mean Radial Error in mm across all images and landmarks
  - `calculate_sdr()` — Successful Detection Rate for thresholds [2.0, 2.5, 3.0, 4.0] mm
  - Robust to partially-annotated images; built-in self-tests all passing
- **[P11 DONE]** Streamlit MVP Dashboard — `app.py`
  - Medical dark-theme UI (Inter font, glassmorphism cards, custom CSS)
  - File uploader + 2-second mock inference spinner
  - OpenCV landmark visualisation with dynamic scaling to any image size
  - Draws U1 axis (Upper-tip→Upper-apex), palatal plane (ANS→PNS), LB–PB corridor
  - `st.metric` for angle/distances; styled cards for all 5 classification fields
  - Sidebar calibration input (mm/px), debug JSON expander, clinical disclaimer
- **[ARCHITECTURE]** Production Architecture Planning — `PRODUCTION_ARCHITECTURE_PLAN.md`
  - Detailed roadmap for microservices architecture with FastAPI backend
  - Plan for Astro frontend replacing Streamlit dashboard
  - Strategy for Docker containerization and CVAT Nuclio integration
- **[FASTAPI BACKEND]** FastAPI Infrastructure — `backend/`
  - Created routing, service logic importing `biomechanics.py`, Pydantic schemas, and endpoints.
- **[BONE THICKNESS]** Bone Thickness Architecture — `bone_thickness_architecture.md`
  - Design document for BoneThicknessCalculator class enhancement
- Phase 1 calibration: `scripts/run_phase1_calibration.py` — 104/104 pass QC → `data/processed/calibration.csv`
- `src/data/cvat_parser.py` — parses calibration + skeleton (10 keypoints) + polygons + tags
- mm/pixel: mean 0.0984, range [0.0974, 0.0990] — single scanner, extremely consistent
- **[P1 DONE]** Fixed 10-keypoint mismatch: `src/phase2/dataset.py` + `src/phase2/model.py` (NUM_KEYPOINTS=10)
- **[P2 DONE]** Created `scripts/parse_annotations.py` → generated `data/processed/landmarks_clean.json`
- **[INGEST DONE]** Created `scripts/ingest_editor_exports.py` to seamlessly merge manual annotations exported from the React Konva `CephCanvasEditor` into `data/processed/landmarks_clean.json` with strict keypoint order and mm/px calibration mapping.
- **[UI POLISH DONE]** Integrated wide banner asset (`logo.webp`) into Astro header layout with fluid aspect-ratio scaling and responsive grouping.
- **[P3 DONE]** Implemented `src/data/splits.py` — 5-fold MultilabelStratifiedKFold + 15% holdout, patient-aware
- **[P4 DONE]** Upgraded `src/phase2/augmentation.py` & `config.yaml` — ±10° via `A.Affine`, added ElasticTransform, GaussNoise, GridDistortion, Perspective (Albumentations 2.x clean, no warnings)
- **[P5 DONE]** Created `src/phase2/loss.py` — AdaptiveWingLoss (Wang et al. ICCV 2019); wired into `src/phase2/train.py` replacing MSELoss
- **[P6 SKIPPED]** Augmentation preview notebook deferred — requires at least 1 annotated image with visible keypoints
- **[P7 DONE]** Created `src/phase2b/segmentation.py` + `src/phase2b/segmentation_dataset.py` — U-Net scaffold with Dice+BCE loss and polygon→mask rasterisation
- **[P8 DONE]** Created `src/phase2c/classifier.py` + `src/phase2c/classifier_dataset.py` — EfficientNet-B3 multi-label scaffold with pos_weight computation, T1-only + Quality_Reject filtering
- Full pipeline plan written: `PIPELINE_PLAN.md` — input/output specs for all phases + 9 priorities
- **[AUDIT 2026-05-07]** Full code audit + 5 bugs fixed; dry-run pipeline test passes; `--debug` mode added to training script

## Planned Work (While Waiting for Annotations)
- **Production Infrastructure**: FastAPI backend infrastructure created (routes, models, services, main.py)
- **Bone Thickness Enhancement**: Implement enhancements based on `bone_thickness_architecture.md`
- **Microservices Migration**: Astro frontend initialized, components designed and assembled
- **Containerization**: Develop Docker images and orchestration configurations
- *(All P9–P11 planned items are now done. Awaiting Dr. annotations for model training.)*

## Dataset
- **Total:** 382 images (104 paired T1+T2 from 52 patients + 278 T1-only)
- **Incoming:** Dr. will provide 300+ additional ceph images
- **Annotations (current XML):** 2 images exported from CVAT — 0 skeleton landmarks, 2 polygons, 2 calibration
- **Calibration CSV:** 104/104 images calibrated (from earlier full export)

## What is next
- **Waiting on Dr.**: export CVAT XML with skeleton annotations — need 20+ for meaningful training
- **P6 (deferred):** `notebooks/04_augmentation_preview.ipynb` — needs ≥1 annotated image
- **Optional:** Install `segmentation-models-pytorch` when ready to train segmentation
- When annotations arrive: `python scripts/run_phase2_train.py --debug --max-images 10` to smoke-test, then full run

## Active blockers
- Dr. annotation: need 20+ skeleton annotations before meaningful training
- Phase 3 thresholds: tipping_threshold_deg and threshold_mm still null (pending Dr.)

## Known bugs remaining
- None. All bugs from audit logged in FAILURES.md and fixed.

## Key file locations
- Annotation XML: `data/annotations.xml` (NOT data/raw/annotations/)
- Active code: `src/data/` (NOT src/phase1/ — that is legacy)
- Config: `config.yaml` (annotation_file corrected, num_keypoints=10)
- Pipeline plan: `PIPELINE_PLAN.md`
- Production architecture: `PRODUCTION_ARCHITECTURE_PLAN.md` — roadmap for microservices migration
- Bone thickness architecture: `bone_thickness_architecture.md` — design document for enhanced measurements
- Failures log: `FAILURES.md` — read before starting any task
- Python venv: `venv/` — always `source venv/bin/activate` before running Python
- **Planned biomechanics:** `src/phase3/biomechanics.py`
- **Planned evaluation:** `src/phase2/evaluate.py`
- **Planned dashboard:** `app.py`