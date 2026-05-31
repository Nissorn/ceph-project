# Ceph-V2 Auto — Cephalometric Landmark Detection & Segmentation

Orthodontic AI preprocessing pipeline: HRNet-W32 landmark detection (0.476mm MRE)
+ U-Net bone/tooth segmentation preprocessing.

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Nissorn/ceph-project.git
cd ceph-project
```

### 2. Obtain model weights

Model weights are **not** included in the repo (excluded via `.gitignore` — too large for git).

**Option A — From project owner:**
Obtain the following files and place them in the correct directories:
1. **Landmark Model Weights (HRNet-W32)**:
   - Place `fold1_best.pth` into: `data/processed/checkpoints/fold1_best.pth`
2. **Segmentation Model Weights (DeepLabV3Plus)**:
   - Place the folder `tversky_deepLabV3plus_resnet34_20250529_20260529_094221` (containing `best_model.pt`) into: `backend/models/`
3. **Calibration Data**:
   - Ensure `calibration.csv` is present at: `data/calibration.csv`

**Option B — Train from scratch:**
```bash
python src/phase2/train.py --config config.yaml --folds 5
```

### 3. Build & run with Docker

```bash
docker compose up --build -d
```

| Service | URL | Purpose |
|---------|-----|---------|
| Web (Astro SPA) | http://localhost:4321 | Upload X-ray, view landmarks & edit |
| API (FastAPI) | http://localhost:8123/api/v1/health | Inference health check (`{"status":"success"}`) |

**Frontend only (no backend):**
```bash
docker compose up --build web
```

---

## Deploying & Sharing on Other Devices

If you want to share this project with other developers or run it on a separate machine/device:

### 1. Bundle and Copy
Send the git repository codebase along with the model weights and data described below (which are gitignored):
- `data/processed/checkpoints/fold1_best.pth`
- `backend/models/tversky_deepLabV3plus_resnet34_20250529_20260529_094221/best_model.pt`
- `data/calibration.csv`

Ensure they are placed in the exact directory tree shown in **Section 2 (Obtain model weights)**.

### 2. Sharing within a Local Network (Multi-device access)
By default, the frontend is built targeting the API on `localhost:8123`. If you want other devices on your Wi-Fi/local network (like an iPad or another laptop) to access the application, you must pass the hosting machine's local IP address during build time:

1. Identify your local IP address (e.g. `192.168.1.150`).
2. Build and launch the containers with `VITE_API_URL` pointing to your IP:
   ```bash
   VITE_API_URL=http://192.168.1.150:8123/api/v1 docker compose up --build -d
   ```
3. Other devices can now navigate to `http://192.168.1.150:4321` to use the application.

---

### 4. Without Docker (Local Development)

**Backend:**
```bash
cd backend
pip install -r requirements.txt          # needs: torch, opencv, timm, etc.
uvicorn backend.app.main:app --reload    # starts on :8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev                              # starts on :4321
```

## Project Structure

```
ceph-v2-auto/
├── src/
│   ├── phase2/          # Landmark detection (HRNet-W32, TTA, hard-argmax)
│   ├── phase3/          # Segmentation preprocessing pipeline
│   └── utils/           # Config, IO helpers
├── backend/
│   ├── app/
│   │   ├── api/v1/      # POST /api/v1/analyze endpoint
│   │   └── services/    # InferenceService (torch, TTA, hard-argmax)
│   └── Dockerfile       # Multi-stage, non-root, HEALTHCHECK
├── frontend/
│   ├── src/
│   │   └── components/ui/
│   │       ├── DashboardApp.tsx   # Upload + Run AI Analysis
│   │       └── CephCanvasEditor.tsx  # Konva canvas, 3-tier confidence rings
│   └── Dockerfile       # Astro → static serve
├── docker-compose.yml   # api + web, volume-mounted weights
└── outputs/checkpoints/ # MODEL WEIGHTS — download separately
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Hard-argmax (not soft-argmax) | Soft-argmax has temperature bug; hard-argmax is deterministic |
| TTA: 5-variant (orig, rot±2°, brig±10%) | Scale variants excluded — geometric inverse produces ~100px errors |
| No horizontal flip | Cephalograms are anatomically directional |
| Simple `/255` normalization | NOT ImageNet mean/std — model trained with /255 only |
| 3-tier confidence rings | Red (<0.70 critical), Yellow (0.70–0.85 warning), Amber (≥0.85 normal) |
| Weights as volume mount | Not baked into image layer — per guardrail, keeps image lean |
| Letterbox 512×512 alignment | Matches landmark pipeline INPUT_SIZE for pixel-perfect alignment |
| Segmentation masks: uint8 class-ID | 0=bg, 1=bone, 2=tooth, 3=pulp — no float, no one-hot |

## API Reference

### POST `/api/v1/analyze`

**Request:** `multipart/form-data`
```
file: <X-ray image (JPEG/PNG/TIFF/BMP)>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "landmarks": {
      "Upper_tip":   {"x": 1303.5, "y": 1504.0, "confidence": 0.94},
      "Upper_apex":  {"x": 1180.2, "y": 1120.3, "confidence": 0.91},
      ...
    },
    "use_tta": true
  }
}
```

### GET `/api/v1/health`
```json
{"status": "ok", "message": "API is running"}
```

## Segmentation Preprocessing

```bash
# Generate aligned mask pairs from clinical annotations
python src/phase3/segmentation_preprocess.py preprocess \
  --annotations data/raw/segmentation/annotations.json \
  --images-dir  data/raw/images \
  --output-dir  data/processed

# Audit dataset readiness before training
python src/phase3/segmentation_preprocess.py audit \
  --images-dir data/processed/segmentation/images \
  --masks-dir  data/processed/segmentation/masks
```

## Known Limitations

- `outputs/checkpoints/` must be obtained separately from the project owner.
  Without it, the Docker healthcheck fails and the API returns 503.
- Docker build requires the ML dependencies (torch, opencv-python-headless, timm)
  which are large (~2–4 GB total). Build on a machine with sufficient bandwidth.
- Frontend `VITE_API_URL` is baked at build time — to change the backend URL
  after building, you must rebuild the web container.