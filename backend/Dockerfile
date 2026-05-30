# ── Stage 1: Build dependencies (pre-compile wheels for CPU) ──────────────────
FROM python:3.11-slim AS builder

# Install system library required by shapely (GDbus dependency for shapely.geometry)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install system deps for OpenCV headless + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps from requirements.txt
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Security: non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/false --no-create-home appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
     libgl1 \
     libglib2.0-0 \
     libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy pre-installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source — project root goes directly to /app/
# so that "backend.app.main" resolves to /app/backend/app/main.py
COPY --chown=appuser:appgroup . .

# ── Model weights — NOT baked into image layers ──────────────────────────────
# Mounted at runtime via docker-compose volumes (see ../docker-compose.yml).
# InferenceService reads from: /app/model-weights/checkpoints/
RUN mkdir -p /app/model-weights/checkpoints && chown -R appuser:appgroup /app

# Switch to non-root
USER appuser

EXPOSE 8000

# No HEALTHCHECK in Dockerfile — docker-compose.yml healthcheck is the authoritative one
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
