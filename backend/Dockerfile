FROM python:3.9-slim

# Install system library required by shapely (GDbus dependency for shapely.geometry)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-multipart opencv-python-headless shapely

# Copy into /app directly — no double-nesting.
# Before (COPY . ./backend/): created /app/backend/backend/app/  ← broken
# After  (COPY . .):          creates /app/app/                 ← correct
# Volume mount - ./backend:/app then maps host ./backend/ → /app
# but /app/app/ stays intact because /app/backend (not /app/app) is what gets shadowed.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]