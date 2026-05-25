FROM python:3.9-slim

# Install system library required by shapely (GDbus dependency for shapely.geometry)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-multipart opencv-python-headless shapely

# Copy into ./backend/ so the package resolves as backend.app.main
COPY . ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]