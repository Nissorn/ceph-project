from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.v1.endpoints import router as api_router

app = FastAPI(
    title="Cephalometric Landmark Detection API",
    description="API for automated cephalometric analysis and biomechanics calculation.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}

app.include_router(api_router, prefix="/api/v1", tags=["analysis"])