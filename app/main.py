"""
Crowd Counting API - FastAPI Application

Provides REST endpoints for crowd counting using two methods:
1. Density Map (CSRNet) - Estimates crowd via density map summation
2. Localization (P2PNet) - Detects individual head positions
"""

import os
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.inference.density_inference import (
    load_model as load_density_model,
    predict_density
)
from app.inference.localization_inference import (
    load_model as load_localization_model,
    predict_localization
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DENSITY_WEIGHTS = os.path.join(WEIGHTS_DIR, "densitymap_model.pth")
LOCALIZATION_WEIGHTS = os.path.join(WEIGHTS_DIR, "p2pnet_model.pth")

# Application state
models_loaded = {"density": False, "localization": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global models_loaded
    
    print("=" * 50)
    print("🚀 Crowd Counting API Starting...")
    print("=" * 50)
    
    # Load density model
    if os.path.exists(DENSITY_WEIGHTS):
        try:
            load_density_model(DENSITY_WEIGHTS, quantize=True)
            models_loaded["density"] = True
            print("✅ Density Map model loaded")
        except Exception as e:
            print(f"❌ Failed to load Density Map model: {e}")
    else:
        print(f"⚠️ Density weights not found: {DENSITY_WEIGHTS}")
    
    # Load localization model
    if os.path.exists(LOCALIZATION_WEIGHTS):
        try:
            load_localization_model(LOCALIZATION_WEIGHTS, quantize=False)
            models_loaded["localization"] = True
            print("✅ Localization model loaded")
        except Exception as e:
            print(f"❌ Failed to load Localization model: {e}")
    else:
        print(f"⚠️ Localization weights not found: {LOCALIZATION_WEIGHTS}")
    
    print("=" * 50)
    print("🎉 API Ready to serve requests!")
    print("=" * 50)
    
    yield
    
    # Cleanup
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Crowd Counting API",
    description="AI-powered crowd counting using Density Map and Point Localization methods",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment.
    
    Returns 200 OK as long as the app is running, regardless of model status.
    This is important for Fly.io to detect the app is alive during slow model loading.
    """
    all_models_ready = all(models_loaded.values())
    return {
        "status": "healthy" if all_models_ready else "starting",
        "ready": all_models_ready,
        "models": models_loaded
    }


@app.post("/predict/density")
async def predict_density_endpoint(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)")
):
    """
    Predict crowd count using Density Map method.
    
    Returns count and density map visualization.
    """
    if not models_loaded["density"]:
        raise HTTPException(
            status_code=503,
            detail="Density Map model not available"
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG/PNG)"
        )
    
    try:
        # Read file
        contents = await file.read()
        image_bytes = io.BytesIO(contents)
        
        # Run inference
        result = predict_density(image_bytes)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/localization")
async def predict_localization_endpoint(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    threshold: float = Query(0.45, ge=0.0, le=1.0, description="Detection confidence threshold")
):
    """
    Predict crowd count using Point Localization method.
    
    Returns count, head positions, and visualization.
    """
    if not models_loaded["localization"]:
        raise HTTPException(
            status_code=503,
            detail="Localization model not available"
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG/PNG)"
        )
    
    try:
        # Read file
        contents = await file.read()
        image_bytes = io.BytesIO(contents)
        
        # Run inference
        result = predict_localization(image_bytes, threshold=threshold)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/api/info")
async def api_info():
    """Get API information and available methods."""
    return {
        "name": "Crowd Counting API",
        "version": "1.0.0",
        "methods": [
            {
                "id": "density",
                "name": "Density Map",
                "description": "Estimates crowd count by generating a density map",
                "available": models_loaded["density"]
            },
            {
                "id": "localization",
                "name": "Point Localization",
                "description": "Detects individual head positions in the crowd",
                "available": models_loaded["localization"]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
