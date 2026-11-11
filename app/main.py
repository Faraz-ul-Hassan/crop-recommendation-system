from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import CropInput, PredictionResponse, TopPrediction

# --- Paths ---
# resolve the project root relative to THIS file (app/main.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "app" / "static"

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# --- Globals for loaded artifacts ---
model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + scaler once at startup, release at shutdown."""
    global model, scaler
    model = joblib.load(MODEL_DIR / "crop_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    print(f"[startup] Model loaded from {MODEL_DIR / 'crop_model.pkl'}")
    print(f"[startup] Scaler loaded from {MODEL_DIR / 'scaler.pkl'}")
    yield
    model = None
    scaler = None
    print("[shutdown] Model and scaler released.")

app = FastAPI(
    title="Crop Recommendation API",
    description="Recommends the most suitable crop based on soil nutrients and environmental conditions.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Health / info endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# --- Prediction endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CropInput):
    # 1. build DataFrame with the SAME column order used in training
    features = pd.DataFrame(
        [[payload.N, payload.P, payload.K, payload.temperature, payload.humidity, payload.ph, payload.rainfall]],
        columns=FEATURE_ORDER
    )

    # 2. scale with the exact scaler used at training time
    scaled = scaler.transform(features)

    # 3. Predict probabilities
    proba = model.predict_proba(scaled)[0]

    # 4. top-3 crops by probability (descending)
    top_idx = np.argsort(proba)[::-1][:3]
    top_3 = [
        TopPrediction(crop=str(model.classes_[i]), probability=float(proba[i]))
        for i in top_idx
    ]

    return PredictionResponse(
        predicted_crop=str(model.classes_[top_idx[0]]),
        confidence=float(proba[top_idx[0]]),
        top_3=top_3,
    )

# --- Frontend serving (only if app/static/ exists) ---
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

