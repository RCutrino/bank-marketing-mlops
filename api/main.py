"""
Bank Marketing Prediction API

Endpoints:
    GET  /health      → service status
    POST /predict     → predict subscription probability
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import ClientFeatures, HealthResponse, PredictionResponse
from src.predict import load_artifact, predict as run_predict

SCENARIO = os.getenv("MODEL_SCENARIO", "without_duration")
MODELS_DIR = os.getenv("MODELS_DIR", "models")

# Global artifact — loaded once at startup
_artifact: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifact
    _artifact = load_artifact(scenario=SCENARIO, models_dir=MODELS_DIR)
    print(f"Model loaded: {SCENARIO} | threshold={_artifact['threshold']:.3f}")
    yield
    _artifact = None


app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predicts whether a client will subscribe to a term deposit.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_artifact is not None,
        scenario=SCENARIO,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ClientFeatures):
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    result = run_predict(_artifact, features.model_dump())

    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        threshold=round(_artifact["threshold"], 4),
    )