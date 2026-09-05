"""
Inference helpers.
Loaded once at API startup; called per request.
"""

import joblib
import pandas as pd
from pathlib import Path


def load_artifact(scenario: str = "without_duration", models_dir: str = "models"):
    path = Path(models_dir) / f"model_{scenario}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run src/train.py first.")
    return joblib.load(path)  # {"pipeline": ..., "threshold": ...}


def predict(artifact: dict, input_data: dict) -> dict:
    """
    Args:
        artifact: dict with keys 'pipeline' and 'threshold'
        input_data: raw feature dict (pre-encoding, as received from API)
    Returns:
        {"prediction": int, "probability": float}
    """
    df = pd.DataFrame([input_data])
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    proba = pipeline.predict_proba(df)[0, 1]
    prediction = int(proba >= threshold)

    return {"prediction": prediction, "probability": round(float(proba), 4)}