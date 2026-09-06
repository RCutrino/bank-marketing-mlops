import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


VALID_PAYLOAD = {
    "age": 41,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500.0,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
    }

MOCK_ARTIFACT = {
    "pipeline": MagicMock(**{
        "predict_proba.return_value": np.array([[0.35, 0.65]])
        }),
    "threshold": 0.414
    }


@pytest.fixture
def client():
    with patch("api.main.load_artifact", return_value=MOCK_ARTIFACT):
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health_ok(client):
    """GET /health deve rispondere 200 con status='ok' e model_loaded=True."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_valid_structure(client):
    """POST /predict con payload valido deve restituire i campi prediction, probability e threshold."""
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probability" in data
    assert "threshold" in data


def test_predict_prediction_is_binary(client):
    """Il campo prediction deve essere 0 o 1 — classificazione binaria."""
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.json()["prediction"] in (0, 1)


def test_predict_probability_in_range(client):
    """La probabilità deve essere un valore float compreso tra 0.0 e 1.0."""
    r = client.post("/predict", json=VALID_PAYLOAD)
    prob = r.json()["probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_missing_field(client):
    """Payload incompleto (campo age mancante) deve restituire 422 Unprocessable Entity."""
    payload = VALID_PAYLOAD.copy()
    del payload["age"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_age(client):
    """Age fuori range (< 18) deve fallire la validazione Pydantic con 422."""
    payload = VALID_PAYLOAD.copy()
    payload["age"] = 5
    r = client.post("/predict", json=payload)
    assert r.status_code == 422