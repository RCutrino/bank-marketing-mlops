# Bank Marketing MLOps

End-to-end MLOps pipeline for predicting customer subscription to a term deposit.
Built on the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

---

## Problem Statement

A Portuguese bank runs phone marketing campaigns. The goal is to predict whether a client will subscribe to a term deposit **before** the call is made — without using `duration` (call length), which is only known after the call ends.

Excluding `duration` prevents data leakage and makes the model deployable in a real CRM context.

---

## Project Structure

```
bank-marketing-mlops/
├── src/
│   ├── features.py          # CustomPreprocessor (pdays, balance transforms)
│   ├── pipeline.py          # sklearn Pipeline builder
│   ├── train.py             # Training entry point + MLflow logging
│   └── predict.py           # Inference helpers
├── api/
│   ├── main.py              # FastAPI app — /health + /predict
│   └── schemas.py           # Pydantic input/output schemas
├── notebooks/
│   └── exploration.ipynb    # EDA + model evaluation (imports from src/)
├── tests/
│   ├── test_features.py     # Unit tests — CustomPreprocessor
│   ├── test_pipeline.py     # Unit tests — pipeline builder
│   └── test_api.py          # Integration tests — API endpoints
├── monitoring/
│   └── drift.py             # KS + chi-square drift detection
├── .github/workflows/
│   └── ci.yml               # GitHub Actions — pytest + Docker build
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## MLOps Phases

| Phase | Description | Tools |
|-------|-------------|-------|
| 1 | Refactor notebook → `src/` | Python, sklearn |
| 2 | Experiment tracking + model registry | MLflow |
| 3 | Prediction REST API | FastAPI, Pydantic |
| 4 | Containerisation | Docker, docker-compose |
| 5 | Testing + CI/CD | pytest, GitHub Actions |
| 6 | Data drift monitoring | scipy (KS-test, chi-square) |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# realistic scenario — without duration (default)
python -m src.train --data data/bank.csv

# benchmark — with duration (data leakage, not deployable)
python -m src.train --data data/bank.csv --scenario with_duration
```

Model saved to `models/model_without_duration.pkl`.

### 3. View MLflow experiments

```bash
mlflow ui
# open http://localhost:5000
```

### 4. Run the API

```bash
uvicorn api.main:app --reload --port 8000
# open http://localhost:8000/docs
```

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 41, "job": "management", "marital": "married",
    "education": "tertiary", "default": "no", "balance": 1500,
    "housing": "yes", "loan": "no", "contact": "cellular",
    "day": 15, "month": "may", "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown"
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.6523,
  "threshold": 0.414
}
```

### 5. Run with Docker

```bash
# build and start API + MLflow
docker-compose up --build

# API  → http://localhost:8000
# MLflow UI → http://localhost:5000
```

### 6. Run tests

```bash
pytest tests/ -v
```

### 7. Monitor data drift

```bash
python -m monitoring.drift \
  --train data/bank.csv \
  --prod data/prod_sample.csv \
  --output monitoring/drift_report.json
```

---

## Model Performance

Evaluated on hold-out test set (20% of data). Realistic scenario — `duration` excluded.

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.773 |
| CV ROC-AUC | 0.786 ± 0.011 |
| Recall (class 1) | 0.72 |
| F1 (macro) | 0.71 |
| Threshold | 0.414 |

Threshold tuned on validation set to maximise precision while keeping recall ≥ 0.75.

---

## Key Design Decisions

- **No `duration`**: prevents data leakage — model is deployable before the call.
- **Threshold tuning**: default 0.5 does not satisfy recall constraint → tuned on validation set.
- **Custom transformer**: `CustomPreprocessor` handles domain-specific transforms (`pdays=-1 → 0`, negative balance clipping) before sklearn `ColumnTransformer`.
- **Model registry**: MLflow registers each run as `bank_marketing_without_duration` for versioning.
- **Stateless API**: artifact loaded once at startup, reused per request.

---

## Dataset

[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) — 11,162 records, 17 features, binary target (`deposit`: yes/no).
