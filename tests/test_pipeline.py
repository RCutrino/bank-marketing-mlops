import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.pipeline import build_preprocessor, build_full_pipeline


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "balance": np.random.randint(-500, 5000, n),
        "pdays": np.random.choice([-1, 0, 10, 30], n),
        "campaign": np.random.randint(1, 10, n),
        "job": np.random.choice(["management", "blue-collar", "technician"], n),
        "marital": np.random.choice(["married", "single", "divorced"], n)
        })
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


@pytest.fixture
def feature_lists(sample_data):
    X, _ = sample_data
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(include="object").columns.tolist()
    return num, cat


def test_preprocessor_fit_transform(sample_data, feature_lists):
    """Il preprocessor deve trasformare X mantenendo lo stesso numero di righe.
    Le colonne devono aumentare rispetto alle numeriche originali per via dell'OHE sulle categoriche."""
    X, y = sample_data
    num, cat = feature_lists
    preprocessor = build_preprocessor(num, cat)
    X_out = preprocessor.fit_transform(X)
    assert X_out.shape[0] == len(X)
    assert X_out.shape[1] > len(num)


def test_full_pipeline_fit_predict(sample_data, feature_lists):
    """La pipeline completa (preprocessor + modello) deve fittare senza errori
    e produrre predizioni binarie (0 o 1) per ogni osservazione."""
    X, y = sample_data
    num, cat = feature_lists
    preprocessor = build_preprocessor(num, cat)
    pipeline = build_full_pipeline(preprocessor, RandomForestClassifier(n_estimators=10, random_state=42))
    pipeline.fit(X, y)
    check_is_fitted(pipeline)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_full_pipeline_predict_proba(sample_data, feature_lists):
    """predict_proba deve restituire probabilità per 2 classi che sommano a 1.0 per ogni riga."""
    X, y = sample_data
    num, cat = feature_lists
    preprocessor = build_preprocessor(num, cat)
    pipeline = build_full_pipeline(preprocessor, RandomForestClassifier(n_estimators=10, random_state=42))
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_preprocessor_handles_unseen_categories(sample_data, feature_lists):
    """Il preprocessor deve gestire categorie mai viste in training senza errori
    grazie a handle_unknown='ignore' nell'OHE."""
    X, y = sample_data
    num, cat = feature_lists
    preprocessor = build_preprocessor(num, cat)
    preprocessor.fit(X, y)

    X_new = X.copy()
    X_new.loc[0, "job"] = "astronaut"  # categoria mai vista in training
    out = preprocessor.transform(X_new)
    assert out.shape[0] == len(X_new)