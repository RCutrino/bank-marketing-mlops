"""
Training entry point with MLflow tracking.

Usage:
    python -m src.train                        # default: without_duration
    python -m src.train --scenario with_duration
    python -m src.train --data path/to/bank.csv
    python -m src.train --experiment my_experiment
"""

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,
                            precision_recall_curve,
                            recall_score,
                            roc_auc_score)
from sklearn.model_selection import (GridSearchCV,
                                    StratifiedKFold,
                                    cross_val_score,
                                    train_test_split)

from src.pipeline import build_full_pipeline, build_preprocessor

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_RECALL = 0.75
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_features(df: pd.DataFrame, scenario: str):
    drop_cols = ["deposit"] if scenario == "with_duration" else ["deposit", "duration"]
    X = df.drop(columns=drop_cols)
    y = df["deposit"]
    return X, y


def make_splits(X: pd.DataFrame, y: pd.Series):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.4, 
        stratify=y, 
        random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        stratify=y_temp, 
        random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_train, X_val, X_test, y_train, y_val, y_test, scenario: str):
    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_train.select_dtypes(include="object").columns.tolist()

    preprocessor = build_preprocessor(num_features, cat_features)

    # --- Logistic Regression ---
    pipeline_lr = build_full_pipeline(
        preprocessor,
        LogisticRegression(max_iter=1000, 
                          random_state=RANDOM_STATE, 
                          class_weight="balanced"))
    pipeline_lr.fit(X_train, y_train)

    # --- Random Forest + GridSearch ---
    pipeline_rf_base = build_full_pipeline(
        preprocessor,
        RandomForestClassifier(random_state=RANDOM_STATE))
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__class_weight": ["balanced", None]
        }
    grid_search = GridSearchCV(pipeline_rf_base, 
                              param_grid, 
                              cv=5, 
                              scoring="roc_auc", 
                              n_jobs=-1)
    print(f"Running GridSearchCV [{scenario}]...")
    grid_search.fit(X_train, y_train)
    pipeline_rf = grid_search.best_estimator_

    # --- Validation ---
    roc_lr = roc_auc_score(y_val, pipeline_lr.predict_proba(X_val)[:, 1])
    roc_rf = roc_auc_score(y_val, pipeline_rf.predict_proba(X_val)[:, 1])

    print(f"Val ROC-AUC  LR={roc_lr:.3f}  RF={roc_rf:.3f}")
    print(f"Best RF params: {grid_search.best_params_}")

    best_pipeline = pipeline_rf if roc_rf >= roc_lr else pipeline_lr
    best_name = "RandomForest" if roc_rf >= roc_lr else "LogisticRegression"
    print(f"Selected: {best_name}")

    # --- Cross-validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best_pipeline, 
                                X_train, y_train, 
                                cv=cv, 
                                scoring="roc_auc", 
                                n_jobs=-1)
    print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # --- Threshold tuning ---
    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    default_recall = recall_score(y_val, (y_val_prob >= 0.5).astype(int))

    if default_recall >= TARGET_RECALL:
        threshold = 0.5
        print("Default threshold (0.5) satisfies recall constraint.")
    else:
        precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_val, y_val_prob)
        candidates = [(p, r, t) for p, r, t in zip(precision_arr, 
                                                  recall_arr, 
                                                  list(thresholds_arr) + [1.0])
                                  if r >= TARGET_RECALL]
        threshold = max(candidates, key=lambda x: x[0])[2] if candidates else 0.5

    print(f"Threshold: {threshold:.3f}")

    # --- Test evaluation ---
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)
    test_roc_auc = roc_auc_score(y_test, y_test_prob)
    report = classification_report(y_test, y_test_pred, output_dict=True)

    print("\n--- Test Performance ---")
    print(classification_report(y_test, y_test_pred))
    print(f"ROC-AUC: {test_roc_auc:.3f}")

    metrics = {
        "val_roc_auc_lr": roc_lr,
        "val_roc_auc_rf": roc_rf,
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std(),
        "test_roc_auc": test_roc_auc,
        "threshold": threshold,
        # per-class metrics from classification report
        "test_precision_0": report["0"]["precision"],
        "test_recall_0": report["0"]["recall"],
        "test_f1_0": report["0"]["f1-score"],
        "test_precision_1": report["1"]["precision"],
        "test_recall_1": report["1"]["recall"],
        "test_f1_1": report["1"]["f1-score"],
        "test_accuracy": report["accuracy"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "test_weighted_f1": report["weighted avg"]["f1-score"]
        }

    best_params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}

    return best_pipeline, threshold, best_name, metrics, best_params


# ---------------------------------------------------------------------------
# Save local artifact
# ---------------------------------------------------------------------------

def save_model(pipeline, threshold: float, scenario: str) -> Path:
    artifact = {"pipeline": pipeline, "threshold": threshold}
    path = MODELS_DIR / f"model_{scenario}.pkl"
    joblib.dump(artifact, path)
    print(f"Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_to_mlflow(pipeline,
                  threshold: float,
                  best_name: str,
                  metrics: dict,
                  best_params: dict,
                  scenario: str,
                  model_path: Path,
                  experiment_name: str):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{best_name}_{scenario}"):

        # --- Tags ---
        mlflow.set_tags({
            "scenario": scenario,
            "model_type": best_name,
            "target_recall": TARGET_RECALL
            })

        # --- Params ---
        mlflow.log_params(best_params)
        mlflow.log_param("scenario", scenario)
        mlflow.log_param("threshold", round(threshold, 4))
        mlflow.log_param("random_state", RANDOM_STATE)

        # --- Metrics ---
        mlflow.log_metrics(metrics)

        # --- Artifact: raw pkl ---
        mlflow.log_artifact(str(model_path), artifact_path="model_pkl")

        # --- Artifact: classification report as JSON ---
        with tempfile.NamedTemporaryFile(mode="w", 
                                        suffix=".json", 
                                        delete=False, 
                                        prefix="clf_report_") as f:
            json.dump(
                {k: v for k, v in metrics.items() if k.startswith("test_")},
                f,
                indent=2)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="reports")

        # --- sklearn model (MLflow native format → enables model registry) ---
        mlflow.sklearn.log_model(
          sk_model=pipeline,
          artifact_path="sklearn_model",
          registered_model_name=f"bank_marketing_{scenario}",
          input_example=None,
          skops_trusted_types=[
            "numpy.dtype",
            "sklearn.compose._column_transformer._RemainderColsList",
            "src.features.CustomPreprocessor"])

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run_id: {run_id}")
        print(f"Model registered as: bank_marketing_{scenario}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",
                        choices=["with_duration", "without_duration"],
                        default="without_duration")
    parser.add_argument("--data", default="data/bank.csv")
    parser.add_argument("--experiment", default="bank_marketing_classification")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_data(args.data)
    X, y = split_features(df, args.scenario)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    pipeline, threshold, best_name, metrics, best_params = train(
        X_train, X_val, X_test, 
        y_train, y_val, y_test, 
        args.scenario)
    model_path = save_model(pipeline, threshold, args.scenario)

    log_to_mlflow(pipeline, 
                  threshold, 
                  best_name, 
                  metrics, 
                  best_params,
                  args.scenario,
                  model_path, 
                  args.experiment)