"""
Drift monitoring — confronta distribuzione feature train vs prod.

Usa Kolmogorov-Smirnov test per feature numeriche,
Chi-square test per feature categoriche.

Usage:
    python -m monitoring.drift --train data/bank.csv --prod data/prod_sample.csv
    python -m monitoring.drift --train data/bank.csv --prod data/prod_sample.csv --threshold 0.05
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


NUMERIC_FEATURES = ["age", "balance", "campaign", "pdays", "previous", "day"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "default", "housing", "loan",
                        "contact", "month", "poutcome"]


# ---------------------------------------------------------------------------
# KS test — numerical features
# ---------------------------------------------------------------------------

def ks_test(train: pd.Series, prod: pd.Series) -> dict:
    """
    Kolmogorov-Smirnov test tra distribuzione train e prod.
    p-value basso (< threshold) → drift rilevato.
    """
    stat, p_value = stats.ks_2samp(train.dropna(), prod.dropna())
    return {"statistic": round(float(stat), 4), "p_value": round(float(p_value), 4)}


# ---------------------------------------------------------------------------
# Chi-square test — categorical features
# ---------------------------------------------------------------------------

def chi2_test(train: pd.Series, prod: pd.Series) -> dict:
    """
    Chi-square test sulla distribuzione delle categorie.
    Confronta frequenze relative train vs prod.
    p-value basso → drift rilevato.
    """
    all_categories = set(train.dropna().unique()) | set(prod.dropna().unique())

    train_counts = train.value_counts()
    prod_counts = prod.value_counts()

    train_freq = np.array([train_counts.get(c, 0) for c in all_categories], dtype=float)
    prod_freq = np.array([prod_counts.get(c, 0) for c in all_categories], dtype=float)

    # evita divisione per zero
    if prod_freq.sum() == 0 or train_freq.sum() == 0:
        return {"statistic": None, "p_value": None, "error": "empty distribution"}

    # normalizza a stessa scala
    prod_expected = prod_freq / prod_freq.sum() * train_freq.sum()
    prod_expected = np.where(prod_expected == 0, 1e-10, prod_expected)

    stat, p_value = stats.chisquare(f_obs=train_freq, f_exp=prod_expected)
    return {"statistic": round(float(stat), 4), "p_value": round(float(p_value), 4)}


# ---------------------------------------------------------------------------
# Main drift report
# ---------------------------------------------------------------------------

def run_drift_report(
    train_path: str,
    prod_path: str,
    threshold: float = 0.05,
    output_path: str | None = None,
) -> dict:
    """
    Esegue drift detection su tutte le feature.
    Ritorna dict con risultati per feature + summary.

    Args:
        train_path:  path CSV dati training (reference)
        prod_path:   path CSV dati produzione (current)
        threshold:   p-value sotto il quale si segnala drift (default 0.05)
        output_path: se fornito, salva report JSON
    """
    train_df = pd.read_csv(train_path)
    prod_df = pd.read_csv(prod_path)

    # rimuovi duration se presente — non disponibile in prod
    for df in [train_df, prod_df]:
        if "duration" in df.columns:
            df.drop(columns=["duration"], inplace=True)

    results = {}
    drifted = []

    # --- Numeriche ---
    for col in NUMERIC_FEATURES:
        if col not in train_df.columns or col not in prod_df.columns:
            continue
        test_result = ks_test(train_df[col], prod_df[col])
        test_result["test"] = "KS"
        test_result["drift"] = test_result["p_value"] < threshold
        results[col] = test_result
        if test_result["drift"]:
            drifted.append(col)

    # --- Categoriche ---
    for col in CATEGORICAL_FEATURES:
        if col not in train_df.columns or col not in prod_df.columns:
            continue
        test_result = chi2_test(train_df[col], prod_df[col])
        test_result["test"] = "chi2"
        test_result["drift"] = (
            test_result["p_value"] is not None and test_result["p_value"] < threshold
        )
        results[col] = test_result
        if test_result["drift"]:
            drifted.append(col)

    summary = {
        "threshold": threshold,
        "total_features": len(results),
        "drifted_features": len(drifted),
        "drifted": drifted,
        "drift_detected": len(drifted) > 0,
    }

    report = {"summary": summary, "features": results}

    # stampa summary
    print("\n===== DRIFT REPORT =====")
    print(f"Reference: {train_path}")
    print(f"Current:   {prod_path}")
    print(f"Threshold: p < {threshold}")
    print(f"\nDrift rilevato su {len(drifted)}/{len(results)} feature")
    if drifted:
        print(f"Feature con drift: {drifted}")
    else:
        print("Nessun drift rilevato.")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport salvato → {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path CSV dati training (reference)")
    parser.add_argument("--prod", required=True, help="Path CSV dati produzione (current)")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", default="monitoring/drift_report.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_drift_report(args.train, args.prod, args.threshold, args.output)