import numpy as np
import pandas as pd
import pytest

from src.features import CustomPreprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "pdays": [-1, 0, 5, -1, 200],
        "balance": [-500, 0, 1500, -100, 3000],
        "age": [25, 40, 35, 50, 60]
        })


def test_pdays_minus1_replaced(sample_df):
    """pdays == -1 significa 'mai contattato' → deve essere rimpiazzato con 0."""
    out = CustomPreprocessor().fit_transform(sample_df)
    assert (out["pdays"] == -1).sum() == 0


def test_pdays_positive_unchanged(sample_df):
    """pdays > 0 sono giorni reali dall'ultimo contatto → non devono essere modificati."""
    out = CustomPreprocessor().fit_transform(sample_df)
    assert out.loc[2, "pdays"] == 5
    assert out.loc[4, "pdays"] == 200


def test_balance_negative_clipped(sample_df):
    """Balance negativo (debito) viene clippato a 0 — outlier non informativo per il modello."""
    out = CustomPreprocessor().fit_transform(sample_df)
    assert (out["balance"] < 0).sum() == 0


def test_balance_positive_unchanged(sample_df):
    """Balance positivo rappresenta ricchezza reale → deve rimanere invariato."""
    out = CustomPreprocessor().fit_transform(sample_df)
    assert out.loc[2, "balance"] == 1500
    assert out.loc[4, "balance"] == 3000


def test_other_columns_untouched(sample_df):
    """Colonne non in scope (es. age) non devono essere modificate dal transformer."""
    out = CustomPreprocessor().fit_transform(sample_df)
    pd.testing.assert_series_equal(out["age"], sample_df["age"])


def test_fit_returns_self(sample_df):
    """fit() deve ritornare self per compatibilità con sklearn Pipeline."""
    cp = CustomPreprocessor()
    assert cp.fit(sample_df) is cp


def test_no_mutation_of_input(sample_df):
    """Il DataFrame originale non deve essere mutato — il transformer lavora su una copia."""
    original = sample_df.copy()
    CustomPreprocessor().fit_transform(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)