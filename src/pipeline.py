import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import CustomPreprocessor


def build_preprocessor(num_features: list[str], cat_features: list[str]) -> Pipeline:
    """
    Returns a fitted-ready sklearn preprocessor Pipeline.
    Steps:
      1. CustomPreprocessor  — domain transforms (pdays, balance)
      2. ColumnTransformer   — num: impute → log1p → scale | cat: impute → OHE
    """
    log_transformer = FunctionTransformer(np.log1p, 
                                          feature_names_out="one-to-one", 
                                          validate=False)

    num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                ("log", log_transformer),
                                ("scaler", StandardScaler())])

    cat_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))])

    col_transformer = ColumnTransformer(
        transformers=[("num", num_transformer, num_features),
                      ("cat", cat_transformer, cat_features)],
        remainder="drop")

    return Pipeline([("custom", CustomPreprocessor()),
                    ("column", col_transformer)])


def build_full_pipeline(preprocessor: Pipeline, model) -> Pipeline:
    """Wraps preprocessor + estimator into a single sklearn Pipeline."""
    return Pipeline([("preprocessor", preprocessor), ("model", model)])