import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    Domain-specific transformations before sklearn ColumnTransformer.
    - pdays == -1 means 'never contacted' → replace with 0
    - balance: clip negative values to 0
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "pdays" in X.columns:
            X["pdays"] = X["pdays"].replace(-1, 0)
        if "balance" in X.columns:
            X["balance"] = X["balance"].clip(lower=0)
        return X