"""Model zoo, preprocessing and evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def linear_preprocessor(cats=CATEGORICAL, nums=NUMERIC) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cats),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
        ]
    )


def tree_preprocessor(cats=CATEGORICAL, nums=NUMERIC) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
            ("num", SimpleImputer(strategy="median"), nums),
        ]
    )


class LogTargetModel:
    """Fit on log(premium), predict on the premium scale with Duan's smearing correction.

    Back-transforming a log-scale prediction with exp() alone predicts the median, which
    understates the mean premium. Smearing (Duan, 1983) rescales by mean(exp(residual)).
    """

    def __init__(self, estimator, name: str = ""):
        self.estimator = estimator
        self.name = name
        self.smear_ = 1.0

    def fit(self, X, y, **fit_kw):
        ly = np.log(y)
        self.estimator.fit(X, ly, **fit_kw)
        resid = ly - self.estimator.predict(X)
        self.smear_ = float(np.mean(np.exp(resid)))
        return self

    def predict_log(self, X):
        return self.estimator.predict(X)

    def predict(self, X):
        return np.exp(self.predict_log(X)) * self.smear_


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "R2": r2_score(y_true, y_pred),
        "R2_log": r2_score(np.log(y_true), np.log(y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MdAPE": float(np.median(ape)),
        "Within_20pct": float(np.mean(ape <= 0.20)),
    }


def segment_metrics(df: pd.DataFrame, y_pred, by: str = "coverage") -> pd.DataFrame:
    rows = []
    for seg, idx in df.groupby(by, observed=True).indices.items():
        m = metrics(df["premium"].values[idx], np.asarray(y_pred)[idx])
        m[by] = seg
        m["n"] = len(idx)
        rows.append(m)
    return pd.DataFrame(rows)
