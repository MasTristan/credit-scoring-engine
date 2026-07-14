"""Inference helpers for the UCI credit-card scoring model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_PATH = Path("models/xgboost_model.json")
FEATURE_NAMES_PATH = Path("models/feature_names.json")


def load_model(path: str | Path = MODEL_PATH) -> xgb.XGBClassifier:
    """Load the serialized XGBoost classifier."""
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def load_feature_names(path: str | Path = FEATURE_NAMES_PATH) -> list[str]:
    with open(path) as f:
        return json.load(f)


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Add missing columns (as 0) and order columns as the model expects.

    Does not mutate the caller's DataFrame.
    """
    X = X.copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    return X[feature_names]


def predict_proba(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return probability of default (class 1) for each row."""
    return model.predict_proba(X)[:, 1]


# ---- Rating / risk-band mapping (Project 1 PD table) ---------------------

_RATING_BUCKETS = [
    (0.001, "AAA/AA"),
    (0.003, "A"),
    (0.010, "BBB"),
    (0.030, "BB"),
    (0.080, "B"),
    (0.200, "CCC"),
]


def score_to_rating(pd_value: float) -> str:
    """Map a PD value to an internal rating bucket."""
    for threshold, label in _RATING_BUCKETS:
        if pd_value < threshold:
            return label
    return "D"


def score_to_risk_band(pd_value: float) -> str:
    """Map a PD value to a coarse risk band for UI colour-coding."""
    if pd_value < 0.05:
        return "LOW"
    if pd_value < 0.15:
        return "MEDIUM"
    return "HIGH"
