from pathlib import Path

import pandas as pd
import pytest

from src.data_prep import build_feature_matrix
from src.predict import (
    align_features,
    load_feature_names,
    load_model,
    predict_proba,
    score_to_rating,
    score_to_risk_band,
)

MODEL_PATH = Path("models/xgboost_model.json")
FEATURE_NAMES = Path("models/feature_names.json")
SAMPLE = Path("data/sample/sample_1000.csv")

artifacts_available = MODEL_PATH.exists() and FEATURE_NAMES.exists()


def test_score_to_rating_extremes():
    assert score_to_rating(0.0) == "AAA/AA"
    assert score_to_rating(0.0005) == "AAA/AA"
    assert score_to_rating(0.5) == "D"
    assert score_to_rating(0.99) == "D"


def test_score_to_rating_buckets_increase_monotonically():
    pds = [0.0005, 0.002, 0.005, 0.02, 0.05, 0.1, 0.3]
    ratings = [score_to_rating(p) for p in pds]
    assert ratings == ["AAA/AA", "A", "BBB", "BB", "B", "CCC", "D"]


def test_score_to_risk_band():
    assert score_to_risk_band(0.0) == "LOW"
    assert score_to_risk_band(0.049) == "LOW"
    assert score_to_risk_band(0.05) == "MEDIUM"
    assert score_to_risk_band(0.149) == "MEDIUM"
    assert score_to_risk_band(0.15) == "HIGH"
    assert score_to_risk_band(0.99) == "HIGH"


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_predict_returns_probabilities_between_zero_and_one():
    model = load_model()
    feature_names = load_feature_names()
    sample = pd.read_csv(SAMPLE).head(10).rename(
        columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"}
    )
    X = build_feature_matrix(sample).drop(columns=["IS_DEFAULT"])
    X = align_features(X, feature_names)
    pd_values = predict_proba(model, X)
    assert pd_values.shape == (10,)
    assert ((pd_values >= 0) & (pd_values <= 1)).all()


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_align_features_orders_and_fills_missing_columns():
    feature_names = load_feature_names()
    X = pd.DataFrame({feature_names[0]: [1.0]})
    aligned = align_features(X, feature_names)
    assert list(aligned.columns) == feature_names
    assert (aligned.iloc[0, 1:] == 0).all()
