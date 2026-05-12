from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_prep import build_feature_matrix
from src.explain import (
    compute_shap_values,
    get_explainer,
    get_global_importance,
    get_waterfall_data,
)
from src.predict import align_features, load_feature_names, load_model

MODEL_PATH = Path("models/xgboost_model.json")
BACKGROUND = Path("models/shap_background.parquet")
SAMPLE = Path("data/sample/sample_1000.csv")

artifacts_available = MODEL_PATH.exists() and BACKGROUND.exists() and SAMPLE.exists()


@pytest.fixture(scope="module")
def loaded():
    model = load_model()
    background = pd.read_parquet(BACKGROUND)
    explainer = get_explainer(model, background)
    feature_names = load_feature_names()
    return model, explainer, feature_names


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_shap_values_have_correct_shape(loaded):
    model, explainer, feature_names = loaded
    sample = pd.read_csv(SAMPLE).head(5).rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
    X = build_feature_matrix(sample).drop(columns=["IS_DEFAULT"])
    X = align_features(X, feature_names)
    sv = compute_shap_values(explainer, X)
    assert sv.shape == (5, len(feature_names))


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_shap_local_accuracy(loaded):
    """SHAP local accuracy: sum(shap_values) + base_value == logit(P(default)).

    With `feature_perturbation="tree_path_dependent"` the explainer attributes
    the model's predicted log-odds exactly. We therefore compare against the
    logit of `predict_proba` (which is what the app shows to the user).
    """
    model, explainer, feature_names = loaded
    sample = pd.read_csv(SAMPLE).head(5).rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
    X = build_feature_matrix(sample).drop(columns=["IS_DEFAULT"])
    X = align_features(X, feature_names)

    sv = compute_shap_values(explainer, X)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).flatten()[-1]
    reconstructed = sv.sum(axis=1) + expected_value

    proba = model.predict_proba(X)[:, 1]
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    logit = np.log(proba / (1 - proba))

    assert np.allclose(reconstructed, logit, atol=1e-3)


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_waterfall_returns_top_n(loaded):
    _, explainer, feature_names = loaded
    sample = pd.read_csv(SAMPLE).head(1).rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
    X = build_feature_matrix(sample).drop(columns=["IS_DEFAULT"])
    X = align_features(X, feature_names)
    sv = compute_shap_values(explainer, X)
    wf = get_waterfall_data(sv[0], X.iloc[0], feature_names, top_n=10)
    assert len(wf) == 10
    assert list(wf.columns) >= ["FEATURE", "VALUE", "SHAP_VALUE", "DIRECTION"]


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_global_importance_sorted_descending(loaded):
    _, explainer, feature_names = loaded
    sample = pd.read_csv(SAMPLE).head(20).rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
    X = build_feature_matrix(sample).drop(columns=["IS_DEFAULT"])
    X = align_features(X, feature_names)
    sv = compute_shap_values(explainer, X)
    gi = get_global_importance(sv, feature_names)
    assert gi["MEAN_ABS_SHAP"].is_monotonic_decreasing
    assert set(gi["FEATURE"]) == set(feature_names)
