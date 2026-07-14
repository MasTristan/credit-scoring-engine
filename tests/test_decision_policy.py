from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_prep import build_feature_matrix
from src.decision_policy import (
    APPROVE,
    DECLINE,
    REVIEW,
    ThreeTierPolicy,
    evaluate_policy,
    fit_policy,
    load_policy,
    save_policy,
)
from src.predict import align_features, load_feature_names, load_model, predict_proba

MODEL_PATH = Path("models/xgboost_model.json")
POLICY_PATH = Path("models/decision_policy.json")
SAMPLE = Path("data/sample/sample_1000.csv")
artifacts_available = MODEL_PATH.exists() and SAMPLE.exists()


def test_decide_maps_each_band():
    policy = ThreeTierPolicy(approve_below=0.10, decline_at_or_above=0.40)
    assert policy.decide(0.05) == APPROVE
    assert policy.decide(0.10) == REVIEW  # boundary: approve is strict <
    assert policy.decide(0.25) == REVIEW
    assert policy.decide(0.40) == DECLINE  # boundary: decline is >=
    assert policy.decide(0.9) == DECLINE


def test_decide_batch_matches_scalar():
    policy = ThreeTierPolicy(approve_below=0.10, decline_at_or_above=0.40)
    p = np.array([0.02, 0.10, 0.39, 0.40, 0.8])
    batch = list(policy.decide_batch(p))
    assert batch == [policy.decide(x) for x in p]


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        ThreeTierPolicy(approve_below=0.5, decline_at_or_above=0.3)  # inverted
    with pytest.raises(ValueError):
        ThreeTierPolicy(approve_below=0.0, decline_at_or_above=0.4)  # out of range


def test_fit_policy_orders_and_respects_targets():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0, 1, size=n)
    y = rng.binomial(1, p)  # calibrated: default rate tracks PD
    policy = fit_policy(y, p, max_approved_bad_rate=0.08, min_declined_bad_rate=0.60)
    assert 0 < policy.approve_below < policy.decline_at_or_above < 1
    # approved population defaults at or below target, declined at or above target
    assert y[p < policy.approve_below].mean() <= 0.08 + 1e-9
    assert y[p >= policy.decline_at_or_above].mean() >= 0.60 - 1e-9


def test_evaluate_policy_partitions_population():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=3000)
    y = rng.binomial(1, p)
    policy = ThreeTierPolicy(0.15, 0.40)
    table = evaluate_policy(y, p, policy)
    assert list(table["decision"]) == [APPROVE, REVIEW, DECLINE]
    assert table["n"].sum() == len(y)
    assert abs(table["share"].sum() - 1.0) < 1e-9


def test_save_load_roundtrip(tmp_path):
    policy = ThreeTierPolicy(0.14, 0.365)
    p = tmp_path / "policy.json"
    save_policy(policy, p)
    loaded = load_policy(p)
    assert loaded == policy


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_committed_policy_separates_risk():
    """Guardrail: approve band must be materially safer than the decline band."""
    policy = load_policy(POLICY_PATH)
    model = load_model()
    feature_names = load_feature_names()
    sample = pd.read_csv(SAMPLE).rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
    y = sample["IS_DEFAULT"].astype(int).values
    X = align_features(build_feature_matrix(sample).drop(columns=["IS_DEFAULT"]), feature_names)
    p = predict_proba(model, X)
    table = evaluate_policy(y, p, policy).set_index("decision")
    assert table.loc[APPROVE, "default_rate"] < 0.15
    assert table.loc[DECLINE, "default_rate"] > 0.45
    assert table.loc[APPROVE, "default_rate"] < table.loc[DECLINE, "default_rate"]
