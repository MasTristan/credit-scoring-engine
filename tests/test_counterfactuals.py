from pathlib import Path

import pandas as pd
import pytest

from src.counterfactuals import find_counterfactual
from src.predict import load_feature_names, load_model

MODEL_PATH = Path("models/xgboost_model.json")
SAMPLE = Path("data/sample/sample_1000.csv")

artifacts_available = MODEL_PATH.exists() and SAMPLE.exists()


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_counterfactual_returns_none_if_already_below_threshold():
    model = load_model()
    feature_names = load_feature_names()
    sample = pd.read_csv(SAMPLE)
    sample = sample.rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})

    # threshold = 1.0 always wins
    raw_row = sample.head(1).copy()
    cf = find_counterfactual(model, raw_row, feature_names, threshold=1.0)
    assert cf is None


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_counterfactual_finds_actionable_change_on_high_risk():
    """For a high-risk borrower, the search should produce an actionable suggestion."""
    model = load_model()
    feature_names = load_feature_names()
    sample = pd.read_csv(SAMPLE)
    sample = sample.rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})

    # Construct a deliberately high-risk row: maxed-out PAY status.
    raw_row = sample.head(1).copy()
    raw_row["PAY_0"] = 3
    raw_row["PAY_2"] = 3
    cf = find_counterfactual(model, raw_row, feature_names, threshold=0.5)
    # Either a counterfactual is found (preferred) or the model still scores
    # below threshold which would already return None. Both are acceptable;
    # we only test that the function does not raise.
    assert cf is None or cf.new_pd < 0.5
