import numpy as np
import pandas as pd
import pytest

from src.fairness import disparate_impact, fairness_summary, per_group_metrics


@pytest.fixture
def synthetic_pop():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "SEX_MALE": rng.integers(0, 2, size=n),
            "AGE": rng.integers(21, 60, size=n),
            "EDUCATION": rng.integers(1, 5, size=n),
        }
    )
    y_true = rng.binomial(1, 0.22, size=n)
    y_proba = rng.uniform(0, 1, size=n)
    return df, y_true, y_proba


def test_per_group_metrics_attributes(synthetic_pop):
    df, y_true, y_proba = synthetic_pop
    rows = per_group_metrics(df, y_true, y_proba, threshold=0.5)
    assert set(rows["attribute"]) == {"SEX", "AGE", "EDUCATION"}
    assert (rows["n"] > 0).all()


def test_disparate_impact_reference_row(synthetic_pop):
    df, y_true, y_proba = synthetic_pop
    rows = per_group_metrics(df, y_true, y_proba, threshold=0.5)
    di = disparate_impact(rows, "SEX")
    assert "di_ratio" in di.columns
    assert di["reference"].sum() == 1
    ref = di[di["reference"]].iloc[0]
    assert ref["di_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_fairness_summary_returns_one_df_per_attribute(synthetic_pop):
    df, y_true, y_proba = synthetic_pop
    summary = fairness_summary(df, y_true, y_proba, threshold=0.5)
    assert set(summary.keys()) == {"SEX", "AGE", "EDUCATION"}
    for attr, sub in summary.items():
        assert "di_ratio" in sub.columns
        assert "eod" in sub.columns
