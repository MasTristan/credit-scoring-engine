import numpy as np
import pandas as pd

from src.data_prep import (
    build_feature_matrix,
    clean_categoricals,
    engineer_features,
    split_train_test,
)


def _toy_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 500_000, n),
            "SEX": rng.choice([1, 2], n),
            "EDUCATION": rng.choice([1, 2, 3, 4, 5, 6, 0], n),
            "MARRIAGE": rng.choice([0, 1, 2, 3], n),
            "AGE": rng.integers(22, 70, n),
        }
    )
    for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        df[c] = rng.integers(-2, 5, n)
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.integers(-1000, 200_000, n)
        df[f"PAY_AMT{i}"] = rng.integers(0, 50_000, n)
    df["IS_DEFAULT"] = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    return df


def test_categorical_cleaning_collapses_unknown_codes():
    raw = _toy_raw()
    cleaned = clean_categoricals(raw)
    assert set(cleaned["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(cleaned["MARRIAGE"].unique()).issubset({1, 2, 3})
    assert set(cleaned["SEX"].unique()).issubset({1, 2})


def test_engineered_features_exist_and_have_no_nulls():
    raw = clean_categoricals(_toy_raw())
    feat = engineer_features(raw)
    for col in [
        "PAY_MEAN",
        "PAY_MAX",
        "DELINQ_COUNT",
        "BILL_MEAN",
        "PAY_AMT_MEAN",
        "UTILIZATION",
        "PAYMENT_RATIO",
        "LIMIT_PER_AGE",
    ]:
        assert col in feat.columns
        assert not feat[col].isnull().any(), f"{col} contains nulls"


def test_no_post_target_leakage_features_in_final_matrix():
    # The target is the only column starting with IS_; nothing post-origination
    # exists in this dataset (no recoveries, payments_received, etc.).
    feat = build_feature_matrix(_toy_raw())
    leakage_candidates = {"PAY_AMT_FUTURE", "BILL_AMT_FUTURE", "RECOVERIES"}
    assert leakage_candidates.isdisjoint(set(feat.columns))


def test_target_distribution_within_expected_range():
    raw = _toy_raw(n=2000)
    rate = raw["IS_DEFAULT"].mean()
    assert 0.10 < rate < 0.35


def test_no_null_after_full_prep():
    feat = build_feature_matrix(_toy_raw())
    assert feat.isnull().sum().sum() == 0


def test_train_test_stratified_split_preserves_default_rate():
    feat = build_feature_matrix(_toy_raw(n=1000))
    train, test = split_train_test(feat, test_size=0.2)
    assert len(train) + len(test) == len(feat)
    # stratified: rates should match within 2 pp
    assert abs(train["IS_DEFAULT"].mean() - test["IS_DEFAULT"].mean()) < 0.02
