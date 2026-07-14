from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    apply_calibrator,
    assess_calibration,
    brier_decomposition,
    expected_calibration_error,
    fit_isotonic,
    reliability_table,
)
from src.data_prep import build_feature_matrix
from src.predict import align_features, load_feature_names, load_model, predict_proba

MODEL_PATH = Path("models/xgboost_model.json")
SAMPLE = Path("data/sample/sample_1000.csv")
artifacts_available = MODEL_PATH.exists() and SAMPLE.exists()


@pytest.fixture
def synthetic_probs():
    rng = np.random.default_rng(0)
    n = 2000
    # 22% base rate, mildly miscalibrated raw scores
    y_true = rng.binomial(1, 0.22, size=n)
    raw_logit = -1.3 + 1.5 * (y_true - 0.22) + rng.normal(0, 0.6, size=n)
    y_proba = 1 / (1 + np.exp(-raw_logit))
    # introduce miscalibration: scale toward extremes
    y_proba = np.clip(y_proba ** 0.7, 1e-6, 1 - 1e-6)
    return y_true, y_proba


def test_reliability_table_columns(synthetic_probs):
    y_true, y_proba = synthetic_probs
    table = reliability_table(y_true, y_proba, n_bins=10)
    assert set(["bin_lower", "bin_upper", "mean_predicted", "observed_rate", "n"]).issubset(table.columns)
    assert table["n"].sum() == len(y_true)
    assert table["mean_predicted"].between(0, 1).all()
    assert table["observed_rate"].between(0, 1).all()


def test_brier_decomposition_identity(synthetic_probs):
    y_true, y_proba = synthetic_probs
    brier, reliab, resol, uncert = brier_decomposition(y_true, y_proba, n_bins=10)
    # Brier = reliability - resolution + uncertainty (up to bin-mean approximation)
    assert abs(brier - (reliab - resol + uncert)) < 0.02


def test_assess_calibration_returns_struct(synthetic_probs):
    y_true, y_proba = synthetic_probs
    result = assess_calibration(y_true, y_proba)
    assert result.brier > 0
    assert result.reliability >= 0
    assert result.resolution >= 0
    assert result.uncertainty > 0
    assert isinstance(result.bins, pd.DataFrame)


def test_isotonic_improves_brier(synthetic_probs):
    y_true, y_proba = synthetic_probs
    iso = fit_isotonic(y_true, y_proba)
    y_recal = apply_calibrator(iso, y_proba)
    brier_raw = ((y_proba - y_true) ** 2).mean()
    brier_recal = ((y_recal - y_true) ** 2).mean()
    # Isotonic regression is the in-sample Brier minimiser among monotone maps,
    # so the recalibrated Brier should be no worse and usually strictly better.
    assert brier_recal <= brier_raw + 1e-9


def test_ece_zero_when_perfectly_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=20000)
    y = rng.binomial(1, p)  # outcomes drawn at the predicted probability
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_ece_large_when_miscalibrated():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, size=20000)
    y = rng.binomial(1, np.clip(p / 2, 0, 1))  # true rate is half the prediction
    assert expected_calibration_error(y, p, n_bins=10) > 0.1


@pytest.mark.skipif(not artifacts_available, reason="trained model not available")
def test_committed_model_is_calibrated():
    """Guardrail: the shipped model must output PDs calibrated to the base rate.

    This locks in the removal of scale_pos_weight; reintroducing it would push
    the mean predicted PD to ~2x the base rate and blow past these bounds.
    """
    model = load_model()
    feature_names = load_feature_names()
    sample = pd.read_csv(SAMPLE).rename(columns={"IS_DEFAULT_TRUE": "IS_DEFAULT"})
    y = sample["IS_DEFAULT"].astype(int).values
    X = align_features(build_feature_matrix(sample).drop(columns=["IS_DEFAULT"]), feature_names)
    p = predict_proba(model, X)
    assert abs(p.mean() - y.mean()) < 0.05, "mean PD should track the base rate"
    assert expected_calibration_error(y, p, n_bins=10) < 0.05
