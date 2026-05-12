import numpy as np
import pandas as pd
import pytest

from src.cost_analysis import (
    CostInputs,
    confusion_at_threshold,
    optimal_threshold,
    portfolio_pnl,
    sweep_thresholds,
)


def test_confusion_at_threshold_extremes():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6])
    conf = confusion_at_threshold(y_true, y_proba, 0.5)
    assert conf == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}

    conf_zero = confusion_at_threshold(y_true, y_proba, 0.0)
    assert conf_zero["tp"] + conf_zero["fp"] == 4

    conf_one = confusion_at_threshold(y_true, y_proba, 1.01)
    assert conf_one["tn"] + conf_one["fn"] == 4


def test_portfolio_pnl_arithmetic():
    conf = {"tp": 5, "fp": 3, "tn": 10, "fn": 2}
    costs = CostInputs(margin_per_tn=100, cost_per_fn=1000, cost_per_fp=50, cost_per_tp=0)
    pnl = portfolio_pnl(conf, costs)
    assert pnl["margin"] == 1000
    assert pnl["fn_loss"] == 2000
    assert pnl["fp_loss"] == 150
    assert pnl["net"] == 1000 - 2000 - 150


def test_sweep_thresholds_monotone_columns():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, size=200)
    y_proba = rng.uniform(0, 1, size=200)
    sweep = sweep_thresholds(y_true, y_proba, CostInputs(), n_steps=51)
    assert len(sweep) == 51
    # As threshold increases, the number of positives predicted must decrease
    assert (sweep["tp"].diff().dropna() <= 0).all()
    assert (sweep["fp"].diff().dropna() <= 0).all()


def test_optimal_threshold_in_unit_interval():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, size=200)
    y_proba = rng.uniform(0, 1, size=200)
    t = optimal_threshold(y_true, y_proba, CostInputs())
    assert 0.0 <= t <= 1.0
