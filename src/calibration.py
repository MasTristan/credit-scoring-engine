"""Calibration analysis and isotonic recalibration.

The XGBoost model is trained with `scale_pos_weight` to compensate for
class imbalance — a side-effect is that the raw predicted PDs are
**not** calibrated as long-run frequencies. This module:

- computes a reliability diagram (binned observed vs. predicted PD),
- decomposes the Brier score into reliability and resolution,
- exposes an isotonic recalibrator fitted on a validation set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationResult:
    """Reliability data + Brier decomposition."""

    bins: pd.DataFrame  # columns: bin_lower, bin_upper, mean_predicted, observed_rate, n
    brier: float
    reliability: float
    resolution: float
    uncertainty: float


def reliability_table(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Return a per-bin reliability table using equal-frequency quantiles.

    Each row contains the bin edges, the mean predicted PD inside the bin,
    the observed default rate inside the bin, and the bin size. Empty
    bins are dropped.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_proba, quantiles))
    if len(edges) < 2:
        edges = np.array([y_proba.min(), y_proba.max() + 1e-9])

    bin_idx = np.clip(np.digitize(y_proba, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_predicted": float(y_proba[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def brier_decomposition(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> tuple[float, float, float, float]:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    Lower reliability is better (perfect = 0). Higher resolution is better.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    n = len(y_true)

    table = reliability_table(y_true, y_proba, n_bins=n_bins)
    base_rate = y_true.mean()

    weights = table["n"] / n
    reliability = float((weights * (table["mean_predicted"] - table["observed_rate"]) ** 2).sum())
    resolution = float((weights * (table["observed_rate"] - base_rate) ** 2).sum())
    uncertainty = float(base_rate * (1 - base_rate))
    brier = float(((y_proba - y_true) ** 2).mean())
    return brier, reliability, resolution, uncertainty


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error over equal-frequency bins.

    Weighted mean absolute gap between predicted PD and observed default rate.
    0 = perfectly calibrated. This is the headline calibration metric reported
    in the model card.
    """
    table = reliability_table(y_true, y_proba, n_bins=n_bins)
    n = len(np.asarray(y_true))
    weights = table["n"] / n
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def assess_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> CalibrationResult:
    bins = reliability_table(y_true, y_proba, n_bins=n_bins)
    brier, reliability, resolution, uncertainty = brier_decomposition(
        y_true, y_proba, n_bins=n_bins
    )
    return CalibrationResult(
        bins=bins,
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
    )


def fit_isotonic(
    y_true: np.ndarray, y_proba: np.ndarray
) -> IsotonicRegression:
    """Fit an isotonic regression mapping raw PD → calibrated PD."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(y_proba), np.asarray(y_true).astype(float))
    return iso


def apply_calibrator(calibrator: IsotonicRegression, y_proba: np.ndarray) -> np.ndarray:
    return np.clip(calibrator.predict(np.asarray(y_proba)), 0.0, 1.0)
