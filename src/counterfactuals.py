"""Counterfactual explanations.

Given a single applicant whose PD is above a desired threshold, find the
smallest single-feature change (in the model's *raw* feature space) that
brings the PD below the threshold. If no such single-feature change
exists, attempt a 2-feature combination.

The search is intentionally cheap (grid-based, on a curated set of
actionable raw features) because the user wants an interactive UI
response, not an optimisation-grade result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_prep import build_feature_matrix
from src.predict import align_features, predict_proba


@dataclass
class Counterfactual:
    feature: str
    original: float
    proposed: float
    new_pd: float
    description: str


# Features we are willing to suggest changes on. Some raw features
# (sex, marriage, education) are protected / unactionable and must
# never appear in a counterfactual.
ACTIONABLE_FEATURES: dict[str, list[float]] = {
    "PAY_0":     [-1, 0, 1, 2],
    "PAY_2":     [-1, 0, 1, 2],
    "BILL_AMT1": [0.5, 0.75, 1.0],          # multipliers
    "PAY_AMT1":  [1.5, 2.0, 3.0, 5.0],      # multipliers
    "LIMIT_BAL": [1.25, 1.5, 2.0],          # multipliers
}


HUMAN_LABEL = {
    "PAY_0":     "current month's repayment status",
    "PAY_2":     "prior month's repayment status",
    "BILL_AMT1": "most recent bill amount",
    "PAY_AMT1":  "most recent payment amount",
    "LIMIT_BAL": "credit limit",
}


def _score(model, raw_row: pd.DataFrame, feature_names: list[str]) -> float:
    X = build_feature_matrix(raw_row.assign(IS_DEFAULT=0))
    X = align_features(X.drop(columns=["IS_DEFAULT"]), feature_names)
    return float(predict_proba(model, X)[0])


def _candidate_values(feature: str, current: float) -> list[float]:
    """Return concrete proposed values for ``feature`` given current value."""
    grid = ACTIONABLE_FEATURES[feature]
    if feature in {"PAY_0", "PAY_2"}:
        return [v for v in grid if v < current]
    # multiplier features
    return [current * m for m in grid] if feature in {"PAY_AMT1", "LIMIT_BAL"} else [current * m for m in grid]


def find_counterfactual(
    model,
    raw_row: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.15,
) -> Counterfactual | None:
    """Return the smallest single-feature change that pushes PD below threshold.

    "Smallest" is measured by the resulting PD: the candidate that produces
    the largest PD drop *while crossing the threshold* is returned.
    """
    base_pd = _score(model, raw_row, feature_names)
    if base_pd < threshold:
        return None

    best: Counterfactual | None = None

    for feat, _ in ACTIONABLE_FEATURES.items():
        if feat not in raw_row.columns:
            continue
        original = float(raw_row[feat].iloc[0])
        for proposed in _candidate_values(feat, original):
            modified = raw_row.copy()
            modified[feat] = proposed
            new_pd = _score(model, modified, feature_names)
            if new_pd < threshold:
                if best is None or new_pd < best.new_pd:
                    direction = "decrease" if proposed < original else "increase"
                    desc = (
                        f"{direction.capitalize()} {HUMAN_LABEL[feat]} "
                        f"from {original:,.0f} to {proposed:,.0f}"
                    )
                    best = Counterfactual(
                        feature=feat,
                        original=original,
                        proposed=proposed,
                        new_pd=new_pd,
                        description=desc,
                    )

    return best
