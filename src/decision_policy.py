"""Three-tier decision policy on the calibrated PD.

A single hard threshold forces a binary accept/reject on every applicant,
including the large middle band where the model is genuinely uncertain, which
is what produces an "unconvincing" false-positive count. Real origination
splits the decision into three tiers:

    APPROVE  : PD < approve_below            (confidently good, auto-accept)
    REVIEW   : approve_below <= PD < decline (grey zone, manual underwriting)
    DECLINE  : PD >= decline_at_or_above     (confidently bad, auto-reject)

The tails are auto-decided with a clear risk gap; the ambiguous middle goes to
a human. Thresholds are fit on the validation split from interpretable risk
targets, never on test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

APPROVE = "APPROVE"
REVIEW = "REVIEW"
DECLINE = "DECLINE"

POLICY_PATH = Path("models/decision_policy.json")


@dataclass(frozen=True)
class ThreeTierPolicy:
    """Two PD cut-offs defining the approve / review / decline bands."""

    approve_below: float
    decline_at_or_above: float

    def __post_init__(self) -> None:
        if not 0.0 < self.approve_below < self.decline_at_or_above < 1.0:
            raise ValueError(
                "Require 0 < approve_below < decline_at_or_above < 1, got "
                f"{self.approve_below} and {self.decline_at_or_above}."
            )

    def decide(self, pd_value: float) -> str:
        if pd_value < self.approve_below:
            return APPROVE
        if pd_value >= self.decline_at_or_above:
            return DECLINE
        return REVIEW

    def decide_batch(self, pd_values: np.ndarray) -> np.ndarray:
        p = np.asarray(pd_values, dtype=float)
        out = np.full(p.shape, REVIEW, dtype=object)
        out[p < self.approve_below] = APPROVE
        out[p >= self.decline_at_or_above] = DECLINE
        return out


def fit_policy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    max_approved_bad_rate: float = 0.08,
    min_declined_bad_rate: float = 0.60,
    min_band: int = 50,
) -> ThreeTierPolicy:
    """Derive the two cut-offs on a held-out (validation) set.

    ``approve_below`` is the highest PD for which the auto-approved population
    still defaults at or below ``max_approved_bad_rate``; ``decline_at_or_above``
    is the lowest PD for which the auto-declined population defaults at or above
    ``min_declined_bad_rate``. Both bands must contain at least ``min_band``
    samples to avoid fitting on noise.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    grid = np.round(np.arange(0.01, 1.0, 0.005), 3)

    approve_below = grid[0]
    for t in grid:
        mask = p < t
        if mask.sum() >= min_band and y[mask].mean() <= max_approved_bad_rate:
            approve_below = float(t)

    decline_at = grid[-1]
    for t in grid[::-1]:
        mask = p >= t
        if mask.sum() >= min_band and y[mask].mean() >= min_declined_bad_rate:
            decline_at = float(t)

    if approve_below >= decline_at:  # pragma: no cover - safety for degenerate data
        approve_below, decline_at = 0.10, 0.40
    return ThreeTierPolicy(approve_below=approve_below, decline_at_or_above=decline_at)


def evaluate_policy(
    y_true: np.ndarray, y_proba: np.ndarray, policy: ThreeTierPolicy
) -> pd.DataFrame:
    """Per-band population share and realised default rate."""
    y = np.asarray(y_true).astype(int)
    decisions = policy.decide_batch(y_proba)
    n = len(y)
    rows = []
    for band in (APPROVE, REVIEW, DECLINE):
        mask = decisions == band
        count = int(mask.sum())
        rows.append(
            {
                "decision": band,
                "n": count,
                "share": count / n if n else 0.0,
                "default_rate": float(y[mask].mean()) if count else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def save_policy(policy: ThreeTierPolicy, path: str | Path = POLICY_PATH) -> None:
    with open(path, "w") as f:
        json.dump(
            {
                "approve_below": policy.approve_below,
                "decline_at_or_above": policy.decline_at_or_above,
            },
            f,
            indent=2,
        )


def load_policy(path: str | Path = POLICY_PATH) -> ThreeTierPolicy:
    with open(path) as f:
        d = json.load(f)
    return ThreeTierPolicy(
        approve_below=d["approve_below"],
        decline_at_or_above=d["decline_at_or_above"],
    )
