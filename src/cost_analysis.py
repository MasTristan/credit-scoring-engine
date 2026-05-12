"""Cost-sensitive evaluation.

A credit decision is FN-asymmetric: a false negative costs the lender
the unpaid exposure × LGD, a false positive costs the lender the
foregone margin on a loan that would have performed. This module
turns the confusion matrix into a euro-denominated P&L and finds the
threshold that maximises it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostInputs:
    margin_per_tn: float = 120.0   # € margin on a correctly accepted, performing contract
    cost_per_fn: float = 3000.0    # € loss on an accepted contract that defaults
    cost_per_fp: float = 120.0     # € foregone margin on a rejected good
    cost_per_tp: float = 0.0       # € — no incremental cost on a correctly rejected bad


def confusion_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, int]:
    """Return TP / FP / TN / FN counts at the given threshold.

    Convention: positive class (1) = "predicted to default" (i.e. rejected).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def portfolio_pnl(
    confusion: dict[str, int], costs: CostInputs
) -> dict[str, float]:
    """Return the euro P&L breakdown for a confusion matrix."""
    margin = confusion["tn"] * costs.margin_per_tn
    fn_loss = confusion["fn"] * costs.cost_per_fn
    fp_loss = confusion["fp"] * costs.cost_per_fp
    tp_cost = confusion["tp"] * costs.cost_per_tp
    net = margin - fn_loss - fp_loss - tp_cost
    return {
        "margin": margin,
        "fn_loss": fn_loss,
        "fp_loss": fp_loss,
        "tp_cost": tp_cost,
        "net": net,
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    costs: CostInputs,
    n_steps: int = 101,
) -> pd.DataFrame:
    """Compute the cost-sensitive P&L curve across thresholds in [0, 1]."""
    rows = []
    thresholds = np.linspace(0.0, 1.0, n_steps)
    for t in thresholds:
        conf = confusion_at_threshold(y_true, y_proba, t)
        pnl = portfolio_pnl(conf, costs)
        rows.append({
            "threshold": float(t),
            **conf,
            **pnl,
        })
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, costs: CostInputs, n_steps: int = 101
) -> float:
    """Threshold maximising portfolio net P&L."""
    sweep = sweep_thresholds(y_true, y_proba, costs, n_steps=n_steps)
    return float(sweep.loc[sweep["net"].idxmax(), "threshold"])
