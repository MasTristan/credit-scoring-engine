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
    cost_per_review: float = 50.0  # € manual-underwriting cost per grey-zone case


def breakeven_pd(costs: CostInputs) -> float:
    """PD at which accepting and declining a contract have equal expected value.

    Accept EV = (1-PD)·margin - PD·cost_fn ; decline EV = -(1-PD)·cost_fp.
    Below this PD accepting is profitable, above it declining is. This is the
    cost-driven counterpart of the risk-target band cut-offs.
    """
    denom = costs.margin_per_tn + costs.cost_per_fp + costs.cost_per_fn
    if denom <= 0:
        return 0.0
    return (costs.margin_per_tn + costs.cost_per_fp) / denom


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


def policy_pnl(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    approve_below: float,
    decline_at_or_above: float,
    costs: CostInputs,
    review_effectiveness: float = 0.5,
) -> dict[str, float]:
    """Euro P&L of the three-tier decision policy.

    - Auto-approve band: accepted → performing goods earn ``margin_per_tn``,
      defaulters cost ``cost_per_fn``.
    - Auto-decline band: rejected → declined goods cost ``cost_per_fp`` (foregone
      margin); declined defaulters are avoided at no cost.
    - Manual-review band: each case costs ``cost_per_review``; a reviewer of
      effectiveness ``e`` in [0, 1] catches a fraction ``e`` of the defaulters
      (e = 0 rubber-stamps and approves all, e = 1 is a perfect underwriter),
      while performing goods are approved and earn margin either way.

    ``review_effectiveness`` makes the value of the human layer explicit: the
    review band only pays off when caught FN losses exceed the handling cost.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    e = float(np.clip(review_effectiveness, 0.0, 1.0))

    appr = p < approve_below
    dec = p >= decline_at_or_above
    rev = ~appr & ~dec

    appr_goods = int(((y == 0) & appr).sum())
    appr_bads = int(((y == 1) & appr).sum())
    dec_goods = int(((y == 0) & dec).sum())
    rev_goods = int(((y == 0) & rev).sum())
    rev_bads = int(((y == 1) & rev).sum())
    n_review = int(rev.sum())

    approve_pnl = appr_goods * costs.margin_per_tn - appr_bads * costs.cost_per_fn
    decline_pnl = -dec_goods * costs.cost_per_fp
    review_pnl = (
        rev_goods * costs.margin_per_tn
        - (1.0 - e) * rev_bads * costs.cost_per_fn
        - n_review * costs.cost_per_review
    )
    return {
        "approve_pnl": float(approve_pnl),
        "decline_pnl": float(decline_pnl),
        "review_pnl": float(review_pnl),
        "review_cost": float(n_review * costs.cost_per_review),
        "n_review": n_review,
        "net": float(approve_pnl + decline_pnl + review_pnl),
    }
