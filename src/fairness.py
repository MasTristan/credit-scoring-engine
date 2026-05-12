"""Fairness audit on protected attributes.

Computes disparate-impact ratio, equal-opportunity difference and
AUC per sub-group. Sub-groups are defined by:

- ``SEX_MALE`` (0 / 1)
- ``AGE`` bands: ``<30``, ``30-45``, ``>45``
- ``EDUCATION`` (raw integer code 1..4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class FairnessRow:
    attribute: str
    group: str
    n: int
    base_rate: float
    selection_rate: float
    tpr: float
    fpr: float
    auc: float


SEX_GROUPS = {"Male": 1, "Female": 0}


def _age_band(age: int | float) -> str:
    if age < 30:
        return "<30"
    if age <= 45:
        return "30-45"
    return ">45"


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def _row_for_mask(
    mask: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    attribute: str,
    group: str,
) -> FairnessRow:
    n = int(mask.sum())
    if n == 0:
        return FairnessRow(attribute, group, 0, np.nan, np.nan, np.nan, np.nan, np.nan)
    y_true_g = y_true[mask]
    y_pred_g = y_pred[mask]
    y_proba_g = y_proba[mask]

    base = float(y_true_g.mean())
    selection = float(y_pred_g.mean())
    pos = y_true_g == 1
    neg = y_true_g == 0
    tpr = float(y_pred_g[pos].mean()) if pos.any() else float("nan")
    fpr = float(y_pred_g[neg].mean()) if neg.any() else float("nan")
    auc = _safe_auc(y_true_g, y_proba_g)
    return FairnessRow(attribute, group, n, base, selection, tpr, fpr, auc)


def per_group_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Return one row per (attribute, group) with the audit metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    rows: list[FairnessRow] = []

    if "SEX_MALE" in df.columns:
        sex_col = df["SEX_MALE"].to_numpy()
        for label, val in SEX_GROUPS.items():
            mask = sex_col == val
            rows.append(_row_for_mask(mask, y_true, y_pred, y_proba, "SEX", label))

    if "AGE" in df.columns:
        ages = df["AGE"].to_numpy()
        bands = np.array([_age_band(a) for a in ages])
        for label in ["<30", "30-45", ">45"]:
            mask = bands == label
            rows.append(_row_for_mask(mask, y_true, y_pred, y_proba, "AGE", label))

    if "EDUCATION" in df.columns:
        edu = df["EDUCATION"].to_numpy()
        labels = {1: "Graduate", 2: "University", 3: "High school", 4: "Other"}
        for code, label in labels.items():
            mask = edu == code
            rows.append(_row_for_mask(mask, y_true, y_pred, y_proba, "EDUCATION", label))

    return pd.DataFrame([r.__dict__ for r in rows])


def disparate_impact(groups: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """For each group on ``attribute``, return DI ratio vs. the largest group."""
    sub = groups[groups["attribute"] == attribute].copy()
    if sub.empty:
        return sub
    ref_idx = sub["n"].idxmax()
    ref_rate = sub.loc[ref_idx, "selection_rate"]
    sub["di_ratio"] = sub["selection_rate"] / ref_rate if ref_rate else np.nan
    sub["eod"] = sub["tpr"] - sub.loc[ref_idx, "tpr"]
    sub["reference"] = sub.index == ref_idx
    return sub.reset_index(drop=True)


def fairness_summary(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    """Return one dataframe per audited attribute, each with DI/EOD added."""
    groups = per_group_metrics(df, y_true, y_proba, threshold)
    out: dict[str, pd.DataFrame] = {}
    for attr in groups["attribute"].unique():
        out[attr] = disparate_impact(groups, attr)
    return out
