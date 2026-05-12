"""SHAP explainability helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def get_explainer(model, background: pd.DataFrame | None = None) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for XGBoost.

    Uses `feature_perturbation="tree_path_dependent"` (recommended for
    XGBoost): it walks the tree paths using the node-cover frequencies as
    weights and guarantees the local-accuracy property
    `sum(shap_values) + base_value == raw_model_output`.

    `background` is unused under this mode but kept in the signature for
    symmetry with the interventional path.
    """
    return shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )


def compute_shap_values(
    explainer: shap.TreeExplainer, X: pd.DataFrame
) -> np.ndarray:
    """Return a (n_samples, n_features) SHAP-value matrix on the log-odds scale."""
    values = explainer.shap_values(X)
    if isinstance(values, list):  # legacy multiclass path
        values = values[1]
    return np.asarray(values)


def get_waterfall_data(
    shap_values: np.ndarray,
    X_row: pd.Series,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Return a tidy DataFrame for waterfall display, sorted by |SHAP|."""
    shap_row = np.asarray(shap_values).reshape(-1)
    df = pd.DataFrame(
        {
            "FEATURE": feature_names,
            "VALUE": [X_row.get(name, np.nan) for name in feature_names],
            "SHAP_VALUE": shap_row,
        }
    )
    df["ABS_SHAP"] = df["SHAP_VALUE"].abs()
    df["DIRECTION"] = np.where(df["SHAP_VALUE"] > 0, "increase", "decrease")
    df["FEATURE_LABEL"] = df.apply(
        lambda r: f"{r['FEATURE']} = {_format_value(r['VALUE'])}", axis=1
    )
    df = df.sort_values("ABS_SHAP", ascending=False).head(top_n).reset_index(drop=True)
    return df.drop(columns=["ABS_SHAP"])


def get_global_importance(
    shap_values_matrix: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Return mean(|SHAP|) per feature, sorted descending."""
    abs_mean = np.abs(np.asarray(shap_values_matrix)).mean(axis=0)
    return (
        pd.DataFrame({"FEATURE": feature_names, "MEAN_ABS_SHAP": abs_mean})
        .sort_values("MEAN_ABS_SHAP", ascending=False)
        .reset_index(drop=True)
    )


def _format_value(v):
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3f}"
    return str(v)
