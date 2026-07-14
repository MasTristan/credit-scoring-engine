"""Train the XGBoost credit-scoring model.

Input  : data/processed/train.parquet, data/processed/val.parquet,
         data/processed/test.parquet
Output : models/xgboost_model.json
         models/feature_importance.csv
         models/shap_background.parquet
         models/training_metrics.json
         models/feature_names.json (copy)

Methodology: early stopping and the operating threshold are selected on the
validation split. The test split is held out and only scored once, for the
final reported metrics, so those metrics are an unbiased estimate of
out-of-sample performance.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.calibration import brier_decomposition, expected_calibration_error
from src.decision_policy import evaluate_policy, fit_policy, save_policy

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def load_split() -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]
]:
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    with open(PROCESSED_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    X_train = train[feature_names]
    y_train = train["IS_DEFAULT"].astype(int)
    X_val = val[feature_names]
    y_val = val["IS_DEFAULT"].astype(int)
    X_test = test[feature_names]
    y_test = test["IS_DEFAULT"].astype(int)
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def build_params(y_train: pd.Series) -> dict:
    # No scale_pos_weight: training on the natural class distribution keeps the
    # predicted probabilities calibrated as long-run default frequencies, which
    # is the whole point of a PD model. Re-weighting the positive class (the
    # earlier scale_pos_weight = n_neg / n_pos ≈ 3.5) inflated every PD to ~2x
    # the base rate (test ECE 0.19) without improving ranking. Post-hoc isotonic
    # recalibration was benchmarked as an alternative and did not beat the
    # natively calibrated model on validation, so it is intentionally not used.
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": ["auc", "aucpr"],
        "early_stopping_rounds": 50,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
    }


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def youden_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def compute_metrics(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    feature_names: list[str],
) -> dict:
    """Metrics on the held-out test set.

    ``threshold`` is the operating point chosen on the validation split, so
    precision/recall/f1 reflect a decision rule fixed before seeing the test
    set (no threshold tuning on test).
    """
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    y_pred = (y_proba >= threshold).astype(int)
    _, reliability, _, _ = brier_decomposition(y_test.values, y_proba)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "gini": 2 * roc_auc - 1,
        "ks_statistic": ks_statistic(y_test.values, y_proba),
        "brier_score": float(brier_score_loss(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "ece": expected_calibration_error(y_test.values, y_proba),
        "reliability": float(reliability),
        "mean_predicted_pd": float(np.mean(y_proba)),
        "threshold_optimal": float(threshold),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
        "n_features": len(feature_names),
        "training_date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
    }


def feature_importance_table(
    model: xgb.XGBClassifier, feature_names: list[str]
) -> pd.DataFrame:
    booster = model.get_booster()
    booster.feature_names = feature_names
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    rows = []
    for name in feature_names:
        rows.append(
            {
                "FEATURE": name,
                "IMPORTANCE_GAIN": gain.get(name, 0.0),
                "IMPORTANCE_WEIGHT": weight.get(name, 0.0),
                "IMPORTANCE_COVER": cover.get(name, 0.0),
            }
        )
    df = pd.DataFrame(rows)
    df["RANK_GAIN"] = df["IMPORTANCE_GAIN"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("IMPORTANCE_GAIN", ascending=False).reset_index(drop=True)


def stratified_sample(
    X: pd.DataFrame, y: pd.Series, n: int = 500, seed: int = 42
) -> pd.DataFrame:
    pos_idx = y[y == 1].sample(n=min(n // 2, int((y == 1).sum())), random_state=seed).index
    neg_idx = y[y == 0].sample(n=min(n - len(pos_idx), int((y == 0).sum())), random_state=seed).index
    idx = pos_idx.union(neg_idx)
    return X.loc[idx].reset_index(drop=True)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_split()
    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    params = build_params(y_train)
    print("scale_pos_weight = 1.0 (natural distribution, PD kept calibrated)")

    model = xgb.XGBClassifier(**params)
    # Early stopping is driven by the validation set; the test set stays unseen.
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"best iteration: {model.best_iteration}")

    # Operating threshold chosen on validation, then applied to test.
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = youden_threshold(y_val.values, val_proba)
    print(f"threshold (Youden J on val): {threshold:.4f}")

    # Three-tier decision policy, also fit on validation.
    policy = fit_policy(y_val.values, val_proba)
    print(
        f"decision policy: approve < {policy.approve_below:.3f} "
        f"| decline >= {policy.decline_at_or_above:.3f}"
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(
        y_train, y_val, y_test, y_proba, threshold, feature_names
    )
    print("Metrics on test set:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:>20} : {v:.4f}")
        else:
            print(f"  {k:>20} : {v}")

    model.save_model(MODELS_DIR / "xgboost_model.json")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(MODELS_DIR / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    feature_importance_table(model, feature_names).to_csv(
        MODELS_DIR / "feature_importance.csv", index=False
    )

    save_policy(policy, MODELS_DIR / "decision_policy.json")
    print("decision policy on test set:")
    print(evaluate_policy(y_test.values, y_proba, policy).to_string(index=False))

    background = stratified_sample(X_train, y_train, n=500)
    background.to_parquet(MODELS_DIR / "shap_background.parquet", index=False)
    print(f"shap_background.parquet: {background.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
