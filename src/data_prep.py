"""Data preparation pipeline for the UCI credit-card-default scoring model.

Input  : data/raw/uci_credit_card.csv
Output : data/processed/train.parquet
         data/processed/val.parquet
         data/processed/test.parquet
         data/processed/feature_names.json
         data/sample/sample_1000.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/uci_credit_card.csv")
PROCESSED_DIR = Path("data/processed")
SAMPLE_DIR = Path("data/sample")

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_FEATURE_COLS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_COLS
    + BILL_COLS
    + PAY_AMT_COLS
)
TARGET_RAW_CANDIDATES = ["default.payment.next.month", "default payment next month"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    for cand in TARGET_RAW_CANDIDATES:
        if cand in df.columns:
            df = df.rename(columns={cand: "IS_DEFAULT"})
            break
    if "IS_DEFAULT" not in df.columns:
        raise ValueError(
            f"Target column not found. Expected one of {TARGET_RAW_CANDIDATES}."
        )
    return df


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3, 4]), 4)
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2, 3]), 3)
    df["SEX"] = df["SEX"].where(df["SEX"].isin([1, 2]), 2)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PAY_MEAN"] = df[PAY_COLS].mean(axis=1)
    df["PAY_MAX"] = df[PAY_COLS].max(axis=1)
    df["DELINQ_COUNT"] = (df[PAY_COLS] > 0).sum(axis=1)
    df["BILL_MEAN"] = df[BILL_COLS].mean(axis=1)
    df["PAY_AMT_MEAN"] = df[PAY_AMT_COLS].mean(axis=1)
    df["UTILIZATION"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["UTILIZATION"] = df["UTILIZATION"].clip(lower=-2.0, upper=5.0).fillna(0.0)
    denom = df["BILL_AMT1"].where(df["BILL_AMT1"] > 0, np.nan)
    df["PAYMENT_RATIO"] = (df["PAY_AMT1"] / denom).clip(lower=0.0, upper=10.0)
    df["PAYMENT_RATIO"] = df["PAYMENT_RATIO"].fillna(0.0)
    df["LIMIT_PER_AGE"] = df["LIMIT_BAL"] / df["AGE"].clip(lower=18)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SEX_MALE"] = (df["SEX"] == 1).astype(int)
    edu = pd.get_dummies(df["EDUCATION"], prefix="EDUCATION", drop_first=True).astype(int)
    mar = pd.get_dummies(df["MARRIAGE"], prefix="MARRIAGE", drop_first=True).astype(int)
    df = df.drop(columns=["SEX", "EDUCATION", "MARRIAGE"])
    df = pd.concat([df, edu, mar], axis=1)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_categoricals(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    return df


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["IS_DEFAULT"],
        random_state=seed,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified three-way split (train / validation / test).

    ``test_size`` and ``val_size`` are fractions of the *whole* dataset. The
    test set is held out first and never touched during training or model
    selection; the validation set drives early stopping and threshold
    selection. The original index is preserved (no reset) so callers can map
    rows back to the raw frame.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be in (0, 1) and sum to < 1.")
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["IS_DEFAULT"],
        random_state=seed,
    )
    val_fraction_of_remainder = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_remainder,
        stratify=train_val["IS_DEFAULT"],
        random_state=seed,
    )
    return train, val, test


def build_sample(raw_df: pd.DataFrame, test_idx: pd.Index, n: int = 1000) -> pd.DataFrame:
    """Build a public sample from the raw test set (raw features + IS_DEFAULT_TRUE)."""
    sample = raw_df.loc[test_idx].sample(n=min(n, len(test_idx)), random_state=42)
    sample = sample.rename(columns={"IS_DEFAULT": "IS_DEFAULT_TRUE"})
    return sample.reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_PATH} ...")
    raw = load_raw()
    print(f"  raw shape: {raw.shape}")
    print(f"  default rate: {raw['IS_DEFAULT'].mean():.4f}")

    null_rates = raw.isnull().mean().sort_values(ascending=False)
    print("  top null rates:")
    print(null_rates.head(5).to_string())

    features = build_feature_matrix(raw)

    # Index is preserved through feature engineering, so the split indices map
    # straight back to the raw rows.
    train, val, test = split_train_val_test(features)
    test_raw_idx = test.index

    feature_names = [c for c in features.columns if c != "IS_DEFAULT"]
    print(f"  n_features: {len(feature_names)}")
    print(f"  train: {len(train)} ({train['IS_DEFAULT'].mean():.4f} default)")
    print(f"  val  : {len(val)} ({val['IS_DEFAULT'].mean():.4f} default)")
    print(f"  test : {len(test)} ({test['IS_DEFAULT'].mean():.4f} default)")

    train.reset_index(drop=True).to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.reset_index(drop=True).to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.reset_index(drop=True).to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    with open(PROCESSED_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Public sample = exactly the raw rows held out in the test split.
    sample = build_sample(raw, test_raw_idx, n=1000)
    sample.to_csv(SAMPLE_DIR / "sample_1000.csv", index=False)
    print(f"  sample_1000.csv: {sample.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
