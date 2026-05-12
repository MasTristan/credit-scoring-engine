"""Data preparation pipeline for the UCI credit-card-default scoring model.

Input  : data/raw/uci_credit_card.csv
Output : data/processed/train.parquet
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

    train, test = split_train_test(features)
    train_raw_idx = train.index
    test_raw_idx = test.index

    feature_names = [c for c in features.columns if c != "IS_DEFAULT"]
    print(f"  n_features: {len(feature_names)}")
    print(f"  train: {len(train)} ({train['IS_DEFAULT'].mean():.4f} default)")
    print(f"  test : {len(test)} ({test['IS_DEFAULT'].mean():.4f} default)")

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    with open(PROCESSED_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Build public sample from the underlying raw rows that ended up in test.
    # We re-run the split on the raw frame with the same seed to recover the
    # test indices in raw space.
    raw_train, raw_test = train_test_split(
        raw,
        test_size=0.2,
        stratify=raw["IS_DEFAULT"],
        random_state=42,
    )
    sample = build_sample(raw, raw_test.index, n=1000)
    sample.to_csv(SAMPLE_DIR / "sample_1000.csv", index=False)
    print(f"  sample_1000.csv: {sample.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
