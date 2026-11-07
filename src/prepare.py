"""Stage 1 (prepare): Partition versioned engineered features into train/test sets.

Loads ``data/processed/engineered_features.csv`` (tracked by DVC), validates its
schema and target integrity, caps extreme ``attempt_count`` outliers so linear
and kernel scalers are not distorted, performs a reproducible stratified split,
and writes ``data/processed/train.csv``, ``data/processed/test.csv``, and
``reports/prepare_summary.json``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.evaluation import save_validation_json
from src.features import (
    ENGINEERED_FEATURES,
    LEAKY_COLS,
    METADATA_COLS,
    TARGET_COL,
    validate_engineered_dataframe,
)


def load_params() -> dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["prepare"]


def prepare_data() -> dict:
    params = load_params()
    input_path = Path("data") / "processed" / "engineered_features.csv"
    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Prepare] Loading engineered dataset from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"[Prepare] Full engineered dataset shape: {df.shape}")

    # Filter to valid binary targets and validate schema
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    validate_engineered_dataframe(df)

    # Cap extreme attempt_count outliers (raw max = 3,824) to stabilize StandardScaler
    max_att = params.get("max_attempt_count", 50)
    raw_max_att = float(df["attempt_count"].max())
    capped_rows = int((df["attempt_count"] > max_att).sum())
    if max_att:
        df["attempt_count"] = df["attempt_count"].clip(lower=0, upper=max_att)

    full_rows = len(df)
    full_pos_rate = float(df[TARGET_COL].mean())

    # Stratified subsample for fast reproducible pipeline execution
    sample_size = params.get("sample_size")
    if sample_size and sample_size < len(df):
        print(f"[Prepare] Stratified subsample to {sample_size:,} rows (seed={params['random_state']})")
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=params["random_state"],
            stratify=df[TARGET_COL],
        )
        df = df.reset_index(drop=True)

    # Keep metadata (for slice model-risk diagnostics), 11 predictors, and target.
    # Explicitly drop `first_try_correct` so downstream stages cannot leak labels.
    keep_cols = [c for c in METADATA_COLS if c in df.columns] + ENGINEERED_FEATURES + [TARGET_COL]
    df = df[keep_cols]

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[TARGET_COL],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "source_dataset": str(input_path),
        "full_dataset_rows": int(full_rows),
        "full_dataset_positive_rate": round(full_pos_rate, 4),
        "sampled_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_positive_rate": round(float(train_df[TARGET_COL].mean()), 4),
        "test_positive_rate": round(float(test_df[TARGET_COL].mean()), 4),
        "predictor_feature_count": len(ENGINEERED_FEATURES),
        "predictor_features": ENGINEERED_FEATURES,
        "excluded_leaky_columns": LEAKY_COLS,
        "metadata_columns_retained_for_slice_checks": [
            c for c in METADATA_COLS if c in df.columns
        ],
        "attempt_count_cap": {
            "max_attempt_count": max_att,
            "raw_max_attempt_count": raw_max_att,
            "capped_rows_in_full_dataset": capped_rows,
        },
        "parameters": params,
    }
    summary_path = Path("reports") / "prepare_summary.json"
    save_validation_json(summary, summary_path)

    print(f"[Prepare] Saved train set ({len(train_df):,} rows) -> {train_path}")
    print(f"[Prepare] Saved test set  ({len(test_df):,} rows) -> {test_path}")
    print(f"[Prepare] Saved summary   -> {summary_path}")
    return summary


if __name__ == "__main__":
    prepare_data()
