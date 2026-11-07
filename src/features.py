"""Feature schema definitions and validation helpers for the ASSISTments pipeline.

Aligns the DVC modelling stages with the versioned feature matrix at
``data/processed/engineered_features.csv`` (documented in
``reports/data_dictionary.md``) while strictly excluding leaky or identifier
columns from model predictors.
"""

from __future__ import annotations

import pandas as pd

TARGET_COL = "correct"

# Entity / concept identifiers retained in train/test CSVs for subgroup slice
# diagnostics, never passed as model features.
METADATA_COLS = [
    "user_id",
    "problem_id",
    "skill_name",
    "opportunity",
]

# Excluded from predictors because it explicitly uses `correct == 1`.
LEAKY_COLS = [
    "first_try_correct",
]

# 11 hypothesis-driven predictor features used across all baseline & SVM models.
ENGINEERED_FEATURES = [
    "log_opportunity",
    "attempt_count",
    "hint_count",
    "bottom_hint",
    "hint_ratio",
    "user_prev_acc",
    "problem_difficulty",
    "skill_difficulty",
    "is_skill_unmapped",
    "log_response_time",
    "is_rapid_guess",
]

EXPECTED_ENGINEERED_CSV_COLS = [
    "user_id",
    "problem_id",
    "skill_name",
    "opportunity",
    "log_opportunity",
    "attempt_count",
    "hint_count",
    "bottom_hint",
    "hint_ratio",
    "first_try_correct",
    "user_prev_acc",
    "problem_difficulty",
    "skill_difficulty",
    "is_skill_unmapped",
    "log_response_time",
    "is_rapid_guess",
    "correct",
]


def validate_engineered_dataframe(df: pd.DataFrame) -> None:
    """Validate required columns, non-null predictors, and binary target values."""
    missing = [c for c in ENGINEERED_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in engineered dataset: {missing}")

    null_counts = df[ENGINEERED_FEATURES + [TARGET_COL]].isna().sum()
    bad_nulls = null_counts[null_counts > 0].to_dict()
    if bad_nulls:
        raise ValueError(f"Unexpected nulls in predictor/target columns: {bad_nulls}")

    unique_targets = set(df[TARGET_COL].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target column {TARGET_COL!r} must be binary {{0, 1}}, got {unique_targets}")


def select_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract ordered 11-feature matrix ``X`` and integer target ``y``."""
    validate_engineered_dataframe(df)
    X = df[ENGINEERED_FEATURES].astype(float)
    y = df[TARGET_COL].astype(int)
    return X, y
