"""Live telemetry buffer and realistic student traffic generator for Phase 3.

Persists live ``/predict`` and tutor feedback records (11 canonical
``ENGINEERED_FEATURES`` plus model prediction, probability, intervention flag,
and optional ground-truth ``correct`` outcome) into a lightweight CSV buffer
(default ``data/live/interactions.csv``, git-ignored) so real and simulated
interactions feed directly into drift checks and ``src/retrain.py``.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.features import TARGET_COL
from src.induce_drift import generate_synthetic_baseline, induce_concept_drift
from src.storage import (
    DEFAULT_LIVE_BUFFER_PATH,
    LocalStorageBackend,
    TELEMETRY_COLUMNS,
    get_storage,
)

__all__ = [
    "DEFAULT_LIVE_BUFFER_PATH",
    "FEATURE_DESCRIPTIONS",
    "TELEMETRY_COLUMNS",
    "append_interactions",
    "generate_simulated_batch",
    "get_live_buffer_path",
    "get_live_buffer_stats",
    "load_live_interactions",
    "reset_live_telemetry",
    "upsert_interaction_outcome",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "log_opportunity": {
        "label": "Skill Practice Depth (ln(1 + opp))",
        "baseline": 2.15,
        "unit": "log-count",
        "meaning": "How many prior times the student practiced this skill.",
    },
    "attempt_count": {
        "label": "Problem Attempts",
        "baseline": 1.0,
        "unit": "tries",
        "meaning": "Repeated tries on the current item (strongest struggle indicator).",
    },
    "hint_count": {
        "label": "Hints Requested",
        "baseline": 0.0,
        "unit": "hints",
        "meaning": "Number of hints opened on the current problem.",
    },
    "bottom_hint": {
        "label": "Viewed Bottom-Out Hint",
        "baseline": 0.0,
        "unit": "0/1 flag",
        "meaning": "1 if the student revealed the final answer-level hint.",
    },
    "hint_ratio": {
        "label": "Hint Reliance Ratio",
        "baseline": 0.0,
        "unit": "0–1 ratio",
        "meaning": "Fraction of available hints consumed on this problem.",
    },
    "user_prev_acc": {
        "label": "Prior Student Accuracy",
        "baseline": 0.74,
        "unit": "0–1 rate",
        "meaning": "Expanding historical accuracy across earlier problems.",
    },
    "problem_difficulty": {
        "label": "Item Pass Rate (Ease)",
        "baseline": 0.70,
        "unit": "0–1 rate",
        "meaning": "Historical pass rate for this specific problem (lower = harder).",
    },
    "skill_difficulty": {
        "label": "Skill Domain Pass Rate",
        "baseline": 0.69,
        "unit": "0–1 rate",
        "meaning": "Historical mastery rate across the broader skill category.",
    },
    "is_skill_unmapped": {
        "label": "Unmapped Skill Flag",
        "baseline": 0.0,
        "unit": "0/1 flag",
        "meaning": "1 when the item is not yet mapped to a named curriculum skill.",
    },
    "log_response_time": {
        "label": "First-Response Latency (ln ms)",
        "baseline": 9.45,
        "unit": "ln(ms)",
        "meaning": "Log milliseconds spent before first answer (~9.4 ≈ 12.7s).",
    },
    "is_rapid_guess": {
        "label": "Rapid Guess (< 3s)",
        "baseline": 0.0,
        "unit": "0/1 flag",
        "meaning": "1 if the student answered in under 3 seconds (disengaged guessing).",
    },
}

_buffer_lock = threading.Lock()


def get_live_buffer_path() -> Path:
    """Resolve the active live interactions CSV path (overridable via env var)."""
    env_path = os.environ.get("LIVE_INTERACTIONS_PATH", "").strip()
    return Path(env_path) if env_path else DEFAULT_LIVE_BUFFER_PATH


def _resolve_backend(buffer_path: Optional[str | Path] = None):
    """Use the active storage backend unless an explicit CSV path override is passed."""
    if buffer_path is not None:
        return LocalStorageBackend(buffer_path=Path(buffer_path))
    return get_storage()


def _derive_opportunity(log_opportunity: float) -> int:
    try:
        return max(1, int(round(math.expm1(float(log_opportunity)))))
    except (OverflowError, ValueError, TypeError):
        return 1


def append_interactions(
    rows: Iterable[dict[str, Any]],
    buffer_path: Optional[str | Path] = None,
) -> int:
    """Thread-safely append one or more interaction records to the active store."""
    return _resolve_backend(buffer_path).append_interactions(rows)


def upsert_interaction_outcome(
    interaction_id: str,
    correct: int,
    fallback_record: Optional[dict[str, Any]] = None,
    buffer_path: Optional[str | Path] = None,
) -> bool:
    """Update ``correct`` in-place for an existing ``session_id`` row (or append once if absent).

    Returns ``True`` if an existing row was updated in-place, ``False`` if newly appended.
    """
    return _resolve_backend(buffer_path).upsert_interaction_outcome(
        interaction_id=interaction_id,
        correct=correct,
        fallback_record=fallback_record,
    )


def load_live_interactions(
    buffer_path: Optional[str | Path] = None,
    labeled_only: bool = False,
    since_row: int = 0,
) -> pd.DataFrame:
    """Load the live interactions buffer as a DataFrame (optionally sliced after watermark)."""
    return _resolve_backend(buffer_path).load_interactions(
        labeled_only=labeled_only,
        since_row=since_row,
    )


def reset_live_telemetry(
    buffer_path: Optional[str | Path] = None,
    drift_dir: Optional[str | Path] = None,
    include_students: bool = False,
) -> dict[str, Any]:
    """Clear the live interactions buffer and generated drift/retrain state."""
    dpath = Path(drift_dir) if drift_dir is not None else None
    return _resolve_backend(buffer_path).reset_telemetry(
        drift_dir=dpath, include_students=include_students
    )


def get_live_buffer_stats(
    buffer_path: Optional[str | Path] = None,
    watermark_rows: int = 0,
) -> dict[str, Any]:
    """Return summary counts (total + post-retrain window) for Prometheus and UI."""
    return _resolve_backend(buffer_path).get_interaction_stats(watermark_rows=watermark_rows)


def generate_simulated_batch(
    profile: str = "normal",
    count: int = 100,
    seed: int = 42,
    reference_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Generate a realistic 11-feature labeled student batch for demos and drift tests.

    Profiles:
      - ``normal``: matches reference ASSISTments distribution (~0%–9% drift share).
      - ``struggling``: elevated attempts/hints on 4–5 features (~36%–45% drift share).
      - ``drifted``: realistic exam/cohort covariate shift across 6–7 behavioral features
        (~55%–73% drift share) + concept drift, keeping structural features stable.
    """
    profile_lc = profile.strip().lower()
    if profile_lc not in {"normal", "struggling", "drifted"}:
        raise ValueError(f"Unsupported profile {profile!r}; use normal, struggling, or drifted.")

    default_ref = REPO_ROOT / "data" / "processed" / "train.csv"
    if not default_ref.exists():
        default_ref = REPO_ROOT / "data" / "processed" / "test.csv"
    ref_path = Path(reference_csv) if reference_csv else default_ref
    if ref_path.exists():
        base = pd.read_csv(ref_path)
        replace = len(base) < count
        df = base.sample(n=count, replace=replace, random_state=seed).reset_index(drop=True)
    else:
        df = generate_synthetic_baseline(n_rows=count, seed=seed)

    rng = np.random.default_rng(seed)

    if profile_lc == "struggling":
        df = df.copy()
        # Shift 4 behavioral struggle features while leaving item/domain/timing stable
        df["attempt_count"] = np.clip(
            df["attempt_count"] + rng.integers(2, 5, size=count), 1.0, 50.0
        )
        df["hint_count"] = np.clip(df["hint_count"] + rng.integers(1, 4, size=count), 0.0, 20.0)
        df["bottom_hint"] = (df["hint_count"] >= 2).astype(float)
        df["hint_ratio"] = np.clip(
            df["hint_count"] / np.maximum(df["attempt_count"] + df["hint_count"], 1.0), 0.0, 1.0
        )
        df[TARGET_COL] = (rng.random(count) < 0.22).astype(int)
    elif profile_lc == "drifted":
        df = df.copy()
        # Shift 7 behavioral/difficulty features (leaving skill_difficulty, is_skill_unmapped,
        # log_opportunity, and is_rapid_guess anchored to reference for realistic ~64% drift)
        df["attempt_count"] = np.clip(
            df["attempt_count"] + rng.poisson(2.6, size=count), 1.0, 50.0
        )
        df["hint_count"] = np.clip(df["hint_count"] + rng.poisson(2.1, size=count), 0.0, 20.0)
        df["bottom_hint"] = np.where(df["hint_count"] >= 2, 1.0, df["bottom_hint"])
        df["hint_ratio"] = np.clip(
            df["hint_count"] / np.maximum(df["attempt_count"] + df["hint_count"], 1.0), 0.0, 1.0
        )
        df["user_prev_acc"] = np.clip(
            df["user_prev_acc"] - 0.24 + rng.normal(0, 0.04, size=count), 0.0, 1.0
        )
        df["problem_difficulty"] = np.clip(df["problem_difficulty"] - 0.22, 0.05, 0.99)
        df["log_response_time"] = np.clip(
            df["log_response_time"] + rng.normal(0.85, 0.25, size=count), 0.2, 24.0
        )
        df = induce_concept_drift(df, flip_rate=0.38, seed=seed)

    # Ensure strictly bounded Pydantic-compatible 11-feature ranges
    df["log_opportunity"] = np.clip(df["log_opportunity"].astype(float), 0.0, 20.0)
    df["attempt_count"] = np.clip(df["attempt_count"].astype(float), 0.0, 50.0)
    df["hint_count"] = np.clip(df["hint_count"].astype(float), 0.0, 50.0)
    df["bottom_hint"] = (df["bottom_hint"].astype(float) >= 0.5).astype(int)
    df["hint_ratio"] = np.clip(df["hint_ratio"].astype(float), 0.0, 1.0)
    df["user_prev_acc"] = np.clip(df["user_prev_acc"].astype(float), 0.0, 1.0)
    df["problem_difficulty"] = np.clip(df["problem_difficulty"].astype(float), 0.0, 1.0)
    df["skill_difficulty"] = np.clip(df["skill_difficulty"].astype(float), 0.0, 1.0)
    df["is_skill_unmapped"] = (df["is_skill_unmapped"].astype(float) >= 0.5).astype(int)
    df["log_response_time"] = np.clip(df["log_response_time"].astype(float), 0.0, 25.0)
    df["is_rapid_guess"] = (df["is_rapid_guess"].astype(float) >= 0.5).astype(int)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df
