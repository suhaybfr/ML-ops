"""
Drift induction utility for Phase 3 monitoring & retraining demonstrations.

Generates controlled **Data / Covariate Drift** and **Model / Concept Drift**
on the 11-feature ASSISTments schema (``src/features.py``) so the team can
demonstrate Evidently drift detection, Prometheus/Grafana alerting, and
drift-triggered retraining end-to-end.

Usage:
    # Induce both covariate and concept drift from test.csv and run monitoring
    python src/induce_drift.py --run-monitoring --no-mlflow

    # Induce covariate drift only with custom intensity
    python src/induce_drift.py --mode data --intensity 1.2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

if not __package__:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from src.features import TARGET_COL, validate_engineered_dataframe  # noqa: E402


def generate_synthetic_baseline(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic 11-feature dataset matching ASSISTments distributions."""
    rng = np.random.default_rng(seed)
    opportunity = rng.integers(1, 25, size=n_rows)
    attempt_count = np.clip(rng.poisson(1.4, size=n_rows) + 1, 1, 50)
    hint_count = np.clip(rng.poisson(0.6, size=n_rows), 0, 10)
    bottom_hint = ((hint_count >= 3) & (rng.random(n_rows) < 0.7)).astype(float)
    hint_ratio = hint_count / np.maximum(attempt_count + hint_count, 1)
    user_prev_acc = np.clip(rng.normal(0.68, 0.14, size=n_rows), 0.0, 1.0)
    problem_difficulty = np.clip(rng.normal(0.66, 0.15, size=n_rows), 0.05, 0.99)
    skill_difficulty = np.clip(rng.normal(0.67, 0.12, size=n_rows), 0.10, 0.99)
    is_skill_unmapped = (rng.random(n_rows) < 0.15).astype(float)
    is_rapid_guess = (rng.random(n_rows) < 0.08).astype(float)
    raw_rt_sec = np.where(
        is_rapid_guess == 1,
        rng.uniform(0.4, 2.8, size=n_rows),
        rng.lognormal(mean=2.2, sigma=0.7, size=n_rows),
    )
    log_response_time = np.log1p(raw_rt_sec)

    # Underlying ground-truth signal consistent with the ASSISTments task
    logit = (
        2.2 * (user_prev_acc - 0.5)
        + 1.6 * (problem_difficulty - 0.5)
        - 0.65 * hint_count
        - 0.45 * (attempt_count - 1)
        - 1.2 * is_rapid_guess
        + 0.8
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    correct = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "user_id": rng.integers(10000, 99999, size=n_rows),
            "problem_id": rng.integers(1000, 9999, size=n_rows),
            "skill_name": np.where(is_skill_unmapped == 1, "Unknown", "Linear Equations"),
            "opportunity": opportunity,
            "log_opportunity": np.log1p(opportunity),
            "attempt_count": attempt_count.astype(float),
            "hint_count": hint_count.astype(float),
            "bottom_hint": bottom_hint,
            "hint_ratio": hint_ratio,
            "user_prev_acc": user_prev_acc,
            "problem_difficulty": problem_difficulty,
            "skill_difficulty": skill_difficulty,
            "is_skill_unmapped": is_skill_unmapped,
            "log_response_time": log_response_time,
            "is_rapid_guess": is_rapid_guess,
            TARGET_COL: correct,
        }
    )
    validate_engineered_dataframe(df)
    return df


def induce_data_drift(
    df: pd.DataFrame,
    intensity: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Induce covariate/data drift simulating harder exams and rushed student behavior."""
    validate_engineered_dataframe(df)
    rng = np.random.default_rng(seed)
    drifted = df.copy()
    n = len(drifted)

    # 1. Students struggle more -> more attempts & hints
    extra_attempts = rng.poisson(max(0.5, 2.5 * intensity), size=n)
    drifted["attempt_count"] = np.clip(drifted["attempt_count"] + extra_attempts, 1.0, 50.0)

    extra_hints = rng.poisson(max(0.5, 2.0 * intensity), size=n)
    drifted["hint_count"] = np.clip(drifted["hint_count"] + extra_hints, 0.0, 20.0)
    drifted["hint_ratio"] = drifted["hint_count"] / np.maximum(
        drifted["attempt_count"] + drifted["hint_count"], 1.0
    )
    drifted["bottom_hint"] = np.where(
        drifted["hint_count"] >= 3, 1.0, drifted["bottom_hint"]
    )

    # 2. Lower historical accuracy & harder problem/skill cohort
    drifted["user_prev_acc"] = np.clip(
        drifted["user_prev_acc"] - 0.22 * intensity + rng.normal(0, 0.05, size=n), 0.0, 1.0
    )
    drifted["problem_difficulty"] = np.clip(
        drifted["problem_difficulty"] - 0.20 * intensity, 0.05, 0.99
    )
    drifted["skill_difficulty"] = np.clip(
        drifted["skill_difficulty"] - 0.18 * intensity, 0.05, 0.99
    )

    # 3. More cold-start / unmapped skills & rapid guessing (<3s)
    unmapped_flip = rng.random(n) < min(0.45, 0.25 * intensity)
    drifted["is_skill_unmapped"] = np.where(unmapped_flip, 1.0, drifted["is_skill_unmapped"])

    rapid_flip = rng.random(n) < min(0.50, 0.30 * intensity)
    drifted["is_rapid_guess"] = np.where(rapid_flip, 1.0, drifted["is_rapid_guess"])
    drifted["log_response_time"] = np.where(
        drifted["is_rapid_guess"] == 1.0,
        np.log1p(rng.uniform(0.3, 2.5, size=n)),
        np.clip(drifted["log_response_time"] - 0.45 * intensity, 0.2, 8.0),
    )
    drifted["log_opportunity"] = np.clip(
        drifted["log_opportunity"] * max(0.3, 1.0 - 0.35 * intensity), 0.0, 6.0
    )

    validate_engineered_dataframe(drifted)
    return drifted


def induce_concept_drift(
    df: pd.DataFrame,
    flip_rate: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """Induce model/concept drift by shifting P(correct | X) on key subpopulations."""
    validate_engineered_dataframe(df)
    rng = np.random.default_rng(seed)
    drifted = df.copy()

    # Simulate a curriculum/rubric change: students who previously got hints or
    # rushed behave differently, and overall target relationship shifts.
    high_risk_mask = (drifted["hint_count"] >= 1) | (drifted["is_rapid_guess"] == 1)
    flip_prob = np.where(high_risk_mask, min(0.95, flip_rate + 0.20), flip_rate * 0.6)
    should_flip = rng.random(len(drifted)) < flip_prob

    drifted[TARGET_COL] = np.where(
        should_flip, 1 - drifted[TARGET_COL], drifted[TARGET_COL]
    ).astype(int)
    validate_engineered_dataframe(drifted)
    return drifted


def run_drift_induction(
    input_path: Optional[str] = os.path.join("data", "processed", "test.csv"),
    ref_output_path: str = os.path.join("reports", "drift", "reference_sample.csv"),
    cur_output_path: str = os.path.join("reports", "drift", "drifted_batch.csv"),
    mode: str = "both",
    intensity: float = 1.0,
    flip_rate: float = 0.35,
    sample_rows: int = 2000,
    seed: int = 42,
    run_monitoring: bool = False,
    model_path: Optional[str] = os.path.join("models", "baseline_model.pkl"),
    log_to_mlflow: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """Load or synthesize reference data, induce drift, save CSVs, and optionally monitor."""
    os.makedirs(os.path.dirname(cur_output_path) or ".", exist_ok=True)

    if input_path and os.path.exists(input_path):
        base_df = pd.read_csv(input_path)
        if sample_rows and len(base_df) > sample_rows:
            base_df = base_df.sample(n=sample_rows, random_state=seed).reset_index(drop=True)
        print(f"[InduceDrift] Loaded {len(base_df):,} rows from {input_path}")
        ref_df = base_df
    else:
        print(
            f"[InduceDrift] Input '{input_path}' not found locally; "
            f"generating {sample_rows:,}-row synthetic 11-feature baseline."
        )
        ref_df = generate_synthetic_baseline(n_rows=sample_rows, seed=seed)
        ref_df.to_csv(ref_output_path, index=False)
        base_df = generate_synthetic_baseline(n_rows=sample_rows, seed=seed + 1)

    cur_df = base_df.copy()
    if mode in ("data", "both"):
        cur_df = induce_data_drift(cur_df, intensity=intensity, seed=seed)
    if mode in ("concept", "both"):
        cur_df = induce_concept_drift(cur_df, flip_rate=flip_rate, seed=seed)

    cur_df.to_csv(cur_output_path, index=False)
    print(f"[InduceDrift] Saved drifted batch ({mode}) -> {cur_output_path}")

    monitoring_summary = None
    if run_monitoring:
        from src.monitoring import run_full_monitoring

        monitoring_summary = run_full_monitoring(
            reference_df=ref_df,
            current_df=cur_df,
            model_path=model_path,
            output_dir=os.path.dirname(cur_output_path) or os.path.join("reports", "drift"),
            tag=f"induced_{mode}",
            log_to_mlflow=log_to_mlflow,
        )

    return ref_df, cur_df, monitoring_summary


def _parse_args():
    parser = argparse.ArgumentParser(description="Induce data & model drift for Phase 3 demo.")
    parser.add_argument(
        "--input",
        default=os.path.join("data", "processed", "test.csv"),
        help="Reference/test CSV path (falls back to synthetic 11-feature data if missing)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("reports", "drift", "drifted_batch.csv"),
        help="Output path for the drifted CSV",
    )
    parser.add_argument(
        "--mode",
        choices=["data", "concept", "both"],
        default="both",
        help="Type of drift to induce (default: both)",
    )
    parser.add_argument("--intensity", type=float, default=1.0, help="Data drift strength")
    parser.add_argument("--flip-rate", type=float, default=0.35, help="Concept drift label flip")
    parser.add_argument("--rows", type=int, default=2000, help="Rows to sample/generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run-monitoring",
        action="store_true",
        help="Immediately run src/monitoring.py on the generated drifted batch",
    )
    parser.add_argument(
        "--model",
        default=os.path.join("models", "baseline_model.pkl"),
        help="Model artifact path for performance drift evaluation",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_drift_induction(
        input_path=args.input,
        cur_output_path=args.output,
        mode=args.mode,
        intensity=args.intensity,
        flip_rate=args.flip_rate,
        sample_rows=args.rows,
        seed=args.seed,
        run_monitoring=args.run_monitoring,
        model_path=args.model,
        log_to_mlflow=not args.no_mlflow,
    )
