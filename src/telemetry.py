"""Live telemetry buffer and realistic student traffic generator for Phase 3.

Persists live ``/predict`` and tutor feedback records (11 canonical
``ENGINEERED_FEATURES`` plus model prediction, probability, intervention flag,
and optional ground-truth ``correct`` outcome) into a lightweight CSV buffer
(default ``data/live/interactions.csv``, git-ignored) so real and simulated
interactions feed directly into drift checks and ``src/retrain.py``.
"""

from __future__ import annotations

import csv
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.features import ENGINEERED_FEATURES, TARGET_COL
from src.induce_drift import generate_synthetic_baseline, induce_concept_drift

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIVE_BUFFER_PATH = REPO_ROOT / "data" / "live" / "interactions.csv"

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

TELEMETRY_COLUMNS: list[str] = [
    "timestamp",
    "session_id",
    "source",
    "skill_name",
    "opportunity",
    *ENGINEERED_FEATURES,
    "prediction",
    "probability",
    "needs_intervention",
    "model_version",
    TARGET_COL,
]

_buffer_lock = threading.Lock()


def get_live_buffer_path() -> Path:
    """Resolve the active live interactions CSV path (overridable via env var)."""
    env_path = os.environ.get("LIVE_INTERACTIONS_PATH", "").strip()
    return Path(env_path) if env_path else DEFAULT_LIVE_BUFFER_PATH


def _derive_opportunity(log_opportunity: float) -> int:
    try:
        return max(1, int(round(math.expm1(float(log_opportunity)))))
    except (OverflowError, ValueError, TypeError):
        return 1


def append_interactions(
    rows: Iterable[dict[str, Any]],
    buffer_path: Optional[str | Path] = None,
) -> int:
    """Thread-safely append one or more interaction records to the live CSV buffer."""
    target = Path(buffer_path) if buffer_path is not None else get_live_buffer_path()
    records = list(rows)
    if not records:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    formatted: list[dict[str, Any]] = []
    for rec in records:
        skill = rec.get("skill_name")
        if not skill:
            unmapped = int(float(rec.get("is_skill_unmapped", 0))) == 1
            skill = "Unknown" if unmapped else "Digital Literacy"
        raw_target = rec.get(TARGET_COL)
        target_val = "" if raw_target is None or raw_target == "" else int(raw_target)
        opp_val = rec.get("opportunity") or _derive_opportunity(rec.get("log_opportunity", 0.0))
        row = {
            "timestamp": rec.get("timestamp") or now_iso,
            "session_id": rec.get("session_id") or "",
            "source": rec.get("source") or "api_predict",
            "skill_name": skill,
            "opportunity": opp_val,
            "prediction": int(rec["prediction"]),
            "probability": round(float(rec["probability"]), 6),
            "needs_intervention": int(bool(rec["needs_intervention"])),
            "model_version": str(rec.get("model_version") or "unknown"),
            TARGET_COL: target_val,
        }
        for col in ENGINEERED_FEATURES:
            row[col] = float(rec[col])
        formatted.append(row)

    with _buffer_lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_header = not target.exists() or target.stat().st_size == 0
        with open(target, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=TELEMETRY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(formatted)

    return len(formatted)


def upsert_interaction_outcome(
    interaction_id: str,
    correct: int,
    fallback_record: Optional[dict[str, Any]] = None,
    buffer_path: Optional[str | Path] = None,
) -> bool:
    """Update ``correct`` in-place for an existing ``session_id`` row (or append once if absent).

    Returns ``True`` if an existing row was updated in-place, ``False`` if newly appended.
    """
    target = Path(buffer_path) if buffer_path is not None else get_live_buffer_path()
    with _buffer_lock:
        if target.exists() and target.stat().st_size > 0 and interaction_id:
            with open(target, "r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            matched = False
            for r in reversed(rows):
                if r.get("session_id") == interaction_id:
                    r[TARGET_COL] = str(int(correct))
                    matched = True
                    break
            if matched:
                with open(target, "w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=TELEMETRY_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                return True

    if fallback_record is not None:
        rec = dict(fallback_record)
        rec["session_id"] = interaction_id
        rec[TARGET_COL] = int(correct)
        append_interactions([rec], buffer_path=target)
    return False


def load_live_interactions(
    buffer_path: Optional[str | Path] = None,
    labeled_only: bool = False,
    since_row: int = 0,
) -> pd.DataFrame:
    """Load the live interactions buffer as a DataFrame (optionally sliced after watermark)."""
    target = Path(buffer_path) if buffer_path is not None else get_live_buffer_path()
    if not target.exists() or target.stat().st_size == 0:
        return pd.DataFrame(columns=TELEMETRY_COLUMNS)

    with _buffer_lock:
        df = pd.read_csv(target)

    if since_row > 0:
        df = df.iloc[since_row:].copy()

    for col in ENGINEERED_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    if labeled_only and TARGET_COL in df.columns:
        df = df[df[TARGET_COL].isin([0, 1])].copy()
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df.reset_index(drop=True)


def reset_live_telemetry(
    buffer_path: Optional[str | Path] = None,
    drift_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Clear the live interactions buffer and generated drift/retrain JSON files."""
    target = Path(buffer_path) if buffer_path is not None else get_live_buffer_path()
    with _buffer_lock:
        if target.exists():
            target.unlink()

    removed_files = []
    if drift_dir is not None:
        dpath = Path(drift_dir)
        if dpath.exists():
            for item in dpath.iterdir():
                if item.is_file() and item.suffix in {".json", ".html", ".csv"}:
                    item.unlink()
                    removed_files.append(item.name)

    return {"reset": True, "buffer_path": str(target), "cleared_drift_files": removed_files}


def get_live_buffer_stats(
    buffer_path: Optional[str | Path] = None,
    watermark_rows: int = 0,
) -> dict[str, Any]:
    """Return summary counts (total + post-retrain window) for Prometheus and UI."""
    df = load_live_interactions(buffer_path=buffer_path, labeled_only=False)
    total = int(len(df))
    if total == 0:
        return {
            "total_rows": 0,
            "labeled_rows": 0,
            "window_rows": 0,
            "intervention_rate": 0.0,
            "buffer_path": str(Path(buffer_path) if buffer_path else get_live_buffer_path()),
        }

    labeled = int(df[TARGET_COL].isin([0, 1]).sum()) if TARGET_COL in df.columns else 0
    window_df = df.iloc[watermark_rows:] if 0 < watermark_rows < total else df
    interventions = (
        float(pd.to_numeric(window_df["needs_intervention"], errors="coerce").fillna(0).mean())
        if "needs_intervention" in window_df.columns and len(window_df) > 0
        else 0.0
    )
    return {
        "total_rows": total,
        "labeled_rows": labeled,
        "window_rows": int(len(df) - min(watermark_rows, total)),
        "intervention_rate": round(interventions, 4),
        "buffer_path": str(Path(buffer_path) if buffer_path else get_live_buffer_path()),
    }


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
