"""CLI bulk interaction simulator for live telemetry, drift demos, and retraining.

Usage:
    # Local mode (scores directly with baseline_model.pkl, logs to data/live/interactions.csv)
    python scripts/simulate_interactions.py --profile drifted --rows 150 --run-drift --run-retrain

    # HTTP API mode (hits a running FastAPI server's /telemetry/simulate endpoint)
    python scripts/simulate_interactions.py --api-url http://localhost:8000 --profile drifted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from src.features import ENGINEERED_FEATURES, TARGET_COL  # noqa: E402
from src.monitoring import run_full_monitoring  # noqa: E402
from src.telemetry import (  # noqa: E402
    append_interactions,
    generate_simulated_batch,
    get_live_buffer_path,
    load_live_interactions,
)


def simulate_via_api(
    api_url: str,
    profile: str,
    rows: int,
    seed: int,
    run_drift: bool,
    run_retrain: bool,
) -> dict:
    url = api_url.rstrip("/") + "/telemetry/simulate"
    resp = requests.post(
        url,
        json={
            "profile": profile,
            "count": rows,
            "seed": seed,
            "run_drift_check": run_drift,
            "trigger_retrain": run_retrain,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def simulate_locally(
    profile: str,
    rows: int,
    seed: int,
    model_path: str,
    buffer_path: str | Path | None = None,
    run_drift: bool = False,
    run_retrain: bool = False,
) -> dict:
    batch_df = generate_simulated_batch(profile=profile, count=rows, seed=seed)
    X = batch_df[ENGINEERED_FEATURES].astype(float)

    model_file = Path(model_path)
    if model_file.exists():
        model = joblib.load(model_file)
        preds = [int(p) for p in model.predict(X)]
        if hasattr(model, "predict_proba"):
            probs = [float(p) for p in model.predict_proba(X)[:, 1]]
        else:
            probs = [float(p) for p in preds]
        version = model_file.stem
    else:
        preds = [int(y) for y in batch_df[TARGET_COL].tolist()]
        probs = [0.85 if p == 1 else 0.15 for p in preds]
        version = "fallback_unloaded"

    records = []
    for idx, row in batch_df.iterrows():
        rec = {col: float(row[col]) for col in ENGINEERED_FEATURES}
        rec.update(
            {
                "source": f"simulator_{profile}",
                "skill_name": str(row.get("skill_name", "Digital Literacy")),
                "prediction": preds[idx],
                "probability": probs[idx],
                "needs_intervention": bool(preds[idx] == 0),
                "model_version": version,
                TARGET_COL: int(row[TARGET_COL]),
            }
        )
        records.append(rec)

    target_buf = Path(buffer_path) if buffer_path else get_live_buffer_path()
    written = append_interactions(records, buffer_path=target_buf)
    intervention_rate = sum(1 for p in preds if p == 0) / max(len(preds), 1)

    result: dict = {
        "profile": profile,
        "generated_rows": written,
        "buffer_path": str(target_buf),
        "intervention_rate": round(intervention_rate, 4),
    }

    if run_drift:
        ref_path = REPO_ROOT / "data" / "processed" / "train.csv"
        if ref_path.exists():
            ref_raw = pd.read_csv(ref_path)
            ref_df = ref_raw.sample(n=min(len(ref_raw), 2000), random_state=seed)
        else:
            ref_df = generate_simulated_batch(
                profile="normal", count=max(rows, 300), seed=seed + 7
            )
        live_df = load_live_interactions(buffer_path=target_buf, labeled_only=True)
        drift = run_full_monitoring(
            reference_df=ref_df,
            current_df=live_df if len(live_df) >= 20 else batch_df,
            model_path=str(model_file) if model_file.exists() else None,
            tag=f"sim_{profile}",
            log_to_mlflow=False,
        )
        result["drift_share"] = drift["drift_share"]
        result["dataset_drifted"] = drift["dataset_drifted"]
        result["model_drift_detected"] = bool(
            (drift.get("model_drift") or {}).get("model_drift_detected", False)
        )

    if run_retrain:
        from src.retrain import retrain_cycle

        retrain_res = retrain_cycle(
            ref_path=str(REPO_ROOT / "data" / "processed" / "train.csv"),
            cur_path=str(target_buf),
            from_live=True,
            force=False,
        )
        result["retrain"] = {
            "status": retrain_res.get("status"),
            "retrained": retrain_res.get("retrained"),
            "promoted": retrain_res.get("promoted"),
            "candidate_f1": retrain_res.get("new_f1"),
        }

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate realistic student interactions for telemetry, drift & retraining."
    )
    parser.add_argument(
        "--profile",
        choices=["normal", "struggling", "drifted"],
        default="drifted",
        help="Traffic profile to simulate (default: drifted)",
    )
    parser.add_argument("--rows", type=int, default=120, help="Number of student interactions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--api-url",
        default=None,
        help="If set (e.g. http://localhost:8000), submit via POST /telemetry/simulate",
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "models" / "baseline_model.pkl"),
        help="Model path for local mode",
    )
    parser.add_argument("--run-drift", action="store_true", help="Run drift check after logging")
    parser.add_argument("--run-retrain", action="store_true", help="Trigger retrain after drift")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.api_url:
        out = simulate_via_api(
            api_url=args.api_url,
            profile=args.profile,
            rows=args.rows,
            seed=args.seed,
            run_drift=args.run_drift,
            run_retrain=args.run_retrain,
        )
    else:
        out = simulate_locally(
            profile=args.profile,
            rows=args.rows,
            seed=args.seed,
            model_path=args.model,
            run_drift=args.run_drift,
            run_retrain=args.run_retrain,
        )
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
