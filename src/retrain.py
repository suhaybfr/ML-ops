"""Phase 3 Retraining Feedback Loop (drift-triggered DVC retrain + Registry promotion).

Kept cleanly decoupled from Phase 2 CI runs (invoke on schedule or against live
production batches rather than static train/test splits).

Workflow:
1. Computes 11-feature drift between reference training data and incoming batch.
2. If drift exceeds ``--drift-threshold`` (or ``--force`` is set), runs
   ``dvc repro train`` to retrain the Random Forest champion stage.
3. Queries the newly produced ``train`` run by explicit stage/model tags.
4. Enforces quality gates and registers/promotes in MLflow Model Registry via
   ``src.register_model`` ONLY when the new F1 beats the current ``@champion``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from mlflow.tracking import MlflowClient

from src.monitoring import compute_drift_report
from src.register_model import REGISTRY_MODEL_NAME, register_and_promote_champion
from src.utils import DEFAULT_EXPERIMENT, setup_mlflow

EXPERIMENT_NAME = DEFAULT_EXPERIMENT
DEFAULT_DRIFT_THRESHOLD = 0.3


def _run_drift_check(ref_path: str, cur_path: str) -> dict:
    """Run 11-feature drift check without polluting baseline experiment runs."""
    if not os.path.exists(ref_path) or not os.path.exists(cur_path):
        print(f"[Retrain] Data files {ref_path} or {cur_path} not found. Skipping drift check.")
        return {"drift_share": 0.0, "dataset_drifted": False, "feature_drift": {}}
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)
    return compute_drift_report(ref_df, cur_df, log_to_mlflow=False)


def _run_dvc_repro(stage: str = "train") -> bool:
    """Trigger DVC pipeline reproduction for the target training stage."""
    print(f"[Retrain] Running: dvc repro {stage}")
    result = subprocess.run(["dvc", "repro", stage], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[Retrain] dvc repro {stage} failed (exit code {result.returncode})")
        return False
    print(f"[Retrain] dvc repro {stage} completed successfully")
    return True


def _find_latest_stage_run(client: MlflowClient, experiment_name: str, stage: str):
    """Find the most recent training run for ``stage`` (ignoring drift_check runs)."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    model_clause = " and tags.model = 'random_forest'" if stage == "train" else ""
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.stage = '{stage}'{model_clause}",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def retrain_cycle(
    ref_path: str,
    cur_path: str,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    stage: str = "train",
    force: bool = False,
    register: bool = True,
    model_name: str = REGISTRY_MODEL_NAME,
) -> dict:
    """Execute drift-gated retraining and conditional champion promotion."""
    tracking_uri = setup_mlflow(experiment_name=EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)

    print("\n[Retrain] === Step 1: Drift Detection ===")
    drift_summary = _run_drift_check(ref_path, cur_path)
    drift_share = float(drift_summary.get("drift_share", 0.0))
    print(f"[Retrain] Drift share: {drift_share:.2%} (threshold: {drift_threshold:.0%})")

    should_retrain = bool(force or (drift_share >= drift_threshold))
    if not should_retrain:
        print("[Retrain] No significant drift detected — skipping retrain.")
        return {"drift_summary": drift_summary, "retrained": False, "promoted": False}

    print(
        f"[Retrain] {'Forced retrain requested.' if force else 'Drift threshold exceeded.'} "
        f"Triggering DVC stage '{stage}'..."
    )

    print("\n[Retrain] === Step 2: DVC Pipeline Reproduction ===")
    if not _run_dvc_repro(stage=stage):
        return {"drift_summary": drift_summary, "retrained": False, "promoted": False}

    print("\n[Retrain] === Step 3: Fetching Latest Training Run ===")
    latest_run = _find_latest_stage_run(client, EXPERIMENT_NAME, stage=stage)
    if latest_run is None:
        print(f"[Retrain] Could not locate MLflow run for stage={stage!r}.")
        return {"drift_summary": drift_summary, "retrained": True, "promoted": False}

    new_run_id = latest_run.info.run_id
    new_f1 = float(latest_run.data.metrics.get("f1_score", 0.0))
    print(f"[Retrain] Candidate run: {new_run_id} ({latest_run.info.run_name}, F1={new_f1:.4f})")

    promoted = False
    reg_result = None
    if register and new_run_id:
        print("\n[Retrain] === Step 4: Quality-Gated Model Registration ===")
        reg_result = register_and_promote_champion(
            model_name=model_name,
            run_id=new_run_id,
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=tracking_uri,
            enforce_gate=(stage == "train"),
        )
        promoted = bool(reg_result["promoted"])

    return {
        "drift_summary": drift_summary,
        "retrained": True,
        "promoted": promoted,
        "new_f1": new_f1,
        "new_run_id": new_run_id,
        "registry": reg_result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drift-triggered retraining feedback loop.")
    parser.add_argument(
        "--ref",
        default=os.path.join("data", "processed", "train.csv"),
        help="Reference (baseline) CSV path",
    )
    parser.add_argument(
        "--cur",
        default=os.path.join("data", "processed", "test.csv"),
        help="Current / incoming batch CSV path",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help=f"Drifted feature share triggering retrain (default: {DEFAULT_DRIFT_THRESHOLD})",
    )
    parser.add_argument(
        "--stage",
        default="train",
        choices=["train", "train_svm"],
        help="DVC training stage to reproduce (default: train -> Random Forest champion)",
    )
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of drift")
    parser.add_argument(
        "--no-register", action="store_true", help="Skip MLflow Model Registry registration"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = retrain_cycle(
        ref_path=args.ref,
        cur_path=args.cur,
        drift_threshold=args.drift_threshold,
        stage=args.stage,
        force=args.force,
        register=not args.no_register,
    )
    print("\n[Retrain] === Summary ===")
    print(f"  Retrained : {result['retrained']}")
    print(f"  Promoted  : {result.get('promoted', False)}")
    if result.get("new_f1"):
        print(f"  New F1    : {result['new_f1']:.4f}")
    sys.exit(0)
