"""
Retraining feedback loop for the Student Performance SVM model.

This script:
1. Loads the reference (training) dataset and a current (production) batch.
2. Runs Evidently drift detection.
3. If drift exceeds the threshold, triggers a DVC pipeline reproduction
   (``dvc repro train_svm``) to retrain on the latest data.
4. Logs the retrain run and drift metrics to MLflow.
5. Registers the new model in the MLflow Model Registry if it beats the
   currently registered champion on F1-score.

Usage:
    # Full auto-retrain cycle (check drift → maybe retrain → maybe promote)
    python src/retrain.py \
        --ref  data/processed/train.csv \
        --cur  data/processed/test.csv \
        --drift-threshold 0.3

    # Force retrain even if no drift
    python src/retrain.py --force

Environment variables (same as CI):
    MLFLOW_TRACKING_USERNAME
    MLFLOW_TRACKING_PASSWORD
"""

import argparse
import os
import subprocess
import sys

import mlflow
import mlflow.sklearn
import pandas as pd

from utils import setup_mlflow

EXPERIMENT_NAME = "student_performance_baseline"
REGISTRY_MODEL_NAME = "student_performance_svm"
DEFAULT_DRIFT_THRESHOLD = 0.3   # retrain if ≥ 30 % of features drifted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_drift_check(ref_path: str, cur_path: str) -> dict:
    """Run Evidently drift check and return summary dict."""
    try:
        from monitoring import compute_drift_report
    except ImportError:
        try:
            from src.monitoring import compute_drift_report
        except ImportError:
            print("[Retrain] monitoring.py not found — skipping drift check.")
            return {"drift_share": 0.0, "dataset_drifted": False, "feature_drift": {}}

    if not os.path.exists(ref_path) or not os.path.exists(cur_path):
        print(f"[Retrain] Data files {ref_path} or {cur_path} not found. Skipping drift check.")
        return {"drift_share": 0.0, "dataset_drifted": False, "feature_drift": {}}
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)
    return compute_drift_report(ref_df, cur_df)


def _run_dvc_repro(stage: str = "train_svm") -> bool:
    """Trigger DVC pipeline reproduction for the given stage."""
    print(f"[Retrain] Running: dvc repro {stage}")
    result = subprocess.run(
        ["dvc", "repro", stage],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"[Retrain] ❌ dvc repro failed (exit code {result.returncode})")
        return False
    print(f"[Retrain] ✅ dvc repro {stage} completed successfully")
    return True


def _get_champion_f1(client: mlflow.tracking.MlflowClient) -> float:
    """Return the F1-score of the current Production model, or 0.0 if none."""
    try:
        versions = client.get_latest_versions(REGISTRY_MODEL_NAME, stages=["Production"])
        if not versions:
            return 0.0
        run = client.get_run(versions[0].run_id)
        return run.data.metrics.get("f1_score", 0.0)
    except Exception as exc:
        print(f"[Retrain] Could not fetch champion metrics: {exc}")
        return 0.0


def _register_model(client: mlflow.tracking.MlflowClient, run_id: str, f1: float) -> None:
    """Register the newly trained model and promote it if it beats the champion."""
    model_uri = f"runs:/{run_id}/model"
    champion_f1 = _get_champion_f1(client)

    mv = mlflow.register_model(model_uri=model_uri, name=REGISTRY_MODEL_NAME)
    print(f"[Retrain] Registered model version {mv.version} (run: {run_id})")

    if f1 > champion_f1:
        client.transition_model_version_stage(
            name=REGISTRY_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(
            f"[Retrain] ✅ Promoted v{mv.version} to Production "
            f"(new F1={f1:.4f} > champion F1={champion_f1:.4f})"
        )
    else:
        client.transition_model_version_stage(
            name=REGISTRY_MODEL_NAME,
            version=mv.version,
            stage="Staging",
        )
        print(
            f"[Retrain] ⚠️  New model moved to Staging — "
            f"F1={f1:.4f} did not beat champion F1={champion_f1:.4f}"
        )


# ---------------------------------------------------------------------------
# Main retrain cycle
# ---------------------------------------------------------------------------

def retrain_cycle(
    ref_path: str,
    cur_path: str,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    force: bool = False,
    register: bool = True,
) -> dict:
    """
    Full retraining feedback loop.

    Parameters
    ----------
    ref_path        : Path to reference (training baseline) CSV.
    cur_path        : Path to current / incoming production batch CSV.
    drift_threshold : Fraction of drifted features that triggers a retrain.
    force           : If True, retrain regardless of drift level.
    register        : If True, register retrained model in the Model Registry.

    Returns
    -------
    dict with keys: ``drift_summary``, ``retrained``, ``promoted``.
    """
    tracking_uri = setup_mlflow(experiment_name=EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # ── Step 1: Drift detection ───────────────────────────────────────────
    print("\n[Retrain] === Step 1: Drift Detection ===")
    drift_summary = _run_drift_check(ref_path, cur_path)
    drift_share = drift_summary.get("drift_share", 0.0)
    print(f"[Retrain] Drift share: {drift_share:.2%}  (threshold: {drift_threshold:.0%})")

    should_retrain = force or (drift_share >= drift_threshold)

    if not should_retrain:
        print("[Retrain] ✅ No significant drift — skipping retrain.")
        return {"drift_summary": drift_summary, "retrained": False, "promoted": False}

    print(
        f"[Retrain] {'Forced retrain requested.' if force else 'Drift threshold exceeded.'} "
        "Triggering DVC pipeline..."
    )

    # ── Step 2: Retrain via DVC ───────────────────────────────────────────
    print("\n[Retrain] === Step 2: DVC Pipeline Reproduction ===")
    success = _run_dvc_repro(stage="train_svm")
    if not success:
        return {"drift_summary": drift_summary, "retrained": False, "promoted": False}

    # ── Step 3: Fetch new run metrics from MLflow ─────────────────────────
    print("\n[Retrain] === Step 3: Fetching New Run Metrics ===")
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    new_f1 = 0.0
    new_run_id = None
    if exp:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            r = runs[0]
            new_run_id = r.info.run_id
            new_f1 = r.data.metrics.get("f1_score", 0.0)
            print(f"[Retrain] Latest run ID  : {new_run_id}")
            print(f"[Retrain] Latest F1-score: {new_f1:.4f}")

    # Log drift metrics and reports to the latest run for traceability
    if new_run_id:
        with mlflow.start_run(run_id=new_run_id):
            mlflow.log_metrics({
                "drift_share": drift_share,
                "retrain_triggered": 1.0,
            })
            html_p = drift_summary.get("html_path")
            if html_p and os.path.exists(html_p):
                mlflow.log_artifact(html_p, artifact_path="drift_reports")
            json_p = drift_summary.get("json_path")
            if json_p and os.path.exists(json_p):
                mlflow.log_artifact(json_p, artifact_path="drift_reports")

    # ── Step 4: Register model ────────────────────────────────────────────
    promoted = False
    if register and new_run_id:
        print("\n[Retrain] === Step 4: Model Registration ===")
        _register_model(client, new_run_id, new_f1)
        promoted = True

    return {
        "drift_summary": drift_summary,
        "retrained": True,
        "promoted": promoted,
        "new_f1": new_f1,
        "new_run_id": new_run_id,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Drift-triggered retraining feedback loop."
    )
    parser.add_argument(
        "--ref",
        default=os.path.join("data", "processed", "train.csv"),
        help="Reference (baseline) CSV path",
    )
    parser.add_argument(
        "--cur",
        default=os.path.join("data", "processed", "test.csv"),
        help="Current / incoming data CSV path",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help=f"Fraction of drifted features that triggers retrain (default: {DEFAULT_DRIFT_THRESHOLD})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if no drift detected",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip MLflow Model Registry registration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = retrain_cycle(
        ref_path=args.ref,
        cur_path=args.cur,
        drift_threshold=args.drift_threshold,
        force=args.force,
        register=not args.no_register,
    )
    print("\n[Retrain] === Summary ===")
    print(f"  Retrained : {result['retrained']}")
    print(f"  Promoted  : {result.get('promoted', False)}")
    if result.get("new_f1"):
        print(f"  New F1    : {result['new_f1']:.4f}")
    sys.exit(0)
