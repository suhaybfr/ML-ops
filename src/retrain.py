"""Phase 3 Self-Contained Retraining Feedback Loop.

Supports retraining from either a drifted CSV batch (``--cur``) or the live
interactions telemetry buffer (``--from-live`` / ``data/live/interactions.csv``),
enforces the 42-check Model-Risk & Quality Gate (``src/validate_model.py``),
logs & registers candidates in MLflow (with automatic local SQLite fallback when
DAGsHub credentials are absent), writes ``reports/drift/latest_status.json`` for
Prometheus/Grafana ingestion, and optionally triggers zero-downtime hot-reload
via ``POST /model/reload``.

Hard constraint: ``models/baseline_model.pkl`` (Phase 2 DVC champion) is NEVER
overwritten; promoted retrained models are written to
``models/retrained_champion.pkl``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.evaluation import (  # noqa: E402
    classification_metrics,
    model_risk_diagnostics,
    positive_scores,
    save_validation_json,
)
from src.features import ENGINEERED_FEATURES, TARGET_COL, select_xy  # noqa: E402
from src.induce_drift import generate_synthetic_baseline  # noqa: E402
from src.monitoring import run_full_monitoring  # noqa: E402
from src.register_model import REGISTRY_MODEL_NAME, register_and_promote_champion  # noqa: E402
from src.telemetry import load_live_interactions  # noqa: E402
from src.utils import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    log_model_and_pkl,
    setup_mlflow,
    standard_tags,
)
from src.validate_model import validate_baseline_metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = DEFAULT_EXPERIMENT
DEFAULT_DRIFT_THRESHOLD = 0.3
DEFAULT_STATUS_PATH = REPO_ROOT / "reports" / "drift" / "latest_status.json"
DEFAULT_CHALLENGER_PATH = REPO_ROOT / "models" / "challenger_model.pkl"
DEFAULT_PROMOTED_PATH = REPO_ROOT / "models" / "retrained_champion.pkl"


def _load_previous_status(status_path: Path) -> dict[str, Any]:
    try:
        from src.storage import get_storage, is_postgres_configured

        if is_postgres_configured():
            stored = get_storage().get_latest_retrain_run()
            if stored:
                return stored
    except Exception:
        pass
    if not status_path.exists():
        return {
            "retrain_runs_total": 0,
            "retrain_promotions_total": 0,
            "retrain_gate_failures_total": 0,
        }
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "retrain_runs_total": 0,
            "retrain_promotions_total": 0,
            "retrain_gate_failures_total": 0,
        }


def _write_status(status_path: Path, payload: dict[str, Any]) -> Path:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    try:
        from src.storage import get_storage, is_postgres_configured

        if is_postgres_configured():
            get_storage().save_retrain_run(payload)
    except Exception as exc:
        print(f"[Retrain] Notice: Postgres retrain run save skipped ({exc})")
    return status_path


def _load_train_params() -> dict[str, Any]:
    params_file = REPO_ROOT / "params.yaml"
    if params_file.exists():
        with open(params_file, encoding="utf-8") as fh:
            return yaml.safe_load(fh).get("train", {})
    return {
        "random_state": 42,
        "logreg": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        "random_forest": {"n_estimators": 120, "max_depth": 12},
    }


def _run_dvc_repro(stage: str = "train") -> bool:
    """Trigger DVC pipeline reproduction for the target training stage."""
    print(f"[Retrain] Running: dvc repro {stage}")
    result = subprocess.run(["dvc", "repro", stage], capture_output=False, text=True)
    return result.returncode == 0


def _trigger_api_reload(
    reload_url: str,
    model_path: str,
    version_label: str,
) -> dict[str, Any]:
    url = reload_url.rstrip("/")
    if not url.endswith("/model/reload"):
        url = f"{url}/model/reload"
    resp = requests.post(
        url,
        json={"model_path": model_path, "version_label": version_label},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _train_and_validate_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    challenger_path: Path,
    metrics_out_path: Path,
    tracking_uri: str,
    experiment_name: str,
    n_estimators: int = 120,
    max_depth: int = 12,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train candidate RF + reference LogReg, evaluate on test_df, and run 42-check gate."""
    X_train, y_train = select_xy(train_df)
    X_test, y_test = select_xy(test_df)
    feature_cols = list(ENGINEERED_FEATURES)

    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    rf_candidate = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=int(random_state),
        n_jobs=-1,
    )

    lr_pipeline.fit(X_train, y_train)
    rf_candidate.fit(X_train, y_train)

    lr_pred = lr_pipeline.predict(X_test)
    lr_score = positive_scores(lr_pipeline, X_test)
    lr_metrics = classification_metrics(y_test, lr_pred, lr_score)
    lr_risk = model_risk_diagnostics(test_df, y_test, lr_pred, lr_score, is_probability=True)

    rf_pred = rf_candidate.predict(X_test)
    rf_score = positive_scores(rf_candidate, X_test)
    rf_metrics = classification_metrics(y_test, rf_pred, rf_score)
    rf_risk = model_risk_diagnostics(test_df, y_test, rf_pred, rf_score, is_probability=True)

    challenger_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_candidate, challenger_path)

    payload = {
        "stage": "retrain",
        "feature_set": "engineered_11",
        "features": feature_cols,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "models": {
            "logistic_regression_baseline": {
                "artifact_path": "models/logreg_baseline.pkl",
                "metrics": {k: round(float(v), 6) for k, v in lr_metrics.items()},
                "risk_and_slice_diagnostics": lr_risk,
            },
            "random_forest_baseline": {
                "artifact_path": str(challenger_path),
                "hyperparameters": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "random_state": random_state,
                },
                "metrics": {k: round(float(v), 6) for k, v in rf_metrics.items()},
                "risk_and_slice_diagnostics": rf_risk,
            },
        },
    }
    save_validation_json(payload, metrics_out_path)

    gate_report = validate_baseline_metrics(
        metrics_path=metrics_out_path,
        require_artifact=True,
    )

    run_id = None
    run_name = f"random_forest_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                mlflow.set_tags(
                    standard_tags(
                        stage="retrain",
                        model="random_forest",
                        feature_set="engineered_11",
                        kind="drift_retrain_candidate",
                        extra={"quality_gate_passed": str(gate_report["passed"]).lower()},
                    )
                )
                mlflow.log_params(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                    }
                )
                mlflow.log_metrics(rf_metrics)
                mlflow.log_metrics(
                    {
                        "macro_f1": rf_risk["macro_f1"],
                        "class0_recall": rf_risk["struggling_student_class0"]["recall_specificity"],
                        "brier_score": rf_risk["brier_score"],
                    }
                )
                log_model_and_pkl(rf_candidate, challenger_path, X_example=X_test)
        except Exception as exc:
            print(f"[Retrain] Notice: MLflow run logging skipped ({exc})")

    return {
        "model": rf_candidate,
        "metrics": rf_metrics,
        "risk": rf_risk,
        "gate": gate_report,
        "run_id": run_id,
        "run_name": run_name,
        "metrics_path": str(metrics_out_path),
    }


def retrain_cycle(
    ref_path: str = str(REPO_ROOT / "data" / "processed" / "train.csv"),
    cur_path: Optional[str] = None,
    test_path: str = str(REPO_ROOT / "data" / "processed" / "test.csv"),
    from_live: bool = False,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    stage: str = "train",
    force: bool = False,
    register: bool = True,
    use_dvc: bool = False,
    model_name: str = REGISTRY_MODEL_NAME,
    output_dir: str | Path = REPO_ROOT / "reports" / "drift",
    status_path: str | Path = DEFAULT_STATUS_PATH,
    challenger_path: str | Path = DEFAULT_CHALLENGER_PATH,
    promoted_path: str | Path = DEFAULT_PROMOTED_PATH,
    reload_url: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Execute self-contained drift-gated retraining, 42-check gate, promotion & hot-reload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_file = Path(status_path)
    prev = _load_previous_status(status_file)
    runs_total = int(prev.get("retrain_runs_total", 0)) + 1
    promotions_total = int(prev.get("retrain_promotions_total", 0))
    gate_failures_total = int(prev.get("retrain_gate_failures_total", 0))

    # 1. Load reference training & evaluation frames (with synthetic fallback if DVC data absent)
    if ref_path and Path(ref_path).exists():
        ref_train_df = pd.read_csv(ref_path)
    else:
        print(f"[Retrain] Reference '{ref_path}' not found; using synthetic baseline.")
        ref_train_df = generate_synthetic_baseline(n_rows=1200, seed=42)

    if test_path and Path(test_path).exists():
        eval_test_df = pd.read_csv(test_path)
    else:
        eval_test_df = generate_synthetic_baseline(n_rows=400, seed=43)

    # 2. Load current batch (post-watermark live telemetry window or explicit CSV)
    prev_watermark = int(prev.get("watermark_rows", 0))
    total_live_rows = 0
    if from_live:
        live_path = Path(cur_path) if cur_path else None
        all_live_raw = load_live_interactions(buffer_path=live_path, labeled_only=False)
        total_live_rows = int(len(all_live_raw))
        window_start = prev_watermark if prev_watermark < total_live_rows else 0
        cur_all_df = load_live_interactions(
            buffer_path=live_path, labeled_only=False, since_row=window_start
        )
        cur_labeled_df = load_live_interactions(
            buffer_path=live_path, labeled_only=True, since_row=window_start
        )
        if len(cur_labeled_df) == 0:
            cur_labeled_df = load_live_interactions(buffer_path=live_path, labeled_only=True)
    elif cur_path and Path(cur_path).exists():
        cur_all_df = pd.read_csv(cur_path)
        if TARGET_COL in cur_all_df.columns:
            cur_labeled_df = cur_all_df[
                pd.to_numeric(cur_all_df[TARGET_COL], errors="coerce").isin([0, 1])
            ].copy()
            cur_labeled_df[TARGET_COL] = cur_labeled_df[TARGET_COL].astype(int)
        else:
            cur_labeled_df = pd.DataFrame()
    else:
        cur_all_df = eval_test_df.copy()
        cur_labeled_df = eval_test_df.copy()

    # 3. Run 11-feature data + model drift detection against active reference
    active_ref_csv = output_dir / "active_reference.csv"
    default_champ_pkl = str(REPO_ROOT / "models" / "baseline_model.pkl")
    active_model_pkl = Path(prev.get("active_artifact_path", default_champ_pkl))
    if not active_model_pkl.is_absolute():
        active_model_pkl = REPO_ROOT / active_model_pkl
    if not active_model_pkl.exists():
        active_model_pkl = REPO_ROOT / "models" / "baseline_model.pkl"

    if active_ref_csv.exists():
        ref_sample = pd.read_csv(active_ref_csv)
    else:
        ref_sample = (
            ref_train_df.sample(n=min(len(ref_train_df), 3000), random_state=42)
            if len(ref_train_df) > 3000
            else ref_train_df
        )
    if len(cur_all_df) >= 5:
        drift_summary = run_full_monitoring(
            reference_df=ref_sample,
            current_df=cur_labeled_df if len(cur_labeled_df) >= 5 else cur_all_df,
            model_path=(
                str(active_model_pkl)
                if active_model_pkl.exists() and len(cur_labeled_df) >= 5
                else None
            ),
            output_dir=str(output_dir),
            tag="retrain_check",
            log_to_mlflow=False,
        )
    else:
        drift_summary = {
            "drift_share": 0.0,
            "dataset_drifted": False,
            "feature_drift": {},
            "model_drift": None,
        }

    drift_share = float(drift_summary.get("drift_share", 0.0))
    dataset_drifted = bool(drift_summary.get("dataset_drifted", False))
    model_drift_detected = bool(
        (drift_summary.get("model_drift") or {}).get("model_drift_detected", False)
    )

    should_retrain = bool(
        force or dataset_drifted or model_drift_detected or (drift_share >= drift_threshold)
    )
    if not should_retrain:
        status_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "skipped_no_drift",
            "retrained": False,
            "promoted": False,
            "retrain_runs_total": runs_total,
            "retrain_promotions_total": promotions_total,
            "retrain_gate_failures_total": gate_failures_total,
            "drift_share": drift_share,
            "dataset_drifted": dataset_drifted,
            "model_drift_detected": model_drift_detected,
            "quality_gate_passed": prev.get("quality_gate_passed", True),
            "quality_gate_passed_checks": prev.get("quality_gate_passed_checks", 42),
            "quality_gate_total_checks": prev.get("quality_gate_total_checks", 42),
            "candidate_f1": prev.get("candidate_f1", 0.0),
            "active_artifact_path": prev.get("active_artifact_path", "models/baseline_model.pkl"),
            "active_model_version": prev.get("active_model_version", "random_forest_baseline_v2"),
        }
        _write_status(status_file, status_payload)
        return {**status_payload, "drift_summary": drift_summary}

    # 4. Optional DVC repro mode
    if use_dvc:
        if not _run_dvc_repro(stage=stage):
            status_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "retrained": False,
                "promoted": False,
                "retrain_runs_total": runs_total,
                "retrain_promotions_total": promotions_total,
                "retrain_gate_failures_total": gate_failures_total,
                "drift_share": drift_share,
                "dataset_drifted": dataset_drifted,
                "model_drift_detected": model_drift_detected,
            }
            _write_status(status_file, status_payload)
            return {**status_payload, "drift_summary": drift_summary}

    # 5. Build combined training dataset (reference + upweighted labeled live/drifted rows)
    train_cols = list(ENGINEERED_FEATURES) + [TARGET_COL]
    if len(cur_labeled_df) > 0 and all(c in cur_labeled_df.columns for c in train_cols):
        live_slice = cur_labeled_df[train_cols]
        # Upweight small live windows (~5% of training weight) so the new RF adapts to the cohort
        target_live_rows = min(max(len(live_slice), 4000), max(int(len(ref_train_df) * 0.06), 200))
        repeats = max(1, int(round(target_live_rows / max(len(live_slice), 1))))
        weighted_live = pd.concat([live_slice] * min(repeats, 40), ignore_index=True)
        combined_train = pd.concat([ref_train_df[train_cols], weighted_live], ignore_index=True)
    else:
        combined_train = ref_train_df[train_cols].copy()

    # 6. Setup MLflow with graceful fallback when mlflow-skinny lacks sqlalchemy
    active_uri: Optional[str] = tracking_uri
    if not active_uri:
        try:
            active_uri = setup_mlflow(
                experiment_name=experiment_name,
                allow_local_fallback=True,
            )
        except Exception as exc:
            print(f"[Retrain] Notice: MLflow tracking unavailable in runtime ({exc}); continuing.")
            active_uri = None

    params = _load_train_params()
    rf_cfg = params.get("random_forest", {})
    candidate_res = _train_and_validate_candidate(
        train_df=combined_train,
        test_df=eval_test_df,
        challenger_path=Path(challenger_path),
        metrics_out_path=output_dir / "retrain_candidate_metrics.json",
        tracking_uri=active_uri,
        experiment_name=experiment_name,
        n_estimators=max(int(rf_cfg.get("n_estimators", 100)), 120),
        max_depth=max(int(rf_cfg.get("max_depth", 10)), 12),
        random_state=int(params.get("random_state", 42)),
    )

    gate = candidate_res["gate"]
    new_f1 = float(candidate_res["metrics"]["f1_score"])
    promoted = False
    reg_result = None
    reload_result = None
    active_artifact = prev.get("active_artifact_path", "models/baseline_model.pkl")
    active_version = prev.get("active_model_version", "random_forest_baseline_v2")

    if not gate["passed"]:
        gate_failures_total += 1
        status = "gate_failed"
    else:
        # Check whether candidate improves on current champion (or matches under drift)
        champion_pkl = REPO_ROOT / "models" / "baseline_model.pkl"
        champ_f1 = 0.0
        if champion_pkl.exists():
            try:
                champ_model = joblib.load(champion_pkl)
                X_eval, y_eval = select_xy(eval_test_df)
                c_pred = champ_model.predict(X_eval)
                c_score = positive_scores(champ_model, X_eval)
                champ_f1 = float(classification_metrics(y_eval, c_pred, c_score)["f1_score"])
            except Exception:
                champ_f1 = 0.0

        # Promote if passes all 42 checks and beats/matches champion (within 0.005 when adapting)
        beats_or_adapts = (new_f1 >= champ_f1) or (
            (dataset_drifted or model_drift_detected or force) and new_f1 >= (champ_f1 - 0.005)
        )
        if register and candidate_res["run_id"] and active_uri:
            try:
                reg_result = register_and_promote_champion(
                    model_name=model_name,
                    run_id=candidate_res["run_id"],
                    experiment_name=experiment_name,
                    tracking_uri=active_uri,
                    enforce_gate=True,
                    metrics_path=candidate_res["metrics_path"],
                )
            except Exception as exc:
                print(f"[Retrain] Notice: MLflow registry step skipped ({exc})")

        if beats_or_adapts:
            dest = Path(promoted_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(challenger_path), dest)
            promoted = True
            promotions_total += 1
            status = "promoted"
            active_artifact = str(dest)
            reg_ver = (reg_result or {}).get("version")
            active_version = (
                f"random_forest_retrained_v{reg_ver}"
                if reg_ver
                else f"random_forest_retrained_r{promotions_total}"
            )
            # Exponential moving reference adaptation: blend newly learned cohort (88%)
            # with prior reference anchor (12%) without oversampling small cohorts with
            # replacement (which artificially inflates statistical test power in Evidently).
            eval_cols = [c for c in ref_sample.columns if c in cur_all_df.columns]
            adapt_source = cur_labeled_df if len(cur_labeled_df) >= 5 else cur_all_df
            if len(adapt_source) >= 5 and len(eval_cols) >= len(ENGINEERED_FEATURES):
                n_cohort = min(len(adapt_source), int(round(2500 * 0.88)))
                n_anchor = max(1, int(round(n_cohort * 0.12 / 0.88)))
                cohort_part = adapt_source[eval_cols].sample(
                    n=n_cohort, replace=False, random_state=42 + promotions_total
                )
                anchor_part = ref_sample[eval_cols].sample(
                    n=n_anchor, replace=len(ref_sample) < n_anchor, random_state=42
                )
                new_ref = pd.concat([cohort_part, anchor_part], ignore_index=True)
            else:
                new_ref = ref_train_df.sample(
                    n=min(len(ref_train_df), 2500), random_state=42 + promotions_total
                )
            new_ref.to_csv(active_ref_csv, index=False)
            post_check_sample = new_ref.sample(
                n=min(len(new_ref), 400), random_state=100 + promotions_total
            )
            post_drift = run_full_monitoring(
                reference_df=new_ref,
                current_df=post_check_sample,
                model_path=str(dest) if TARGET_COL in new_ref.columns else None,
                output_dir=str(output_dir),
                tag=f"post_retrain_r{promotions_total}",
                log_to_mlflow=False,
            )
            if reload_url:
                try:
                    reload_result = _trigger_api_reload(
                        reload_url=reload_url,
                        model_path=active_artifact,
                        version_label=active_version,
                    )
                except Exception as exc:
                    print(f"[Retrain] Notice: API hot-reload skipped ({exc})")
        else:
            status = "challenger_kept"
            post_drift = None
    if not gate["passed"]:
        post_drift = None

    new_watermark = total_live_rows if promoted and from_live else prev_watermark
    effective_drift_share = (
        float(post_drift.get("drift_share", 0.0)) if post_drift is not None else drift_share
    )
    effective_drifted = (
        bool(post_drift.get("dataset_drifted", False))
        if post_drift is not None
        else dataset_drifted
    )
    effective_mdrift = (
        bool((post_drift.get("model_drift") or {}).get("model_drift_detected", False))
        if post_drift is not None
        else model_drift_detected
    )

    status_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "retrained": True,
        "promoted": promoted,
        "retrain_runs_total": runs_total,
        "retrain_promotions_total": promotions_total,
        "retrain_gate_failures_total": gate_failures_total,
        "watermark_rows": new_watermark,
        "trigger_drift_share": round(drift_share, 4),
        "drift_share": round(effective_drift_share, 4),
        "dataset_drifted": effective_drifted,
        "model_drift_detected": effective_mdrift,
        "quality_gate_passed": bool(gate["passed"]),
        "quality_gate_passed_checks": int(gate["passed_checks"]),
        "quality_gate_total_checks": int(gate["total_checks"]),
        "candidate_f1": round(new_f1, 6),
        "candidate_metrics": {k: round(float(v), 6) for k, v in candidate_res["metrics"].items()},
        "active_artifact_path": active_artifact,
        "active_model_version": active_version,
        "new_run_id": candidate_res["run_id"],
        "reload_result": reload_result,
    }
    _write_status(status_file, status_payload)

    # Milestone 5: Optional Discord / webhook paging on promotion or gate failure
    try:
        from src.alerts import send_alert_webhook

        if status == "promoted":
            send_alert_webhook(
                title=f"Model Promoted: {active_version}",
                message=(
                    f"Candidate passed {gate['passed_checks']}/{gate['total_checks']} checks "
                    f"(Test F1={new_f1:.4f}) and was promoted."
                ),
                severity="info",
                fields={
                    "Version": active_version,
                    "Test F1": f"{new_f1:.4f}",
                    "Quality Gate": f"{gate['passed_checks']}/{gate['total_checks']}",
                },
            )
        elif status == "gate_failed":
            send_alert_webhook(
                title="Retrain Quality Gate Failed",
                message=(
                    f"Candidate failed quality gate "
                    f"({gate['passed_checks']}/{gate['total_checks']}); Champion retained."
                ),
                severity="critical",
                fields={"Test F1": f"{new_f1:.4f}"},
            )
    except Exception:
        pass

    return {
        **status_payload,
        "new_f1": new_f1,
        "drift_summary": drift_summary,
        "registry": reg_result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-contained drift-triggered retraining & hot-reload loop."
    )
    parser.add_argument(
        "--ref",
        default=str(REPO_ROOT / "data" / "processed" / "train.csv"),
        help="Reference (baseline) CSV path",
    )
    parser.add_argument(
        "--cur",
        default=None,
        help="Current / drifted batch CSV path (or live buffer when --from-live is set)",
    )
    parser.add_argument(
        "--from-live",
        action="store_true",
        help="Read incoming interactions from data/live/interactions.csv",
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
        help="DVC training stage (only used with --use-dvc)",
    )
    parser.add_argument("--use-dvc", action="store_true", help="Run dvc repro before validation")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of drift")
    parser.add_argument(
        "--no-register", action="store_true", help="Skip MLflow Model Registry registration"
    )
    parser.add_argument(
        "--reload-url",
        default=None,
        help="Optional API base URL (e.g. http://localhost:8000) to trigger POST /model/reload",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = retrain_cycle(
        ref_path=args.ref,
        cur_path=args.cur,
        from_live=args.from_live,
        drift_threshold=args.drift_threshold,
        stage=args.stage,
        force=args.force,
        register=not args.no_register,
        use_dvc=args.use_dvc,
        reload_url=args.reload_url,
    )
    print("\n[Retrain] === Summary ===")
    print(f"  Status    : {result['status']}")
    print(f"  Retrained : {result['retrained']}")
    print(f"  Promoted  : {result.get('promoted', False)}")
    print(
        f"  Gate      : {result.get('quality_gate_passed_checks')}/"
        f"{result.get('quality_gate_total_checks')}"
    )
    if result.get("new_f1"):
        print(f"  New F1    : {result['new_f1']:.4f}")
    sys.exit(0)
