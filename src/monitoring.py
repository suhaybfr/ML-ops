"""Monitoring module — data drift & model performance drift detection.

Computes statistical drift across all 11 canonical ``ENGINEERED_FEATURES``
defined in ``src.features`` between the training reference distribution and
incoming prediction batches. Uses Evidently AI when installed, with a built-in
two-sample Kolmogorov-Smirnov (``scipy.stats.ks_2samp``) fallback so drift
reports (HTML + JSON) work cleanly across all Python versions.

When a trained model artifact is provided (``--model``), it also evaluates
model/concept drift (F1, Accuracy, ROC-AUC, and struggling-student Class-0
Recall degradation) and generates an Evidently ``ClassificationPreset`` report.

Results are written to ``reports/drift/`` as HTML + JSON (including
``latest_drift_summary.json`` for Prometheus/Grafana ingestion).

Usage (standalone):
    python -m src.monitoring --ref data/processed/train.csv \
                             --cur data/processed/test.csv \
                             --no-mlflow
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html as html_lib
import json
import os
from pathlib import Path
import sys
from typing import Optional

if not __package__:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.evaluation import (  # noqa: E402
    classification_metrics,
    model_risk_diagnostics,
    positive_scores,
)
from src.features import ENGINEERED_FEATURES, TARGET_COL  # noqa: E402

# Canonical 11 engineered predictor features (kept as FEATURE_COLS for compatibility)
FEATURE_COLS = list(ENGINEERED_FEATURES)
MONITORING_EXPERIMENT = "student_performance_monitoring"


def _load_evidently_components():
    """Lazily resolve Evidently Report / Presets / ColumnMapping across 0.4-0.7+."""
    import pandas.core.dtypes.common as _pcommon

    if not hasattr(_pcommon, "infer_dtype_from_object"):
        try:
            from pandas._libs.lib import infer_dtype as _infer_dtype

            _pcommon.infer_dtype_from_object = lambda val: (_infer_dtype(val), None)  # noqa: E731
        except Exception:
            pass

    column_mapping_cls = None
    for mod_name in (
        "evidently",
        "evidently.pipeline.column_mapping",
        "evidently.legacy.pipeline.column_mapping",
    ):
        try:
            mod = __import__(mod_name, fromlist=["ColumnMapping"])
            column_mapping_cls = getattr(mod, "ColumnMapping", None)
            if column_mapping_cls is not None:
                break
        except Exception:
            continue

    for report_mod, preset_mod in (
        ("evidently.report", "evidently.metric_preset"),
        ("evidently.legacy.report", "evidently.legacy.metric_preset"),
    ):
        try:
            rmod = __import__(report_mod, fromlist=["Report"])
            pmod = __import__(preset_mod, fromlist=["DataDriftPreset", "ClassificationPreset"])
            return (
                getattr(rmod, "Report"),
                getattr(pmod, "DataDriftPreset"),
                getattr(pmod, "ClassificationPreset"),
                column_mapping_cls,
            )
        except Exception:
            continue

    return None, None, None, None


def _compute_ks_fallback_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    shared_features: list[str],
    alpha: float = 0.05,
    dataset_drift_threshold: float = 0.3,
) -> tuple[dict, str]:
    """Deterministic KS two-sample drift fallback when Evidently is not installed."""
    drift_by_columns: dict[str, dict] = {}
    drifted_count = 0

    for col in shared_features:
        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().to_numpy()
        cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().to_numpy()
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            stat, p_val, drifted = 0.0, 1.0, False
        else:
            res = ks_2samp(ref_vals, cur_vals)
            stat, p_val = float(res.statistic), float(res.pvalue)
            # Require both statistical significance and practical KS effect (>= 0.10)
            drifted = bool(p_val < alpha and stat >= 0.10)
        if drifted:
            drifted_count += 1
        drift_by_columns[col] = {
            "column_name": col,
            "stattest_name": "ks_2samp",
            "drift_score": round(stat, 6),
            "p_value": round(p_val, 6),
            "ref_mean": round(float(np.mean(ref_vals)), 4) if len(ref_vals) else None,
            "cur_mean": round(float(np.mean(cur_vals)), 4) if len(cur_vals) else None,
            "drift_detected": drifted,
        }

    drift_share = float(drifted_count / len(shared_features)) if shared_features else 0.0
    dataset_drift = bool(drift_share >= dataset_drift_threshold)

    report_dict = {
        "engine": "scipy_ks_fallback",
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {
                    "number_of_columns": len(shared_features),
                    "number_of_drifted_columns": drifted_count,
                    "share_of_drifted_columns": round(drift_share, 6),
                    "dataset_drift": dataset_drift,
                    "drift_by_columns": drift_by_columns,
                },
            }
        ],
    }

    rows_html = "\n".join(
        f"<tr><td>{html_lib.escape(col)}</td>"
        f"<td>{info['drift_score']:.4f}</td>"
        f"<td>{info['p_value']:.4g}</td>"
        f"<td>{info['ref_mean']}</td>"
        f"<td>{info['cur_mean']}</td>"
        f"<td>{'DRIFTED' if info['drift_detected'] else 'OK'}</td></tr>"
        for col, info in drift_by_columns.items()
    )
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Data Drift Report</title></head><body>"
        f"<h1>Data Drift Summary (drift share: {drift_share:.2%})</h1>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Feature</th><th>KS Statistic</th><th>p-value</th>"
        "<th>Ref Mean</th><th>Cur Mean</th><th>Status</th></tr>"
        f"{rows_html}</table></body></html>\n"
    )
    return report_dict, html_doc


def _extract_drift_summary(report_dict: dict) -> dict:
    """Parse drift report dictionary into a compact summary with per-feature scores."""
    summary: dict = {
        "drift_share": 0.0,
        "dataset_drifted": False,
        "feature_drift": {},
        "feature_drift_scores": {},
    }
    try:
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            if "share_of_drifted_columns" in result:
                summary["drift_share"] = float(result["share_of_drifted_columns"])
                summary["dataset_drifted"] = bool(result.get("dataset_drift", False))
            drift_by_col = result.get("drift_by_columns", {})
            for col, col_result in drift_by_col.items():
                if col in FEATURE_COLS:
                    summary["feature_drift"][col] = bool(col_result.get("drift_detected", False))
                    raw_score = col_result.get("drift_score")
                    if raw_score is not None:
                        try:
                            summary["feature_drift_scores"][col] = round(float(raw_score), 6)
                        except (TypeError, ValueError):
                            pass
    except Exception as exc:
        print(f"[Monitoring] Warning: could not parse drift summary: {exc}")
    return summary


def _write_latest_summary(output_dir: str, payload: dict, save_to_db: bool = True) -> str:
    """Write standardized ``latest_drift_summary.json`` and mirror to Postgres if configured."""
    latest_path = os.path.join(output_dir, "latest_drift_summary.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    if save_to_db:
        try:
            from src.storage import get_storage, is_postgres_configured

            if is_postgres_configured():
                get_storage().save_drift_snapshot(payload)
        except Exception as exc:
            print(f"[Monitoring] Notice: Postgres drift snapshot save skipped ({exc})")
    return latest_path


def _log_drift_to_mlflow(
    tag: str,
    drift_results: dict,
    model_drift: Optional[dict] = None,
) -> None:
    """Log drift artifacts to dedicated monitoring experiment without polluting baseline runs."""
    try:
        import mlflow
        from src.utils import setup_mlflow

        setup_mlflow(experiment_name=MONITORING_EXPERIMENT)
        with mlflow.start_run(run_name=f"drift_check_{tag}"):
            mlflow.set_tags(
                {
                    "phase": "phase_3_monitoring",
                    "kind": "data_and_model_drift" if model_drift else "data_drift",
                    "feature_set": "engineered_11",
                }
            )
            artifact_keys = (
                "html_path",
                "json_path",
                "classification_html_path",
                "latest_summary_path",
            )
            for key in artifact_keys:
                path = drift_results.get(key)
                if path and os.path.exists(path):
                    mlflow.log_artifact(path, artifact_path="drift_reports")

            metrics_payload = {
                "drift_share": float(drift_results.get("drift_share", 0.0)),
                "dataset_drifted": (
                    1.0 if drift_results.get("dataset_drifted", False) else 0.0
                ),
                "drifted_features_count": float(
                    sum(1 for v in drift_results.get("feature_drift", {}).values() if v)
                ),
            }
            if model_drift:
                metrics_payload.update(
                    {
                        "ref_f1_score": float(model_drift["reference_metrics"]["f1_score"]),
                        "cur_f1_score": float(model_drift["current_metrics"]["f1_score"]),
                        "f1_drop": float(model_drift["f1_drop"]),
                        "model_drift_detected": (
                            1.0 if model_drift.get("model_drift_detected", False) else 0.0
                        ),
                    }
                )
            mlflow.log_metrics(metrics_payload)
            print(
                f"[Monitoring] Drift artifacts logged to MLflow experiment "
                f"{MONITORING_EXPERIMENT!r}."
            )
    except Exception as exc:
        print(f"[Monitoring] Notice: MLflow artifact logging skipped ({exc})")


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = os.path.join("reports", "drift"),
    tag: Optional[str] = None,
    log_to_mlflow: bool = False,
    _save_to_db: bool = True,
) -> dict:
    """Generate a data-drift report across the 11 engineered features."""
    os.makedirs(output_dir, exist_ok=True)
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    shared_features = [
        c for c in FEATURE_COLS if c in reference_df.columns and c in current_df.columns
    ]
    if not shared_features:
        raise ValueError(
            f"None of the 11 ENGINEERED_FEATURES were found in both DataFrames: {FEATURE_COLS}"
        )

    html_path = os.path.join(output_dir, f"drift_report_{tag}.html")
    json_path = os.path.join(output_dir, f"drift_report_{tag}.json")

    Report, DataDriftPreset, _, ColumnMapping = _load_evidently_components()
    if Report is not None and DataDriftPreset is not None:
        has_target = TARGET_COL in reference_df.columns and TARGET_COL in current_df.columns
        cols = shared_features + ([TARGET_COL] if has_target else [])
        mapping = None
        if ColumnMapping is not None:
            try:
                mapping = ColumnMapping(
                    target=TARGET_COL if has_target else None,
                    numerical_features=shared_features,
                )
            except Exception:
                mapping = None

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df[cols],
            current_data=current_df[cols],
            column_mapping=mapping,
        )
        report.save_html(html_path)
        report_dict = report.as_dict()
        engine = "evidently"
    else:
        report_dict, html_doc = _compute_ks_fallback_report(
            reference_df, current_df, shared_features
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        engine = "scipy_ks_fallback"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)

    drift_results = _extract_drift_summary(report_dict)
    drifted_count = sum(1 for v in drift_results["feature_drift"].values() if v)
    drift_results["engine"] = engine
    drift_results["evaluated_features"] = shared_features
    drift_results["total_features_checked"] = len(shared_features)
    drift_results["drifted_features_count"] = drifted_count
    drift_results["html_path"] = html_path
    drift_results["json_path"] = json_path

    feature_details: dict[str, dict] = {}
    for col in shared_features:
        ref_s = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        cur_s = pd.to_numeric(current_df[col], errors="coerce").dropna()
        r_mean = round(float(ref_s.mean()), 4) if len(ref_s) else 0.0
        c_mean = round(float(cur_s.mean()), 4) if len(cur_s) else 0.0
        score = float(drift_results.get("feature_drift_scores", {}).get(col, 0.0))
        if score == 0.0 and len(ref_s) and len(cur_s):
            score = round(float(ks_2samp(ref_s.to_numpy(), cur_s.to_numpy()).statistic), 6)
            drift_results.setdefault("feature_drift_scores", {})[col] = score
        feature_details[col] = {
            "drifted": bool(drift_results["feature_drift"].get(col, False)),
            "drift_score": round(score, 4),
            "ref_mean": r_mean,
            "cur_mean": c_mean,
            "delta": round(c_mean - r_mean, 4),
        }
    drift_results["feature_details"] = feature_details

    summary_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "engine": engine,
        "drift_share": drift_results["drift_share"],
        "dataset_drifted": drift_results["dataset_drifted"],
        "drifted_features_count": drifted_count,
        "total_features_checked": len(shared_features),
        "feature_drift": drift_results["feature_drift"],
        "feature_drift_scores": drift_results.get("feature_drift_scores", {}),
        "feature_details": feature_details,
        "html_path": html_path,
        "json_path": json_path,
        "model_drift": None,
    }
    drift_results["latest_summary_path"] = _write_latest_summary(
        output_dir, summary_payload, save_to_db=_save_to_db
    )

    print(f"[Monitoring] Engine: {engine} | Drift report written to {html_path}")
    print(
        f"[Monitoring] Drift share: {drift_results['drift_share']:.2%} "
        f"({len(shared_features)} engineered features checked)"
    )
    for feat, drifted in drift_results["feature_drift"].items():
        status = "DRIFTED" if drifted else "OK"
        print(f"  {feat:<25}: {status}")

    if log_to_mlflow:
        _log_drift_to_mlflow(tag, drift_results)

    return drift_results


def compute_classification_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = os.path.join("reports", "drift"),
    tag: Optional[str] = None,
    prediction_col: str = "prediction",
) -> Optional[str]:
    """Generate an Evidently classification quality report when Evidently is installed."""
    os.makedirs(output_dir, exist_ok=True)
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    Report, _, ClassificationPreset, ColumnMapping = _load_evidently_components()
    if Report is None or ClassificationPreset is None:
        print("[Monitoring] Evidently not installed; skipping HTML classification preset.")
        return None

    shared_features = [
        c for c in FEATURE_COLS if c in reference_df.columns and c in current_df.columns
    ]
    mapping = None
    if ColumnMapping is not None:
        try:
            mapping = ColumnMapping(
                target=TARGET_COL,
                prediction=prediction_col if prediction_col in reference_df.columns else None,
                numerical_features=shared_features,
            )
        except Exception:
            mapping = None

    try:
        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=mapping)

        html_path = os.path.join(output_dir, f"classification_report_{tag}.html")
        report.save_html(html_path)
        print(f"[Monitoring] Classification report written to {html_path}")
        return html_path
    except Exception as exc:
        print(f"[Monitoring] Warning: ClassificationPreset skipped ({exc})")
        return None


def _safe_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict:
    """Compute classification metrics safely even on small single-class manual windows."""
    if len(np.unique(y_true)) < 2:
        acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": 0.5,
            "pr_auc": float(np.mean(y_true)) if len(y_true) else 0.0,
            "r2_score": 0.0,
            "mcc": 0.0,
        }
    return classification_metrics(y_true, y_pred, y_score)


def evaluate_model_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model_or_path,
    output_dir: str = os.path.join("reports", "drift"),
    tag: Optional[str] = None,
    f1_drop_threshold: float = 0.05,
) -> dict:
    """Compare model quality on reference vs current batches to detect model/concept drift."""
    os.makedirs(output_dir, exist_ok=True)
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    model = joblib.load(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    shared_features = [
        c for c in FEATURE_COLS if c in reference_df.columns and c in current_df.columns
    ]
    if TARGET_COL not in reference_df.columns or TARGET_COL not in current_df.columns:
        raise ValueError(f"Both reference and current datasets must contain '{TARGET_COL}'.")

    ref_valid = reference_df.copy()
    cur_valid = current_df.copy()
    ref_valid[TARGET_COL] = pd.to_numeric(ref_valid[TARGET_COL], errors="coerce")
    cur_valid[TARGET_COL] = pd.to_numeric(cur_valid[TARGET_COL], errors="coerce")
    ref_valid = ref_valid[ref_valid[TARGET_COL].isin([0, 1])].reset_index(drop=True)
    cur_valid = cur_valid[cur_valid[TARGET_COL].isin([0, 1])].reset_index(drop=True)
    if len(ref_valid) == 0 or len(cur_valid) == 0:
        raise ValueError(
            "Both reference and current datasets must have at least one labeled binary target row."
        )

    X_ref = ref_valid[shared_features].astype(float)
    y_ref = ref_valid[TARGET_COL].astype(int).to_numpy()
    X_cur = cur_valid[shared_features].astype(float)
    y_cur = cur_valid[TARGET_COL].astype(int).to_numpy()

    ref_pred = model.predict(X_ref)
    ref_score = positive_scores(model, X_ref)
    cur_pred = model.predict(X_cur)
    cur_score = positive_scores(model, X_cur)

    ref_metrics = _safe_classification_metrics(y_ref, ref_pred, ref_score)
    cur_metrics = _safe_classification_metrics(y_cur, cur_pred, cur_score)

    supports_proba = bool(hasattr(model, "predict_proba"))
    ref_risk = model_risk_diagnostics(
        ref_valid, y_ref, ref_pred, ref_score, is_probability=supports_proba
    )
    cur_risk = model_risk_diagnostics(
        cur_valid, y_cur, cur_pred, cur_score, is_probability=supports_proba
    )

    ref_c0_rec = ref_risk["struggling_student_class0"]["recall_specificity"]
    cur_c0_rec = cur_risk["struggling_student_class0"]["recall_specificity"]

    f1_drop = round(float(ref_metrics["f1_score"] - cur_metrics["f1_score"]), 6)
    acc_drop = round(float(ref_metrics["accuracy"] - cur_metrics["accuracy"]), 6)
    c0_recall_drop = round(float(ref_c0_rec - cur_c0_rec), 6)

    model_drift_detected = bool(
        f1_drop >= f1_drop_threshold
        or acc_drop >= f1_drop_threshold
        or c0_recall_drop >= f1_drop_threshold
    )

    ref_with_pred = ref_valid.copy()
    cur_with_pred = cur_valid.copy()
    ref_with_pred["prediction"] = ref_pred
    cur_with_pred["prediction"] = cur_pred
    cls_html = compute_classification_report(
        ref_with_pred, cur_with_pred, output_dir=output_dir, tag=tag
    )

    result = {
        "model_drift_detected": model_drift_detected,
        "f1_drop_threshold": float(f1_drop_threshold),
        "f1_drop": f1_drop,
        "accuracy_drop": acc_drop,
        "class0_recall_drop": c0_recall_drop,
        "reference_metrics": {k: round(float(v), 6) for k, v in ref_metrics.items()},
        "current_metrics": {k: round(float(v), 6) for k, v in cur_metrics.items()},
        "reference_class0_recall": round(float(ref_c0_rec), 6),
        "current_class0_recall": round(float(cur_c0_rec), 6),
        "classification_html_path": cls_html,
    }

    print(
        f"[Monitoring] Model drift: {'DETECTED' if model_drift_detected else 'OK'} | "
        f"Ref F1={ref_metrics['f1_score']:.4f} -> Cur F1={cur_metrics['f1_score']:.4f} "
        f"(drop={f1_drop:+.4f}) | Class-0 Recall={ref_c0_rec:.4f} -> {cur_c0_rec:.4f}"
    )
    return result


def run_full_monitoring(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model_path: Optional[str] = None,
    output_dir: str = os.path.join("reports", "drift"),
    tag: Optional[str] = None,
    f1_drop_threshold: float = 0.05,
    log_to_mlflow: bool = False,
) -> dict:
    """Run data drift and (optionally) model performance drift together."""
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    will_eval_model = bool(model_path and os.path.exists(model_path))
    drift_results = compute_drift_report(
        reference_df,
        current_df,
        output_dir=output_dir,
        tag=tag,
        log_to_mlflow=False,
        _save_to_db=not will_eval_model,
    )

    model_drift = None
    if will_eval_model:
        try:
            model_drift = evaluate_model_drift(
                reference_df,
                current_df,
                model_path,
                output_dir=output_dir,
                tag=tag,
                f1_drop_threshold=f1_drop_threshold,
            )
        except ValueError as exc:
            print(f"[Monitoring] Notice: model drift check skipped ({exc})")
            model_drift = None
    if model_drift is not None:
        drift_results["model_drift"] = model_drift
        drift_results["classification_html_path"] = model_drift.get("classification_html_path")

        latest_path = drift_results["latest_summary_path"]
        with open(latest_path, "r", encoding="utf-8") as f:
            latest_payload = json.load(f)
        latest_payload["model_drift"] = model_drift
        _write_latest_summary(output_dir, latest_payload, save_to_db=True)
    elif will_eval_model:
        latest_path = drift_results["latest_summary_path"]
        with open(latest_path, "r", encoding="utf-8") as f:
            latest_payload = json.load(f)
        _write_latest_summary(output_dir, latest_payload, save_to_db=True)

    if log_to_mlflow:
        _log_drift_to_mlflow(tag, drift_results, model_drift=model_drift)

    # Milestone 5: Optional Discord / webhook paging on covariate or concept drift
    try:
        from src.alerts import send_alert_webhook

        is_data_drift = bool(drift_results.get("dataset_drifted", False))
        is_model_drift = bool((model_drift or {}).get("model_drift_detected", False))
        if is_data_drift or is_model_drift:
            sev = "critical" if is_model_drift else "warning"
            send_alert_webhook(
                title="Drift Detected on Student Performance Pipeline",
                message=(
                    f"Tag `{tag}` flagged covariate/model drift "
                    f"(drift_share={drift_results.get('drift_share', 0.0):.1%}, "
                    f"model_drift={is_model_drift})."
                ),
                severity=sev,
                fields={
                    "Drift Share": f"{drift_results.get('drift_share', 0.0):.1%}",
                    "Drifted Features": f"{drift_results.get('drifted_features_count', 0)}/11",
                    "Model Drift": str(is_model_drift),
                },
            )
    except Exception:
        pass

    return drift_results


def _synthetic_11_feature_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "log_opportunity": rng.uniform(0.69, 4.5, n),
            "attempt_count": rng.integers(1, 5, n).astype(float),
            "hint_count": rng.integers(0, 3, n).astype(float),
            "bottom_hint": rng.integers(0, 2, n).astype(float),
            "hint_ratio": rng.uniform(0.0, 1.0, n),
            "user_prev_acc": rng.uniform(0.3, 0.95, n),
            "problem_difficulty": rng.uniform(0.3, 0.9, n),
            "skill_difficulty": rng.uniform(0.4, 0.85, n),
            "is_skill_unmapped": rng.integers(0, 2, n).astype(float),
            "log_response_time": rng.uniform(7.5, 11.5, n),
            "is_rapid_guess": rng.integers(0, 2, n).astype(float),
            TARGET_COL: rng.integers(0, 2, n),
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 11-feature data & model drift detection.")
    parser.add_argument("--ref", required=True, help="Path to reference (training) CSV")
    parser.add_argument("--cur", required=True, help="Path to current (incoming) CSV")
    parser.add_argument("--model", default=None, help="Optional path to trained .pkl model")
    parser.add_argument(
        "--output", default=os.path.join("reports", "drift"), help="Output directory"
    )
    parser.add_argument("--tag", default=None, help="Label appended to output filenames")
    parser.add_argument(
        "--f1-drop-threshold",
        type=float,
        default=0.05,
        help="F1 / Class-0 recall drop threshold to flag model drift (default: 0.05)",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log drift artifacts to the student_performance_monitoring MLflow experiment",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Explicitly disable MLflow drift logging (default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if os.path.exists(args.ref) and os.path.exists(args.cur):
        ref_df = pd.read_csv(args.ref)
        cur_df = pd.read_csv(args.cur)
    else:
        print("[Monitoring] Reference/current CSVs missing; using synthetic 11-feature frames.")
        ref_df = _synthetic_11_feature_frame(200, seed=42)
        cur_df = _synthetic_11_feature_frame(100, seed=43)

    should_log = bool(args.log_mlflow and not args.no_mlflow)
    run_full_monitoring(
        ref_df,
        cur_df,
        model_path=args.model,
        output_dir=args.output,
        tag=args.tag,
        f1_drop_threshold=args.f1_drop_threshold,
        log_to_mlflow=should_log,
    )
