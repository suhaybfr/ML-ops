"""Monitoring module — data drift detection and prediction distribution checks.

Computes statistical drift across all 11 canonical ``ENGINEERED_FEATURES``
defined in ``src.features`` between the training reference distribution and
incoming prediction batches. Uses Evidently AI when installed, with a built-in
two-sample Kolmogorov-Smirnov (``scipy.stats.ks_2samp``) fallback so drift
reports (HTML + JSON) work cleanly across all Python versions.

Usage (standalone):
    python -m src.monitoring --ref data/processed/train.csv \
                             --cur data/processed/test.csv \
                             --no-mlflow
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features import ENGINEERED_FEATURES, TARGET_COL

# Canonical 11 engineered predictor features (kept as FEATURE_COLS for compatibility)
FEATURE_COLS = list(ENGINEERED_FEATURES)
MONITORING_EXPERIMENT = "student_performance_monitoring"


def _load_evidently_components():
    """Lazily resolve Evidently Report / Presets / ColumnMapping across 0.4–0.7+."""
    import pandas.core.dtypes.common as _pcommon

    if not hasattr(_pcommon, "infer_dtype_from_object"):
        try:
            from pandas._libs.lib import infer_dtype as _infer_dtype

            _pcommon.infer_dtype_from_object = lambda val: (_infer_dtype(val), None)
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
    dataset_drift_threshold: float = 0.5,
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
        f"<td>{'DRIFTED' if info['drift_detected'] else 'OK'}</td></tr>"
        for col, info in drift_by_columns.items()
    )
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Data Drift Report</title></head><body>"
        f"<h1>Data Drift Summary (drift share: {drift_share:.2%})</h1>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Status</th></tr>"
        f"{rows_html}</table></body></html>\n"
    )
    return report_dict, html_doc


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = os.path.join("reports", "drift"),
    tag: Optional[str] = None,
    log_to_mlflow: bool = False,
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
    else:
        report_dict, html_doc = _compute_ks_fallback_report(
            reference_df, current_df, shared_features
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_doc)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)

    drift_results = _extract_drift_summary(report_dict)
    drift_results["evaluated_features"] = shared_features
    drift_results["html_path"] = html_path
    drift_results["json_path"] = json_path

    print(f"[Monitoring] Drift report written to {html_path}")
    print(
        f"[Monitoring] Drift share: {drift_results['drift_share']:.2%} "
        f"({len(shared_features)} engineered features checked)"
    )
    for feat, drifted in drift_results["feature_drift"].items():
        status = "DRIFTED" if drifted else "OK"
        print(f"  {feat:<25}: {status}")

    # Log to dedicated monitoring experiment (never pollutes student_performance_baseline)
    if log_to_mlflow:
        try:
            import mlflow
            from src.utils import setup_mlflow

            setup_mlflow(experiment_name=MONITORING_EXPERIMENT)
            with mlflow.start_run(run_name=f"drift_check_{tag}"):
                mlflow.set_tags(
                    {
                        "phase": "phase_3_monitoring",
                        "kind": "data_drift",
                        "feature_set": "engineered_11",
                    }
                )
                mlflow.log_artifact(html_path, artifact_path="drift_reports")
                mlflow.log_artifact(json_path, artifact_path="drift_reports")
                mlflow.log_metrics(
                    {
                        "drift_share": float(drift_results.get("drift_share", 0.0)),
                        "dataset_drifted": 1.0
                        if drift_results.get("dataset_drifted", False)
                        else 0.0,
                    }
                )
                print(
                    f"[Monitoring] Drift artifacts logged to MLflow experiment "
                    f"{MONITORING_EXPERIMENT!r}."
                )
        except Exception as exc:
            print(f"[Monitoring] Notice: MLflow artifact logging skipped ({exc})")

    return drift_results


def _extract_drift_summary(report_dict: dict) -> dict:
    """Parse drift report dictionary into a compact summary."""
    summary: dict = {"drift_share": 0.0, "dataset_drifted": False, "feature_drift": {}}
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
    except Exception as exc:
        print(f"[Monitoring] Warning: could not parse drift summary: {exc}")
    return summary


def compute_classification_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = os.path.join("reports", "drift"),
    tag: Optional[str] = None,
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
            mapping = ColumnMapping(target=TARGET_COL, numerical_features=shared_features)
        except Exception:
            mapping = None

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=mapping)

    html_path = os.path.join(output_dir, f"classification_report_{tag}.html")
    report.save_html(html_path)
    print(f"[Monitoring] Classification report written to {html_path}")
    return html_path


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
    parser = argparse.ArgumentParser(description="Run 11-feature data drift detection.")
    parser.add_argument("--ref", required=True, help="Path to reference (training) CSV")
    parser.add_argument("--cur", required=True, help="Path to current (incoming) CSV")
    parser.add_argument(
        "--output", default=os.path.join("reports", "drift"), help="Output directory"
    )
    parser.add_argument("--tag", default=None, help="Label appended to output filenames")
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
    compute_drift_report(
        ref_df,
        cur_df,
        output_dir=args.output,
        tag=args.tag,
        log_to_mlflow=should_log,
    )
