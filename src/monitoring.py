"""
Monitoring module — data drift detection and prediction logging.

Uses Evidently AI to compute statistical drift between the training
reference distribution and incoming prediction batches.  Results are
written to ``reports/drift/`` as HTML + JSON so they can be viewed in
a browser or parsed by downstream alerting scripts.

Usage (standalone):
    python src/monitoring.py --ref data/processed/train.csv \
                             --cur data/processed/test.csv
"""

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import pandas as pd

# Compatibility patch: evidently imports infer_dtype_from_object which was removed in pandas >= 2.2
import pandas.core.dtypes.common as _pcommon
if not hasattr(_pcommon, 'infer_dtype_from_object'):
    try:
        from pandas._libs.lib import infer_dtype as _infer_dtype
        _pcommon.infer_dtype_from_object = lambda val: (_infer_dtype(val), None)
    except Exception:
        pass

# Evidently v0.4+ API
try:
    from evidently import ColumnMapping
except ImportError:
    try:
        from evidently.pipeline.column_mapping import ColumnMapping
    except ImportError:
        try:
            from evidently.legacy.pipeline.column_mapping import ColumnMapping
        except ImportError:
            ColumnMapping = None

from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.report import Report

FEATURE_COLS = [
    "skill_id",
    "attempt_count",
    "ms_first_response",
    "hint_count",
    "overlap_time",
]
TARGET_COL = "correct"

# ---------------------------------------------------------------------------
# Core drift report
# ---------------------------------------------------------------------------


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = os.path.join("reports", "drift"),
    tag: Optional[str] = None,
    log_to_mlflow: bool = True,
) -> dict:
    """
    Generate an Evidently data-drift report comparing reference vs current data.

    Parameters
    ----------
    reference_df : Training / baseline dataset with feature columns.
    current_df   : Current / incoming batch to check for drift.
    output_dir   : Directory where HTML and JSON reports are written.
    tag          : Optional label appended to output filenames (e.g. a date).

    Returns
    -------
    dict with ``drift_share`` (fraction of drifted features) and per-feature
    drift flags.
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = tag or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Keep only shared feature columns
    shared_features = [c for c in FEATURE_COLS if c in reference_df.columns and c in current_df.columns]

    mapping = None
    if ColumnMapping is not None:
        try:
            mapping = ColumnMapping(
                target=TARGET_COL if TARGET_COL in reference_df.columns else None,
                numerical_features=shared_features,
            )
        except Exception:
            mapping = None

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[shared_features + ([TARGET_COL] if TARGET_COL in reference_df.columns else [])],
        current_data=current_df[shared_features + ([TARGET_COL] if TARGET_COL in current_df.columns else [])],
        column_mapping=mapping,
    )

    html_path = os.path.join(output_dir, f"drift_report_{tag}.html")
    json_path = os.path.join(output_dir, f"drift_report_{tag}.json")

    report.save_html(html_path)

    report_dict = report.as_dict()
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)

    # Extract summary
    drift_results = _extract_drift_summary(report_dict)
    print(f"[Monitoring] Drift report written to {html_path}")
    print(f"[Monitoring] Drift share: {drift_results['drift_share']:.2%}")
    for feat, drifted in drift_results["feature_drift"].items():
        status = "⚠️  DRIFTED" if drifted else "✅ OK"
        print(f"  {feat:<25}: {status}")

    drift_results["html_path"] = html_path
    drift_results["json_path"] = json_path

    # Log drift reports directly as MLflow artifacts
    if log_to_mlflow:
        try:
            import mlflow
            from utils import setup_mlflow
            setup_mlflow(experiment_name="student_performance_baseline")
            with mlflow.start_run(run_name=f"drift_check_{tag}"):
                mlflow.log_artifact(html_path, artifact_path="drift_reports")
                mlflow.log_artifact(json_path, artifact_path="drift_reports")
                mlflow.log_metrics({
                    "drift_share": float(drift_results.get("drift_share", 0.0)),
                    "dataset_drifted": 1.0 if drift_results.get("dataset_drifted", False) else 0.0,
                })
                print("[Monitoring] Drift report artifacts logged to MLflow successfully!")
        except Exception as exc:
            print(f"[Monitoring] Notice: MLflow artifact logging skipped ({exc})")

    return drift_results


def _extract_drift_summary(report_dict: dict) -> dict:
    """Parse Evidently report dict into a compact drift summary."""
    summary = {"drift_share": 0.0, "dataset_drifted": False, "feature_drift": {}}
    try:
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            # Dataset-level drift
            if "share_of_drifted_columns" in result:
                summary["drift_share"] = result["share_of_drifted_columns"]
                summary["dataset_drifted"] = result.get("dataset_drift", False)
            # Per-feature drift
            drift_by_col = result.get("drift_by_columns", {})
            for col, col_result in drift_by_col.items():
                summary["feature_drift"][col] = col_result.get("drift_detected", False)
    except Exception as exc:
        print(f"[Monitoring] Warning: could not parse drift summary: {exc}")
    return summary


# ---------------------------------------------------------------------------
# Classification performance report (requires ground-truth labels)
# ---------------------------------------------------------------------------


def compute_classification_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str = os.path.join("reports", "drift"),
    tag: Optional[str] = None,
) -> None:
    """Generate an Evidently classification quality report."""
    os.makedirs(output_dir, exist_ok=True)
    tag = tag or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    shared_features = [c for c in FEATURE_COLS if c in reference_df.columns and c in current_df.columns]
    mapping = None
    if ColumnMapping is not None:
        try:
            mapping = ColumnMapping(
                target=TARGET_COL,
                numerical_features=shared_features,
            )
        except Exception:
            mapping = None

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=mapping,
    )

    html_path = os.path.join(output_dir, f"classification_report_{tag}.html")
    report.save_html(html_path)
    print(f"[Monitoring] Classification report written to {html_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Evidently drift detection.")
    parser.add_argument("--ref", required=True, help="Path to reference (training) CSV")
    parser.add_argument("--cur", required=True, help="Path to current (incoming) CSV")
    parser.add_argument("--output", default=os.path.join("reports", "drift"), help="Output directory")
    parser.add_argument("--tag", default=None, help="Label appended to output filenames")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip logging drift artifacts to MLflow")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ref = pd.read_csv(args.ref)
    cur = pd.read_csv(args.cur)
    compute_drift_report(ref, cur, output_dir=args.output, tag=args.tag, log_to_mlflow=not args.no_mlflow)
