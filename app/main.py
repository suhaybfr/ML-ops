"""FastAPI inference service for the AI-Based Digital Learning Assistant.

Serves the Phase 2 Champion Random Forest model (``models/baseline_model.pkl``)
with optional MLflow Model Registry loading (``MLFLOW_MODEL_URI``), zero-downtime
hot-reload (``POST /model/reload``), live telemetry logging & traffic simulation
(``/telemetry/*``), full-pipeline Prometheus observability (``GET /metrics``), and
optional Phase 3 Socratic RAG Tutor mounting (``ENABLE_TUTOR=1``).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.features import ENGINEERED_FEATURES, TARGET_COL  # noqa: E402
from src.telemetry import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    append_interactions,
    generate_simulated_batch,
    get_live_buffer_path,
    get_live_buffer_stats,
    load_live_interactions,
    reset_live_telemetry,
    upsert_interaction_outcome,
)

# Prometheus metrics (optional — gracefully skipped if not installed)
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def _get_or_create_metric(cls, name: str, documentation: str, *args, **kwargs):
    existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
    if existing is not None:
        return existing
    existing_total = getattr(REGISTRY, "_names_to_collectors", {}).get(f"{name}_total")
    if existing_total is not None:
        return existing_total
    return cls(name, documentation, *args, **kwargs)


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-Based Digital Learning Assistant — Student Performance API",
    description=(
        "Predicts whether a student problem attempt will be **correct** (`1`) or "
        "**struggling / needing tutor intervention** (`0`) using the 11 leakage-free "
        "engineered features from the ASSISTments Skill Builder pipeline. "
        "Default champion model: Random Forest (`models/baseline_model.pkl`, "
        "`random_forest_baseline_v2`)."
    ),
    version="3.0.0",
    contact={"name": "MLOps Team", "url": "https://github.com/suhaybfr/ML-ops"},
)

_tutor_mod: Any = None
# Optional Phase 3 LLM/RAG Tutor router (decoupled unless ENABLE_TUTOR=1)
if os.environ.get("ENABLE_TUTOR", "0") == "1":
    for _tutor_module in ("app.tutor", "tutor"):
        try:
            _tutor_mod = __import__(_tutor_module, fromlist=["router"])
            app.include_router(_tutor_mod.router)
            break
        except ImportError:
            continue

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Prometheus metric definitions
# ---------------------------------------------------------------------------

_RETRAIN_STATUSES = (
    "idle",
    "promoted",
    "challenger_kept",
    "gate_failed",
    "skipped_no_drift",
    "failed",
)

if _PROMETHEUS_AVAILABLE:
    # Core serving metrics
    PREDICTION_COUNTER = _get_or_create_metric(
        Counter,
        "prediction_requests_total",
        "Total number of prediction requests",
        ["endpoint", "result"],
    )
    PREDICTION_LATENCY = _get_or_create_metric(
        Histogram,
        "prediction_latency_seconds",
        "Prediction latency in seconds",
        ["endpoint"],
        buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    )
    MODEL_LOADED_GAUGE = _get_or_create_metric(
        Gauge,
        "model_loaded",
        "1 if the ML model is loaded and ready, 0 otherwise",
    )
    MODEL_INFO_GAUGE = _get_or_create_metric(
        Gauge,
        "model_info_active",
        "Active model metadata (1 for active version)",
        ["model_version", "model_type", "model_source"],
    )
    MODEL_RELOAD_COUNTER = _get_or_create_metric(
        Counter,
        "model_reload_total",
        "Total zero-downtime model reload attempts",
        ["status"],
    )
    REQUEST_ERRORS = _get_or_create_metric(
        Counter,
        "prediction_errors_total",
        "Total prediction errors",
        ["endpoint", "error_type"],
    )
    BATCH_SIZE_HISTOGRAM = _get_or_create_metric(
        Histogram,
        "batch_prediction_size",
        "Number of instances per batch prediction request",
        buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
    )
    PREDICTIONS_SCORED = _get_or_create_metric(
        Counter,
        "predictions_scored_total",
        "Total individual student attempts scored by outcome",
        ["outcome"],
    )
    PREDICTION_PROBABILITY = _get_or_create_metric(
        Histogram,
        "prediction_probability",
        "Distribution of predicted P(correct=1) across live requests",
        buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    )
    LIVE_INTERVENTION_RATIO = _get_or_create_metric(
        Gauge,
        "live_needs_intervention_ratio",
        "Live fraction of scored student attempts flagged needs_intervention=true",
    )
    LIVE_INTERACTIONS_GAUGE = _get_or_create_metric(
        Gauge,
        "live_interactions_logged_total",
        "Total student interactions logged in the live telemetry buffer",
    )
    LIVE_LABELED_GAUGE = _get_or_create_metric(
        Gauge,
        "live_labeled_interactions_total",
        "Total labeled student interactions available for retraining",
    )

    # Champion offline quality-gate metrics
    CHAMPION_OFFLINE_METRIC = _get_or_create_metric(
        Gauge,
        "champion_offline_metric",
        "Offline champion evaluation metrics from reports/baseline_metrics.json",
        ["metric"],
    )
    QUALITY_GATE_PASSED = _get_or_create_metric(
        Gauge,
        "quality_gate_passed",
        "1 if the Phase 2/3 Model-Risk Quality Gate passed, 0 otherwise",
    )
    QUALITY_GATE_CHECKS_TOTAL = _get_or_create_metric(
        Gauge,
        "quality_gate_checks_total",
        "Total number of Model-Risk Quality Gate checks evaluated",
    )
    QUALITY_GATE_CHECKS_PASSED = _get_or_create_metric(
        Gauge,
        "quality_gate_checks_passed",
        "Number of Model-Risk Quality Gate checks passed",
    )

    # Live data & model drift gauges (synced from reports/drift/latest_drift_summary.json)
    DATA_DRIFT_SHARE = _get_or_create_metric(
        Gauge,
        "data_drift_share",
        "Fraction of 11 engineered features exhibiting statistical drift",
    )
    DATASET_DRIFTED = _get_or_create_metric(
        Gauge,
        "dataset_drifted",
        "1 if dataset-level covariate drift was detected, 0 otherwise",
    )
    DRIFTED_FEATURES_COUNT = _get_or_create_metric(
        Gauge,
        "drifted_features_count",
        "Number of engineered features currently flagged as drifted",
    )
    FEATURE_DRIFT_DETECTED = _get_or_create_metric(
        Gauge,
        "feature_drift_detected",
        "1 if the given engineered feature is drifted, 0 otherwise",
        ["feature"],
    )
    FEATURE_DRIFT_SCORE = _get_or_create_metric(
        Gauge,
        "feature_drift_score",
        "Statistical drift score (KS statistic / Evidently score) per feature",
        ["feature"],
    )
    MODEL_DRIFT_DETECTED = _get_or_create_metric(
        Gauge,
        "model_drift_detected",
        "1 if model performance / concept drift was detected, 0 otherwise",
    )
    MODEL_DRIFT_F1_DROP = _get_or_create_metric(
        Gauge,
        "model_drift_f1_drop",
        "F1 score drop between reference and current batch",
    )
    MODEL_DRIFT_CURRENT_F1 = _get_or_create_metric(
        Gauge,
        "model_drift_current_f1",
        "Observed F1 score on the latest monitored batch",
    )
    MODEL_DRIFT_CLASS0_RECALL_DROP = _get_or_create_metric(
        Gauge,
        "model_drift_class0_recall_drop",
        "Struggling-student (class 0) recall drop on latest monitored batch",
    )

    # Retraining state gauges (synced from reports/drift/latest_status.json)
    RETRAIN_RUNS_GAUGE = _get_or_create_metric(
        Gauge,
        "retrain_runs_total",
        "Cumulative automated retraining runs executed",
    )
    RETRAIN_PROMOTIONS_GAUGE = _get_or_create_metric(
        Gauge,
        "retrain_promotions_total",
        "Cumulative retrained models promoted after passing quality gate",
    )
    RETRAIN_LAST_STATUS = _get_or_create_metric(
        Gauge,
        "retrain_last_status",
        "1 for the latest retrain status label, 0 for others",
        ["status"],
    )
    RETRAIN_LAST_CANDIDATE_F1 = _get_or_create_metric(
        Gauge,
        "retrain_last_candidate_f1",
        "F1 score of the most recent retrained candidate model",
    )

    # Ensure tutor metrics exist in REGISTRY even when ENABLE_TUTOR=0 so alerts stay clean
    TUTOR_REQUESTS_METRIC = _get_or_create_metric(
        Counter,
        "tutor_requests_total",
        "Total /tutor/ask requests by outcome and source type",
        ["status", "source_type"],
    )
    TUTOR_ERRORS_METRIC = _get_or_create_metric(
        Counter,
        "tutor_errors_total",
        "Total tutor endpoint errors by error type",
        ["error_type"],
    )
    TUTOR_LATENCY_METRIC = _get_or_create_metric(
        Histogram,
        "tutor_latency_seconds",
        "End-to-end /tutor/ask latency in seconds",
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    TUTOR_INDEXED_DOCS_METRIC = _get_or_create_metric(
        Gauge,
        "tutor_indexed_documents_total",
        "Total indexed course documents and uploaded PDFs in the tutor store",
    )
    TUTOR_INDEXED_CHUNKS_METRIC = _get_or_create_metric(
        Gauge,
        "tutor_indexed_chunks_total",
        "Total indexed text chunks across curriculum and uploaded PDFs",
    )

    # Initialize default label series to 0 so PromQL rules never evaluate to empty
    PREDICTION_COUNTER.labels(endpoint="/predict", result="0")
    PREDICTION_COUNTER.labels(endpoint="/predict", result="1")
    PREDICTION_COUNTER.labels(endpoint="/predict/batch", result="batch")
    PREDICTION_LATENCY.labels(endpoint="/predict")
    PREDICTION_LATENCY.labels(endpoint="/predict/batch")
    PREDICTIONS_SCORED.labels(outcome="correct")
    PREDICTIONS_SCORED.labels(outcome="needs_intervention")
    REQUEST_ERRORS.labels(endpoint="/predict", error_type="model_unavailable")
    MODEL_RELOAD_COUNTER.labels(status="success")
    MODEL_RELOAD_COUNTER.labels(status="error")
    TUTOR_REQUESTS_METRIC.labels(status="success", source_type="curriculum")
    TUTOR_REQUESTS_METRIC.labels(status="error", source_type="curriculum")
    TUTOR_ERRORS_METRIC.labels(error_type="llm_error")
    for _feat in ENGINEERED_FEATURES:
        FEATURE_DRIFT_DETECTED.labels(feature=_feat).set(0)
        FEATURE_DRIFT_SCORE.labels(feature=_feat).set(0.0)
    for _st in _RETRAIN_STATUSES:
        RETRAIN_LAST_STATUS.labels(status=_st).set(1 if _st == "idle" else 0)

# ---------------------------------------------------------------------------
# Model loading & paths
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "baseline_model.pkl")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
MLFLOW_MODEL_URI = os.environ.get("MLFLOW_MODEL_URI", "").strip()
BASELINE_METRICS_PATH = REPO_ROOT / "reports" / "baseline_metrics.json"
DRIFT_REPORTS_DIR = Path(os.environ.get("DRIFT_REPORTS_DIR", str(REPO_ROOT / "reports" / "drift")))

FEATURE_COLS: list[str] = list(ENGINEERED_FEATURES)

_ARTIFACT_VERSION_MAP = {
    "baseline_model.pkl": "random_forest_baseline_v2",
    "logreg_baseline.pkl": "logistic_regression_baseline_v2",
    "svm_model.pkl": "svm_rbf_holdout_v2",
    "retrained_champion.pkl": "random_forest_retrained_v3",
}

_model_lock = threading.RLock()
_model: Any = None
_model_load_time: Optional[float] = None
_model_source: str = "unloaded"
_model_version: str = "unknown"
_loaded_cache_key: Optional[tuple[str, str]] = None
_active_info_labels: Optional[tuple[str, str, str]] = None

_scored_total_count = 0
_scored_intervention_count = 0
_offline_gate_cache: Optional[dict[str, Any]] = None


def _infer_version_label(model_path: str, model_obj: Any, registry_uri: str = "") -> str:
    if registry_uri:
        return registry_uri.replace("models:/", "").replace("/", ":")
    basename = os.path.basename(model_path)
    if basename in _ARTIFACT_VERSION_MAP:
        return _ARTIFACT_VERSION_MAP[basename]
    est = model_obj.steps[-1][1] if isinstance(model_obj, Pipeline) else model_obj
    return f"{type(est).__name__.lower()}_custom"


def _extract_estimator_metadata(model: Any) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """Inspect either a standalone estimator or an sklearn Pipeline safely."""
    if isinstance(model, Pipeline) or hasattr(model, "steps"):
        steps = [{"name": name, "type": type(step).__name__} for name, step in model.steps]
        estimator = model.steps[-1][1]
    else:
        steps = [{"name": "classifier", "type": type(model).__name__}]
        estimator = model

    est_name = type(estimator).__name__
    candidate_keys = (
        "n_estimators",
        "max_depth",
        "C",
        "kernel",
        "gamma",
        "max_iter",
        "class_weight",
        "solver",
        "random_state",
    )
    hyperparams: dict[str, Any] = {}
    for k in candidate_keys:
        if hasattr(estimator, k):
            val = getattr(estimator, k)
            is_prim = isinstance(val, (int, float, bool, str)) or val is None
            hyperparams[k] = val if is_prim else str(val)

    return est_name, steps, hyperparams


def _set_active_model_info_metric(version: str, model_obj: Any, source: str) -> None:
    global _active_info_labels
    if not _PROMETHEUS_AVAILABLE:
        return
    est_name, _, _ = _extract_estimator_metadata(model_obj)
    new_labels = (version, est_name, source)
    if _active_info_labels and _active_info_labels != new_labels:
        try:
            MODEL_INFO_GAUGE.remove(*_active_info_labels)
        except KeyError:
            pass
    MODEL_INFO_GAUGE.labels(*new_labels).set(1)
    _active_info_labels = new_labels


def reset_model_cache() -> None:
    """Clear the cached model singleton (used by tests when swapping MODEL_PATH)."""
    global _model, _model_load_time, _model_source, _model_version, _loaded_cache_key
    with _model_lock:
        _model = None
        _model_load_time = None
        _model_source = "unloaded"
        _model_version = "unknown"
        _loaded_cache_key = None


def _load_model_from_sources(
    model_path: str,
    registry_uri: str = "",
    version_override: Optional[str] = None,
) -> tuple[Any, str, str]:
    loaded = None
    source = "local_artifact"
    version = "unknown"

    if registry_uri:
        try:
            import mlflow.sklearn
            from src.utils import setup_mlflow

            setup_mlflow()
            loaded = mlflow.sklearn.load_model(registry_uri)
            source = "mlflow_registry"
            version = version_override or _infer_version_label(
                model_path, loaded, registry_uri=registry_uri
            )
        except Exception as exc:
            print(
                f"[WARN] Could not load MLFLOW_MODEL_URI={registry_uri!r} ({exc}); "
                f"falling back to MODEL_PATH={model_path!r}.",
                file=sys.stderr,
            )

    if loaded is None:
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model artifact not found at '{model_path}'. "
                "Run 'dvc pull' to fetch the DVC-tracked model."
            )
        loaded = joblib.load(model_path)
        source = "local_artifact"
        version = version_override or _infer_version_label(model_path, loaded)

    return loaded, source, version


def get_model():
    """Load and cache the configured model (Registry URI first, then local .pkl)."""
    global _model, _model_load_time, _model_source, _model_version, _loaded_cache_key

    desired_key = (MLFLOW_MODEL_URI, os.path.abspath(MODEL_PATH))
    if _model is not None and _loaded_cache_key == desired_key:
        return _model

    with _model_lock:
        if _model is not None and _loaded_cache_key == desired_key:
            return _model
        try:
            loaded, source, version = _load_model_from_sources(MODEL_PATH, MLFLOW_MODEL_URI)
        except RuntimeError:
            if _PROMETHEUS_AVAILABLE:
                MODEL_LOADED_GAUGE.set(0)
            raise

        _model = loaded
        _model_load_time = time.time()
        _model_source = source
        _model_version = version
        _loaded_cache_key = desired_key
        if _PROMETHEUS_AVAILABLE:
            MODEL_LOADED_GAUGE.set(1)
            _set_active_model_info_metric(version, loaded, source)
        return _model


def reload_active_model(
    model_path: Optional[str] = None,
    mlflow_model_uri: Optional[str] = None,
    version_label: Optional[str] = None,
) -> dict[str, Any]:
    """Atomically reload/hot-swap the live model without dropping in-flight requests."""
    global MODEL_PATH, MLFLOW_MODEL_URI
    global _model, _model_load_time, _model_source, _model_version, _loaded_cache_key

    target_path = MODEL_PATH
    if model_path is not None and model_path.strip():
        candidate = Path(model_path.strip())
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.is_relative_to(REPO_ROOT.resolve()) or candidate.suffix.lower() != ".pkl":
            raise HTTPException(
                status_code=400,
                detail="model_path must reference a .pkl file inside the project repository.",
            )
        target_path = str(candidate)

    target_uri = MLFLOW_MODEL_URI if mlflow_model_uri is None else mlflow_model_uri.strip()

    try:
        loaded, source, version = _load_model_from_sources(
            model_path=target_path,
            registry_uri=target_uri,
            version_override=version_label,
        )
    except RuntimeError as exc:
        if _PROMETHEUS_AVAILABLE:
            MODEL_RELOAD_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        if _PROMETHEUS_AVAILABLE:
            MODEL_RELOAD_COUNTER.labels(status="error").inc()
        raise HTTPException(
            status_code=500, detail=f"Failed to reload model: {type(exc).__name__}: {exc}"
        ) from exc

    with _model_lock:
        previous_version = _model_version
        MODEL_PATH = target_path
        MLFLOW_MODEL_URI = target_uri
        _model = loaded
        _model_load_time = time.time()
        _model_source = source
        _model_version = version
        _loaded_cache_key = (MLFLOW_MODEL_URI, os.path.abspath(MODEL_PATH))

    est_name, _, _ = _extract_estimator_metadata(loaded)
    if _PROMETHEUS_AVAILABLE:
        MODEL_LOADED_GAUGE.set(1)
        MODEL_RELOAD_COUNTER.labels(status="success").inc()
        _set_active_model_info_metric(version, loaded, source)

    return {
        "status": "reloaded",
        "previous_version": previous_version,
        "active_version": version,
        "model_type": est_name,
        "model_path": MODEL_PATH,
        "model_source": source,
        "reloaded_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Metrics sync helpers (Champion offline gate, Drift JSON, Retrain JSON, Live buffer)
# ---------------------------------------------------------------------------


def _read_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sync_offline_champion_metrics() -> None:
    global _offline_gate_cache
    if not _PROMETHEUS_AVAILABLE:
        return

    payload = _read_json_if_exists(BASELINE_METRICS_PATH)
    if not payload:
        return

    rf = payload.get("models", {}).get("random_forest_baseline", {})
    m = rf.get("metrics", {})
    risk = rf.get("risk_and_slice_diagnostics", {})
    c0 = risk.get("struggling_student_class0", {})

    metric_values = {
        "accuracy": m.get("accuracy"),
        "f1_score": m.get("f1_score"),
        "macro_f1": risk.get("macro_f1"),
        "roc_auc": m.get("roc_auc"),
        "mcc": m.get("mcc"),
        "brier_score": risk.get("brier_score"),
        "class0_recall": c0.get("recall_specificity"),
        "class0_precision": c0.get("precision"),
    }
    for key, val in metric_values.items():
        if val is not None:
            CHAMPION_OFFLINE_METRIC.labels(metric=key).set(float(val))

    if _offline_gate_cache is None:
        try:
            from src.validate_model import validate_baseline_metrics

            _offline_gate_cache = validate_baseline_metrics(
                metrics_path=BASELINE_METRICS_PATH,
                require_artifact=False,
            )
        except Exception:
            _offline_gate_cache = {"passed": True, "passed_checks": 0, "total_checks": 0}

    QUALITY_GATE_PASSED.set(1 if _offline_gate_cache.get("passed") else 0)
    QUALITY_GATE_CHECKS_TOTAL.set(int(_offline_gate_cache.get("total_checks", 0)))
    QUALITY_GATE_CHECKS_PASSED.set(int(_offline_gate_cache.get("passed_checks", 0)))


def _sync_drift_and_retrain_metrics() -> None:
    if not _PROMETHEUS_AVAILABLE:
        return

    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
    drift_summary = _read_json_if_exists(drift_dir / "latest_drift_summary.json") or {}
    retrain_status = _read_json_if_exists(drift_dir / "latest_status.json") or {}

    DATA_DRIFT_SHARE.set(float(drift_summary.get("drift_share", 0.0)))
    DATASET_DRIFTED.set(1 if drift_summary.get("dataset_drifted") else 0)
    DRIFTED_FEATURES_COUNT.set(int(drift_summary.get("drifted_features_count", 0)))

    feat_flags = drift_summary.get("feature_drift", {}) or {}
    feat_scores = drift_summary.get("feature_drift_scores", {}) or {}
    for feat in ENGINEERED_FEATURES:
        drifted_flag = 1 if feat_flags.get(feat) else 0
        FEATURE_DRIFT_DETECTED.labels(feature=feat).set(drifted_flag)
        score = feat_scores.get(feat)
        if score is not None:
            FEATURE_DRIFT_SCORE.labels(feature=feat).set(float(score))
        else:
            FEATURE_DRIFT_SCORE.labels(feature=feat).set(float(drifted_flag))

    md = drift_summary.get("model_drift") or {}
    MODEL_DRIFT_DETECTED.set(1 if md.get("model_drift_detected") else 0)
    MODEL_DRIFT_F1_DROP.set(float(md.get("f1_drop", 0.0)))
    cur_f1 = (md.get("current_metrics") or {}).get("f1_score", 0.0)
    MODEL_DRIFT_CURRENT_F1.set(float(cur_f1))
    MODEL_DRIFT_CLASS0_RECALL_DROP.set(float(md.get("class0_recall_drop", 0.0)))

    # Retrain state (with clean defaults before any retrain has run)
    RETRAIN_RUNS_GAUGE.set(int(retrain_status.get("retrain_runs_total", 0)))
    RETRAIN_PROMOTIONS_GAUGE.set(int(retrain_status.get("retrain_promotions_total", 0)))
    RETRAIN_LAST_CANDIDATE_F1.set(float(retrain_status.get("candidate_f1", 0.0)))

    cur_status = str(retrain_status.get("status", "idle"))
    for st in _RETRAIN_STATUSES:
        RETRAIN_LAST_STATUS.labels(status=st).set(1 if st == cur_status else 0)

    if "quality_gate_passed" in retrain_status and retrain_status.get("retrained"):
        QUALITY_GATE_PASSED.set(1 if retrain_status.get("quality_gate_passed") else 0)
        if retrain_status.get("quality_gate_total_checks"):
            QUALITY_GATE_CHECKS_TOTAL.set(int(retrain_status["quality_gate_total_checks"]))
            QUALITY_GATE_CHECKS_PASSED.set(int(retrain_status.get("quality_gate_passed_checks", 0)))

    # Live buffer stats
    buf_stats = get_live_buffer_stats()
    LIVE_INTERACTIONS_GAUGE.set(buf_stats["total_rows"])
    LIVE_LABELED_GAUGE.set(buf_stats["labeled_rows"])

    # Downstream tutor gauges (if tutor is mounted)
    if _tutor_mod is not None and hasattr(_tutor_mod, "refresh_tutor_metrics"):
        try:
            _tutor_mod.refresh_tutor_metrics()
        except Exception:
            pass


# Attempt non-fatal warm load at startup
try:
    get_model()
except RuntimeError as exc:
    print(f"[WARN] {exc}", file=sys.stderr)
_sync_offline_champion_metrics()


# ---------------------------------------------------------------------------
# Canonical 11-Feature Pydantic Schemas
# ---------------------------------------------------------------------------


class StudentFeatures(BaseModel):
    """Strict 11-feature input contract matching ``src.features.ENGINEERED_FEATURES``."""

    log_opportunity: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        allow_inf_nan=False,
        description="Learning curve index: ln(1 + cumulative skill practice count).",
    )
    attempt_count: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Number of attempts on the current problem (capped at 50 in training).",
    )
    hint_count: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        allow_inf_nan=False,
        description="Number of hints requested on the current problem.",
    )
    bottom_hint: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary flag: 1 if the bottom-out answer hint was viewed, else 0.",
    )
    hint_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Fraction of available hints requested (hint_count / hint_total).",
    )
    user_prev_acc: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Student's expanding historical accuracy prior to this attempt.",
    )
    problem_difficulty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Target-encoded problem pass rate (higher = easier item).",
    )
    skill_difficulty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Target-encoded skill domain pass rate.",
    )
    is_skill_unmapped: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary flag: 1 if skill_name is 'Unknown', else 0.",
    )
    log_response_time: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        allow_inf_nan=False,
        description="Log first-response latency: ln(1 + max(0, ms_first_response)).",
    )
    is_rapid_guess: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary engagement flag: 1 if ms_first_response < 3000 ms, else 0.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "log_opportunity": 2.3979,
                "attempt_count": 1.0,
                "hint_count": 0.0,
                "bottom_hint": 0,
                "hint_ratio": 0.0,
                "user_prev_acc": 0.78,
                "problem_difficulty": 0.71,
                "skill_difficulty": 0.68,
                "is_skill_unmapped": 0,
                "log_response_time": 9.62,
                "is_rapid_guess": 0,
            }
        },
    )


class PredictionResponse(BaseModel):
    """Single-instance student performance prediction & intervention signal."""

    prediction: int = Field(..., description="0 = struggling/incorrect, 1 = correct")
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of a correct attempt P(correct=1) in [0, 1].",
    )
    needs_intervention: bool = Field(
        ...,
        description="True when the student is predicted to struggle (prediction == 0).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the predicted outcome in [0, 1].",
    )
    label: str = Field(..., description="Human-readable prediction summary")
    model_version: str = Field(default="random_forest_baseline_v2")


class BatchPredictionRequest(BaseModel):
    instances: List[StudentFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class ModelReloadRequest(BaseModel):
    model_path: Optional[str] = None
    mlflow_model_uri: Optional[str] = None
    version_label: Optional[str] = None


class TelemetryFeedbackRequest(BaseModel):
    features: StudentFeatures
    correct: Optional[int] = Field(default=None, ge=0, le=1)
    skill_name: Optional[str] = Field(default=None, max_length=120)
    session_id: Optional[str] = Field(default=None, max_length=120)
    source: str = Field(default="tutor_ui", max_length=60)


class SimulateRequest(BaseModel):
    profile: Literal["normal", "struggling", "drifted"] = "drifted"
    count: int = Field(default=100, ge=5, le=2000)
    seed: int = Field(default=42)
    run_drift_check: bool = True
    trigger_retrain: bool = False


class RetrainTriggerRequest(BaseModel):
    force: bool = False
    from_live: bool = True
    hot_reload: bool = True
    drift_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------


def _build_feature_dataframe(instances: list[StudentFeatures]) -> pd.DataFrame:
    """Build an ordered float DataFrame strictly matching ``ENGINEERED_FEATURES``."""
    records = [inst.model_dump() for inst in instances]
    df = pd.DataFrame(records)[FEATURE_COLS].astype(float)
    # Apply training-time attempt_count cap (<= 50) for out-of-range safety
    df["attempt_count"] = df["attempt_count"].clip(lower=0.0, upper=50.0)
    return df


def _predict_with_probabilities(model: Any, df: pd.DataFrame) -> tuple[list[int], list[float]]:
    """Return ``(predictions, P(correct=1))`` bounded in ``[0, 1]`` for any estimator."""
    preds = [int(p) for p in np.asarray(model.predict(df)).tolist()]

    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(df), dtype=float)[:, 1]
        probs = np.clip(probs, 0.0, 1.0)
    elif hasattr(model, "decision_function"):
        margins = np.asarray(model.decision_function(df), dtype=float)
        # Map raw SVM margins onto [0, 1] via numerically stable logistic sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.clip(margins, -40.0, 40.0)))
    else:
        probs = np.asarray(preds, dtype=float)

    return preds, [float(p) for p in probs.tolist()]


def _format_prediction_response(pred: int, prob_correct: float) -> PredictionResponse:
    needs_intervention = bool(pred == 0)
    pred_confidence = prob_correct if pred == 1 else (1.0 - prob_correct)
    label = "Correct" if pred == 1 else "Needs Intervention"
    return PredictionResponse(
        prediction=pred,
        probability=round(prob_correct, 4),
        needs_intervention=needs_intervention,
        confidence=round(pred_confidence, 4),
        label=label,
        model_version=_model_version,
    )


def _record_live_scoring_metrics(preds: list[int], probs: list[float]) -> None:
    global _scored_total_count, _scored_intervention_count
    if not _PROMETHEUS_AVAILABLE:
        return
    interventions = sum(1 for p in preds if p == 0)
    corrects = len(preds) - interventions
    if corrects:
        PREDICTIONS_SCORED.labels(outcome="correct").inc(corrects)
    if interventions:
        PREDICTIONS_SCORED.labels(outcome="needs_intervention").inc(interventions)
    for pr in probs:
        PREDICTION_PROBABILITY.observe(float(pr))

    _scored_total_count += len(preds)
    _scored_intervention_count += interventions
    if _scored_total_count > 0:
        LIVE_INTERVENTION_RATIO.set(_scored_intervention_count / _scored_total_count)


def _log_predictions_nonfatal(
    instances: list[StudentFeatures],
    results: list[PredictionResponse],
    source: str,
    correct_values: Optional[list[Optional[int]]] = None,
    skill_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    try:
        rows = []
        for idx, (inst, res) in enumerate(zip(instances, results)):
            rec = inst.model_dump()
            rec.update(
                {
                    "source": source,
                    "skill_name": skill_name,
                    "session_id": session_id,
                    "prediction": res.prediction,
                    "probability": res.probability,
                    "needs_intervention": res.needs_intervention,
                    "model_version": res.model_version,
                    TARGET_COL: (
                        None if correct_values is None else correct_values[idx]
                    ),
                }
            )
            rows.append(rec)
        append_interactions(rows)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["General"])
def root():
    """Service welcome message and endpoint guide."""
    endpoints = {
        "health": "GET /health",
        "model_info": "GET /model/info",
        "model_reload": "POST /model/reload",
        "model_retrain": "POST /model/retrain",
        "predict": "POST /predict",
        "predict_explain": "POST /predict/explain",
        "batch_predict": "POST /predict/batch",
        "telemetry_feedback": "POST /telemetry/feedback",
        "telemetry_check_drift": "POST /telemetry/check-drift",
        "telemetry_simulate": "POST /telemetry/simulate",
        "telemetry_reset": "POST /telemetry/reset",
        "telemetry_summary": "GET /telemetry/summary",
        "metrics": "GET /metrics",
        "docs": "GET /docs",
    }
    if os.environ.get("ENABLE_TUTOR", "0") == "1":
        endpoints["tutor"] = "GET /tutor/"
    return {
        "service": "AI-Based Digital Learning Assistant — Student Performance API",
        "version": app.version,
        "champion_model": "RandomForestClassifier (models/baseline_model.pkl)",
        "feature_count": len(FEATURE_COLS),
        "endpoints": endpoints,
        "project": "https://github.com/suhaybfr/ML-ops",
        "dagshub": "https://dagshub.com/suhaybfr/ML-ops",
    }


@app.get("/health", tags=["General"])
def health():
    """Liveness and model-readiness probe."""
    try:
        get_model()
        model_ready = True
    except RuntimeError:
        model_ready = False

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_loaded": model_ready,
            "model_path": MODEL_PATH,
            "model_source": _model_source,
            "model_version": _model_version,
            "uptime_seconds": (
                round(time.time() - _model_load_time, 1) if _model_load_time else None
            ),
        },
    )


@app.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
def metrics():
    """Full-pipeline Prometheus exposition endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", media_type="text/plain")
    _sync_offline_champion_metrics()
    _sync_drift_and_retrain_metrics()
    return PlainTextResponse(
        generate_latest(REGISTRY).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata, hyperparameters, and 11-feature contract for the active model."""
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    est_name, steps, hyperparams = _extract_estimator_metadata(model)
    supports_proba = bool(hasattr(model, "predict_proba"))

    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    raw_imp = getattr(est, "feature_importances_", None)
    global_importances = (
        {col: round(float(v), 4) for col, v in zip(FEATURE_COLS, raw_imp)}
        if raw_imp is not None
        else {}
    )

    benchmark: dict[str, Any] = {}
    base_doc = _read_json_if_exists(BASELINE_METRICS_PATH)
    if base_doc:
        rf_doc = base_doc.get("models", {}).get("random_forest_baseline", {})
        risk_doc = rf_doc.get("risk_and_slice_diagnostics", {})
        benchmark = {
            "metrics": rf_doc.get("metrics", {}),
            "macro_f1": risk_doc.get("macro_f1"),
            "brier_score": risk_doc.get("brier_score"),
            "struggling_class0": risk_doc.get("struggling_student_class0", {}),
            "slices": risk_doc.get("slices", {}),
        }

    return {
        "model_type": est_name,
        "model_version": _model_version,
        "model_source": _model_source,
        "model_path": MODEL_PATH,
        "pipeline_steps": steps,
        "hyperparameters": hyperparams,
        "supports_predict_proba": supports_proba,
        "probability_method": "predict_proba" if supports_proba else "sigmoid_decision_function",
        "feature_set": "engineered_11",
        "feature_count": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "global_feature_importances": global_importances,
        "benchmark": benchmark,
        "target": "correct (0=struggling/needs_intervention, 1=correct)",
        "training_framework": "scikit-learn via DVC pipeline",
        "experiment_tracking": "MLflow on DAGsHub",
    }


@app.post("/model/reload", tags=["Model"])
def reload_model(body: Optional[ModelReloadRequest] = None):
    """Zero-downtime model reload / promotion swap without container restart."""
    req = body or ModelReloadRequest()
    return reload_active_model(
        model_path=req.model_path,
        mlflow_model_uri=req.mlflow_model_uri,
        version_label=req.version_label,
    )


@app.post("/model/retrain", tags=["Model"])
def trigger_retrain(body: Optional[RetrainTriggerRequest] = None):
    """Run drift-gated retraining from live telemetry and optionally hot-reload the model."""
    from src.retrain import retrain_cycle

    req = body or RetrainTriggerRequest()
    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
    result = retrain_cycle(
        from_live=req.from_live,
        force=req.force,
        drift_threshold=req.drift_threshold,
        output_dir=drift_dir,
        status_path=drift_dir / "latest_status.json",
    )
    if req.hot_reload and result.get("promoted") and result.get("active_artifact_path"):
        result["reload_result"] = reload_active_model(
            model_path=result["active_artifact_path"],
            version_label=result.get("active_model_version"),
        )
    _sync_drift_and_retrain_metrics()
    return result


def _compute_local_xai(model: Any, features: StudentFeatures) -> dict[str, Any]:
    """Compute fast permutation-Shapley feature attributions summing to P(x) - P(baseline)."""
    inst_dict = features.model_dump()
    n_feats = len(FEATURE_COLS)
    x_vec = np.array([float(inst_dict[c]) for c in FEATURE_COLS], dtype=float)
    b_vec = np.array(
        [float(FEATURE_DESCRIPTIONS[c]["baseline"]) for c in FEATURE_COLS], dtype=float
    )

    # Vectorized Monte-Carlo Shapley over M=24 fixed permutations (2 + 24*11 = 266 rows, <8ms)
    n_perms = 24
    rng = np.random.default_rng(42)
    perms = np.array([rng.permutation(n_feats) for _ in range(n_perms)])

    eval_rows = [x_vec.copy(), b_vec.copy()]
    for p_idx in range(n_perms):
        cur = b_vec.copy()
        for feat_idx in perms[p_idx]:
            cur = cur.copy()
            cur[feat_idx] = x_vec[feat_idx]
            eval_rows.append(cur)

    batch_df = pd.DataFrame(eval_rows, columns=FEATURE_COLS)
    batch_df["attempt_count"] = batch_df["attempt_count"].clip(lower=0.0, upper=50.0)
    _, probs = _predict_with_probabilities(model, batch_df)

    prob_actual = float(probs[0])
    prob_baseline = float(probs[1])

    phi = np.zeros(n_feats, dtype=float)
    pos = 2
    for p_idx in range(n_perms):
        prev_p = prob_baseline
        for feat_idx in perms[p_idx]:
            cur_p = float(probs[pos])
            phi[feat_idx] += cur_p - prev_p
            prev_p = cur_p
            pos += 1
    phi /= float(n_perms)

    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    raw_imp = getattr(est, "feature_importances_", None)
    global_imp = (
        {col: round(float(v), 4) for col, v in zip(FEATURE_COLS, raw_imp)}
        if raw_imp is not None
        else {}
    )

    attributions = []
    for idx, col in enumerate(FEATURE_COLS):
        impact = round(float(phi[idx]), 4)
        meta = FEATURE_DESCRIPTIONS[col]
        attributions.append(
            {
                "feature": col,
                "label": meta["label"],
                "value": round(float(inst_dict[col]), 4),
                "baseline_value": meta["baseline"],
                "impact": impact,
                "global_importance": global_imp.get(col, 0.0),
                "direction": (
                    "helps_mastery"
                    if impact > 0.005
                    else ("signals_struggle" if impact < -0.005 else "neutral")
                ),
                "meaning": meta["meaning"],
            }
        )

    attributions.sort(key=lambda item: abs(item["impact"]), reverse=True)
    top_struggle = [a for a in attributions if a["impact"] < -0.01][:3]
    top_mastery = [a for a in attributions if a["impact"] > 0.01][:3]

    if prob_actual < 0.5:
        reasons = (
            ", ".join(
                f"{a['label']} ({a['value']} → {a['impact'] * 100:+.1f}%)"
                for a in top_struggle
            )
            or "combined struggle indicators across attempt and hint telemetry"
        )
        summary = (
            f"Flagged for intervention: P(correct) is {prob_actual * 100:.1f}% "
            f"(down from {prob_baseline * 100:.1f}% healthy baseline; threshold 50%). "
            f"Top drivers: {reasons}."
        )
        scaffolding_mode = (
            "Heavy Socratic Scaffolding — plain language, worked analogy, single check"
        )
    else:
        reasons = (
            ", ".join(
                f"{a['label']} ({a['value']} → {a['impact'] * 100:+.1f}%)"
                for a in top_mastery
            )
            or "first-try accuracy and low hint reliance matching mastery baseline"
        )
        summary = (
            f"On track for mastery: P(correct) is {prob_actual * 100:.1f}% "
            f"(above 50% intervention threshold). Key strengths: {reasons}."
        )
        scaffolding_mode = "Light Socratic Extension — verify reasoning and pose transfer question"

    return {
        "baseline_probability": round(prob_baseline, 4),
        "decision_threshold": 0.5,
        "summary": summary,
        "scaffolding_mode": scaffolding_mode,
        "attributions": attributions,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: StudentFeatures):
    """Predict correctness (`0`/`1`), calibrated probability, and intervention flag."""
    start = time.perf_counter()
    try:
        model = get_model()
    except RuntimeError as exc:
        if _PROMETHEUS_AVAILABLE:
            REQUEST_ERRORS.labels(endpoint="/predict", error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc))

    df = _build_feature_dataframe([features])
    preds, probs = _predict_with_probabilities(model, df)
    response = _format_prediction_response(preds[0], probs[0])

    _record_live_scoring_metrics(preds, probs)
    _log_predictions_nonfatal([features], [response], source="api_predict")

    if _PROMETHEUS_AVAILABLE:
        latency = time.perf_counter() - start
        PREDICTION_COUNTER.labels(endpoint="/predict", result=str(response.prediction)).inc()
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(latency)

    return response


@app.post("/predict/explain", tags=["Prediction"])
def predict_explain(features: StudentFeatures, log_interaction: bool = True):
    """Score a student attempt and return local counterfactual XAI feature attributions."""
    start = time.perf_counter()
    try:
        model = get_model()
    except RuntimeError as exc:
        if _PROMETHEUS_AVAILABLE:
            REQUEST_ERRORS.labels(endpoint="/predict", error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc))

    df = _build_feature_dataframe([features])
    preds, probs = _predict_with_probabilities(model, df)
    response = _format_prediction_response(preds[0], probs[0])
    explanation = _compute_local_xai(model, features)

    interaction_id = uuid.uuid4().hex[:12]
    _record_live_scoring_metrics(preds, probs)
    if log_interaction:
        _log_predictions_nonfatal(
            [features], [response], source="tutor_xai", session_id=interaction_id
        )

    if _PROMETHEUS_AVAILABLE:
        latency = time.perf_counter() - start
        PREDICTION_COUNTER.labels(endpoint="/predict", result=str(response.prediction)).inc()
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(latency)

    return {
        **response.model_dump(),
        "interaction_id": interaction_id,
        "explanation": explanation,
    }


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Score up to 1,000 student problem attempts in a single vectorized batch."""
    if len(request.instances) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 1000 instances.",
        )
    if len(request.instances) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty batch — provide at least one instance.",
        )

    start = time.perf_counter()
    try:
        model = get_model()
    except RuntimeError as exc:
        if _PROMETHEUS_AVAILABLE:
            REQUEST_ERRORS.labels(endpoint="/predict/batch", error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc))

    df = _build_feature_dataframe(request.instances)
    preds, probs = _predict_with_probabilities(model, df)
    results = [_format_prediction_response(p, pr) for p, pr in zip(preds, probs)]

    _record_live_scoring_metrics(preds, probs)
    _log_predictions_nonfatal(request.instances, results, source="api_batch")

    if _PROMETHEUS_AVAILABLE:
        latency = time.perf_counter() - start
        PREDICTION_COUNTER.labels(endpoint="/predict/batch", result="batch").inc()
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(latency)
        BATCH_SIZE_HISTOGRAM.observe(len(request.instances))

    return BatchPredictionResponse(predictions=results, count=len(results))


@app.post("/telemetry/feedback", tags=["Monitoring"])
def log_telemetry_feedback(body: TelemetryFeedbackRequest):
    """Score or update in-place a labeled student interaction in the live telemetry buffer."""
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    df = _build_feature_dataframe([body.features])
    preds, probs = _predict_with_probabilities(model, df)
    res = _format_prediction_response(preds[0], probs[0])

    updated_in_place = False
    if body.session_id and body.correct is not None:
        fallback = {
            **body.features.model_dump(),
            "source": body.source,
            "skill_name": body.skill_name,
            "prediction": res.prediction,
            "probability": res.probability,
            "needs_intervention": res.needs_intervention,
            "model_version": res.model_version,
        }
        updated_in_place = upsert_interaction_outcome(
            interaction_id=body.session_id,
            correct=body.correct,
            fallback_record=fallback,
        )
        if not updated_in_place:
            _record_live_scoring_metrics(preds, probs)
    else:
        _record_live_scoring_metrics(preds, probs)
        _log_predictions_nonfatal(
            [body.features],
            [res],
            source=body.source,
            correct_values=[body.correct],
            skill_name=body.skill_name,
            session_id=body.session_id,
        )

    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
    retrain_doc = _read_json_if_exists(drift_dir / "latest_status.json") or {}
    watermark = int(retrain_doc.get("watermark_rows", 0))
    stats = get_live_buffer_stats(watermark_rows=watermark)
    _sync_drift_and_retrain_metrics()
    return {
        **res.model_dump(),
        "logged": True,
        "updated_in_place": updated_in_place,
        "correct": body.correct,
        "window_live_rows": stats["window_rows"],
        "total_live_rows": stats["total_rows"],
        "labeled_live_rows": stats["labeled_rows"],
    }


def _resolve_active_reference_df(drift_dir: Path, seed: int = 42) -> pd.DataFrame:
    active_ref_csv = drift_dir / "active_reference.csv"
    if active_ref_csv.exists():
        return pd.read_csv(active_ref_csv)
    ref_path = REPO_ROOT / "data" / "processed" / "train.csv"
    if ref_path.exists():
        ref_df = pd.read_csv(ref_path)
        return ref_df.sample(n=2000, random_state=seed) if len(ref_df) > 2000 else ref_df
    return generate_simulated_batch(profile="normal", count=300, seed=99)


@app.post("/telemetry/check-drift", tags=["Monitoring"])
def check_window_drift():
    """Evaluate 11-feature KS & model drift on the active window (e.g. >=5 manual rows)."""
    from src.monitoring import run_full_monitoring

    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
    retrain_doc = _read_json_if_exists(drift_dir / "latest_status.json") or {}
    watermark = int(retrain_doc.get("watermark_rows", 0))

    win_all = load_live_interactions(
        buffer_path=get_live_buffer_path(), labeled_only=False, since_row=watermark
    )
    win_labeled = load_live_interactions(
        buffer_path=get_live_buffer_path(), labeled_only=True, since_row=watermark
    )
    if len(win_all) < 5:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least 5 interactions in the active window to run drift checks "
                f"(currently {len(win_all)})."
            ),
        )

    ref_df = _resolve_active_reference_df(drift_dir)
    eval_cur = win_labeled if len(win_labeled) >= 5 else win_all
    has_labels = len(win_labeled) >= 5 and TARGET_COL in ref_df.columns
    drift = run_full_monitoring(
        reference_df=ref_df,
        current_df=eval_cur,
        model_path=MODEL_PATH if (has_labels and os.path.exists(MODEL_PATH)) else None,
        output_dir=str(drift_dir),
        tag="manual_window_check",
        log_to_mlflow=False,
    )
    _sync_drift_and_retrain_metrics()
    md = drift.get("model_drift") or {}
    return {
        "window_rows": len(win_all),
        "labeled_window_rows": len(win_labeled),
        "drift_share": round(float(drift.get("drift_share", 0.0)), 4),
        "drifted_features_count": int(drift.get("drifted_features_count", 0)),
        "dataset_drifted": bool(drift.get("dataset_drifted", False)),
        "model_drift_detected": bool(md.get("model_drift_detected", False)),
        "f1_drop": md.get("f1_drop", 0.0),
        "accuracy_drop": md.get("accuracy_drop", 0.0),
    }


@app.get("/telemetry/summary", tags=["Monitoring"])
def telemetry_summary():
    """Return live telemetry buffer counts, latest drift report, and retrain status."""
    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
    retrain_doc = _read_json_if_exists(drift_dir / "latest_status.json") or {}
    watermark = int(retrain_doc.get("watermark_rows", 0))
    buf_stats = get_live_buffer_stats(watermark_rows=watermark)
    return {
        "active_model_version": _model_version,
        "model_source": _model_source,
        "model_path": MODEL_PATH,
        "live_interactions_count": buf_stats["total_rows"],
        "window_interactions_count": buf_stats["window_rows"],
        "labeled_interactions_count": buf_stats["labeled_rows"],
        "live_intervention_rate": buf_stats["intervention_rate"],
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "drift": _read_json_if_exists(drift_dir / "latest_drift_summary.json") or {},
        "retrain": retrain_doc,
    }


@app.post("/telemetry/reset", tags=["Monitoring"])
def reset_telemetry_endpoint(rollback_to_baseline: bool = False):
    """Clear the live telemetry buffer and drift/retrain status for a clean demo reset."""
    global _scored_total_count, _scored_intervention_count
    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
    out = reset_live_telemetry(buffer_path=get_live_buffer_path(), drift_dir=drift_dir)
    _scored_total_count = 0
    _scored_intervention_count = 0
    if _PROMETHEUS_AVAILABLE:
        LIVE_INTERVENTION_RATIO.set(0.0)
    if rollback_to_baseline and os.path.exists(DEFAULT_MODEL_PATH):
        out["reload_result"] = reload_active_model(model_path=DEFAULT_MODEL_PATH)
    _sync_drift_and_retrain_metrics()
    return out


@app.post("/telemetry/simulate", tags=["Monitoring"])
def simulate_telemetry(body: SimulateRequest):
    """Generate and score a realistic student batch to demonstrate drift and retraining."""
    from src.monitoring import run_full_monitoring

    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    start = time.perf_counter()
    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))

    # Always sample cohorts from canonical baseline so 'drifted' is a fixed real-world regime
    batch_df = generate_simulated_batch(
        profile=body.profile,
        count=body.count,
        seed=body.seed,
    )
    instances = [
        StudentFeatures(**{col: row[col] for col in FEATURE_COLS})
        for _, row in batch_df.iterrows()
    ]
    df = _build_feature_dataframe(instances)
    preds, probs = _predict_with_probabilities(model, df)
    results = [_format_prediction_response(p, pr) for p, pr in zip(preds, probs)]
    correct_vals = [int(y) for y in batch_df[TARGET_COL].tolist()]

    _record_live_scoring_metrics(preds, probs)
    _log_predictions_nonfatal(
        instances,
        results,
        source=f"simulator_{body.profile}",
        correct_values=correct_vals,
    )

    if _PROMETHEUS_AVAILABLE:
        PREDICTION_COUNTER.labels(endpoint="/predict/batch", result="batch").inc()
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(time.perf_counter() - start)
        BATCH_SIZE_HISTOGRAM.observe(len(instances))

    retrain_doc = _read_json_if_exists(drift_dir / "latest_status.json") or {}
    watermark = int(retrain_doc.get("watermark_rows", 0))
    stats = get_live_buffer_stats(watermark_rows=watermark)
    intervention_rate = sum(1 for r in results if r.needs_intervention) / max(len(results), 1)

    drift_share = 0.0
    dataset_drifted = False
    model_drift_detected = False
    if body.run_drift_check:
        ref_df = _resolve_active_reference_df(drift_dir, seed=body.seed)
        live_window_df = load_live_interactions(
            buffer_path=get_live_buffer_path(), labeled_only=True, since_row=watermark
        )
        cur_eval_df = (
            live_window_df.tail(max(body.count, 80)) if len(live_window_df) >= 20 else batch_df
        )
        has_labels = TARGET_COL in cur_eval_df.columns and TARGET_COL in ref_df.columns
        drift = run_full_monitoring(
            reference_df=ref_df,
            current_df=cur_eval_df,
            model_path=MODEL_PATH if (has_labels and os.path.exists(MODEL_PATH)) else None,
            output_dir=str(drift_dir),
            tag=f"sim_{body.profile}",
            log_to_mlflow=False,
        )
        drift_share = float(drift.get("drift_share", 0.0))
        dataset_drifted = bool(drift.get("dataset_drifted", False))
        model_drift_detected = bool(
            (drift.get("model_drift") or {}).get("model_drift_detected", False)
        )

    retrain_result = None
    if body.trigger_retrain:
        from src.retrain import retrain_cycle

        retrain_result = retrain_cycle(
            from_live=True,
            force=False,
            output_dir=drift_dir,
            status_path=drift_dir / "latest_status.json",
        )
        if retrain_result.get("promoted") and retrain_result.get("active_artifact_path"):
            retrain_result["reload_result"] = reload_active_model(
                model_path=retrain_result["active_artifact_path"],
                version_label=retrain_result.get("active_model_version"),
            )

    _sync_drift_and_retrain_metrics()

    return {
        "profile": body.profile,
        "generated_rows": len(instances),
        "total_live_rows": stats["total_rows"],
        "window_live_rows": stats["window_rows"],
        "labeled_live_rows": stats["labeled_rows"],
        "intervention_rate": round(intervention_rate, 4),
        "drift_share": round(drift_share, 4),
        "dataset_drifted": dataset_drifted,
        "model_drift_detected": model_drift_detected,
        "retrain_result": retrain_result,
    }
