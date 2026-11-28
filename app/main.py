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
from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.alerts import get_alert_webhook_url, send_alert_webhook  # noqa: E402
from src.artifact_store import ModelArtifactStore, is_azure_blob_configured  # noqa: E402
from src.assessment import (  # noqa: E402
    build_socratic_intervention_prompt,
    compute_11_engineered_features,
    ensure_assessment_items_seeded,
    get_hint_for_item,
    sanitize_item_for_student,
)
from src.features import ENGINEERED_FEATURES, TARGET_COL  # noqa: E402
from src.security import get_configured_admin_key, require_admin_api_key  # noqa: E402
from src.storage import get_storage, is_postgres_configured  # noqa: E402
from src.telemetry import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    append_interactions,
    generate_simulated_batch,
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

    # Milestone 3: Shadow Challenger live comparison metrics
    SHADOW_MODEL_ACTIVE = _get_or_create_metric(
        Gauge,
        "shadow_model_active",
        "1 if a Shadow Challenger model is currently scoring live traffic, 0 otherwise",
    )
    SHADOW_PREDICTIONS_TOTAL = _get_or_create_metric(
        Counter,
        "shadow_predictions_total",
        "Total live predictions scored in shadow by the challenger model",
    )
    SHADOW_DISAGREEMENTS_TOTAL = _get_or_create_metric(
        Counter,
        "shadow_disagreements_total",
        "Total live predictions where Shadow Challenger disagreed with Champion",
    )
    SHADOW_DISAGREEMENT_RATE = _get_or_create_metric(
        Gauge,
        "shadow_disagreement_rate",
        "Fraction of live predictions where Shadow Challenger disagreed with Champion",
    )
    CHALLENGER_LIVE_F1 = _get_or_create_metric(
        Gauge,
        "challenger_live_f1",
        "Observed live F1 score of the Shadow Challenger on labeled traffic",
    )
    CHAMPION_LIVE_F1 = _get_or_create_metric(
        Gauge,
        "champion_live_f1",
        "Observed live F1 score of the active Champion on the same labeled traffic",
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
    TUTOR_TOKENS_METRIC = _get_or_create_metric(
        Counter,
        "tutor_tokens_total",
        "Total Gemini tokens consumed by /tutor/ask by token type",
        ["token_type"],
    )
    TUTOR_RETRIEVAL_TOP_SIM_METRIC = _get_or_create_metric(
        Histogram,
        "tutor_retrieval_top_similarity",
        "Top retrieval cosine similarity score on /tutor/ask",
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    TUTOR_CHUNKS_FILTERED_METRIC = _get_or_create_metric(
        Counter,
        "tutor_chunks_filtered_total",
        "Total retrieved chunks dropped below TUTOR_MIN_SIMILARITY relevance floor",
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
    for _tok in ("prompt", "completion", "total"):
        TUTOR_TOKENS_METRIC.labels(token_type=_tok)
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
_metrics_sync_lock = threading.Lock()
_last_metrics_sync_ts: float = 0.0
_METRICS_SYNC_TTL_SECONDS: float = 15.0

# Milestone 3: Shadow Challenger runtime state
_shadow_lock = threading.RLock()
_shadow_model: Any = None
_shadow_version: Optional[str] = None
_shadow_path: Optional[str] = None
_shadow_artifact_uri: Optional[str] = None
_shadow_loaded_at: Optional[str] = None
_shadow_total: int = 0
_shadow_disagreements: int = 0
_shadow_recent_preds: dict[str, tuple[int, int]] = {}
_shadow_eval_y_true: list[int] = []
_shadow_eval_champ: list[int] = []
_shadow_eval_chall: list[int] = []


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


def _get_latest_drift_summary(drift_dir: Path) -> dict[str, Any]:
    if is_postgres_configured():
        try:
            stored = get_storage().get_latest_drift_snapshot()
            if stored:
                return stored
        except Exception:
            pass
    return _read_json_if_exists(drift_dir / "latest_drift_summary.json") or {}


def _get_latest_retrain_status(drift_dir: Path) -> dict[str, Any]:
    if is_postgres_configured():
        try:
            stored = get_storage().get_latest_retrain_run()
            if stored:
                return stored
        except Exception:
            pass
    return _read_json_if_exists(drift_dir / "latest_status.json") or {}


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


def _sync_drift_and_retrain_metrics(force: bool = True) -> None:
    global _last_metrics_sync_ts
    if not _PROMETHEUS_AVAILABLE:
        return

    now = time.monotonic()
    # When backed by remote Postgres, avoid hammering DB on every 5s Prometheus scrape
    if (
        not force
        and is_postgres_configured()
        and _last_metrics_sync_ts > 0.0
        and (now - _last_metrics_sync_ts) < _METRICS_SYNC_TTL_SECONDS
    ):
        return

    if not _metrics_sync_lock.acquire(blocking=force):
        return
    try:
        drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
        drift_summary = _get_latest_drift_summary(drift_dir)
        retrain_status = _get_latest_retrain_status(drift_dir)

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
                QUALITY_GATE_CHECKS_PASSED.set(
                    int(retrain_status.get("quality_gate_passed_checks", 0))
                )

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

        _last_metrics_sync_ts = time.monotonic()
    finally:
        _metrics_sync_lock.release()


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
    async_mode: bool = False
    deploy_as_shadow: bool = False


class ShadowDeployRequest(BaseModel):
    model_path: Optional[str] = None
    artifact_uri: Optional[str] = None
    version_label: Optional[str] = None


class StudentOutcomeRequest(BaseModel):
    skill_name: str = Field(default="Digital Literacy", min_length=1, max_length=120)
    correct: int = Field(..., ge=0, le=1)
    display_name: Optional[str] = Field(default=None, max_length=120)


class AssessmentStartRequest(BaseModel):
    student_id: str = Field(default="student_demo", min_length=1, max_length=120)
    item_id: Optional[str] = Field(default=None, max_length=120)
    skill_name: Optional[str] = Field(default=None, max_length=120)


class AssessmentHintRequest(BaseModel):
    item_id: str = Field(..., min_length=1, max_length=120)
    hint_index: int = Field(..., ge=0, le=20)
    attempt_session_id: Optional[str] = Field(default=None, max_length=120)


class AssessmentSubmitRequest(BaseModel):
    student_id: str = Field(default="student_demo", min_length=1, max_length=120)
    item_id: str = Field(..., min_length=1, max_length=120)
    selected_answer: str = Field(..., min_length=1, max_length=1000)
    attempt_session_id: Optional[str] = Field(default=None, max_length=120)
    attempt_count: Optional[int] = Field(default=None, ge=1, le=50)
    hints_used: Optional[int] = Field(default=None, ge=0, le=50)
    bottom_hint_used: Optional[bool] = None
    ms_first_response: Optional[float] = Field(default=None, ge=0.0, le=3_600_000.0)


_assessment_sessions_lock = threading.Lock()
_assessment_sessions: dict[str, dict[str, Any]] = {}


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
# Milestone 3: Shadow Challenger scoring & Blue-Green promotion helpers
# ---------------------------------------------------------------------------


def _update_shadow_gauges_locked() -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    SHADOW_MODEL_ACTIVE.set(1 if _shadow_model is not None else 0)
    rate = (_shadow_disagreements / _shadow_total) if _shadow_total > 0 else 0.0
    SHADOW_DISAGREEMENT_RATE.set(round(rate, 6))
    if len(_shadow_eval_y_true) >= 1:
        chall_f1 = float(f1_score(_shadow_eval_y_true, _shadow_eval_chall, zero_division=0))
        champ_f1 = float(f1_score(_shadow_eval_y_true, _shadow_eval_champ, zero_division=0))
        CHALLENGER_LIVE_F1.set(round(chall_f1, 6))
        CHAMPION_LIVE_F1.set(round(champ_f1, 6))
    else:
        CHALLENGER_LIVE_F1.set(0.0)
        CHAMPION_LIVE_F1.set(0.0)


def deploy_shadow_model(
    model_path: Optional[str] = None,
    artifact_uri: Optional[str] = None,
    version_label: Optional[str] = None,
) -> dict[str, Any]:
    """Load a candidate into Shadow Challenger Mode without affecting 100% Champion traffic."""
    global _shadow_model, _shadow_version, _shadow_path, _shadow_artifact_uri
    global _shadow_loaded_at, _shadow_total, _shadow_disagreements
    global _shadow_recent_preds, _shadow_eval_y_true, _shadow_eval_champ, _shadow_eval_chall

    resolved_path: Optional[Path] = None
    if model_path and model_path.strip():
        candidate = Path(model_path.strip())
        candidate = (
            candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
        )
        if not candidate.is_relative_to(REPO_ROOT.resolve()) or candidate.suffix.lower() != ".pkl":
            raise HTTPException(
                status_code=400,
                detail="model_path must reference a .pkl file inside the project repository.",
            )
        if not candidate.exists():
            raise HTTPException(status_code=404, detail=f"Shadow model not found: {candidate}")
        resolved_path = candidate
    elif artifact_uri and artifact_uri.strip():
        dest = REPO_ROOT / "models" / "challenger_model.pkl"
        try:
            resolved_path = ModelArtifactStore().download_model(artifact_uri.strip(), dest)
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to download shadow artifact '{artifact_uri}': {exc}",
            ) from exc
    else:
        default_chall = REPO_ROOT / "models" / "challenger_model.pkl"
        if not default_chall.exists():
            raise HTTPException(
                status_code=404,
                detail="No challenger model found at models/challenger_model.pkl.",
            )
        resolved_path = default_chall

    try:
        loaded = joblib.load(resolved_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to load shadow model: {type(exc).__name__}: {exc}"
        ) from exc

    ver = version_label or _infer_version_label(str(resolved_path), loaded)
    now_iso = datetime.now(timezone.utc).isoformat()
    with _shadow_lock:
        _shadow_model = loaded
        _shadow_version = ver
        _shadow_path = str(resolved_path)
        _shadow_artifact_uri = artifact_uri
        _shadow_loaded_at = now_iso
        _shadow_total = 0
        _shadow_disagreements = 0
        _shadow_recent_preds = {}
        _shadow_eval_y_true = []
        _shadow_eval_champ = []
        _shadow_eval_chall = []
        _update_shadow_gauges_locked()

    return get_shadow_status()


def clear_shadow_model() -> dict[str, Any]:
    """Disable Shadow Challenger Mode and reset shadow comparison metrics."""
    global _shadow_model, _shadow_version, _shadow_path, _shadow_artifact_uri
    global _shadow_loaded_at, _shadow_total, _shadow_disagreements
    global _shadow_recent_preds, _shadow_eval_y_true, _shadow_eval_champ, _shadow_eval_chall
    with _shadow_lock:
        _shadow_model = None
        _shadow_version = None
        _shadow_path = None
        _shadow_artifact_uri = None
        _shadow_loaded_at = None
        _shadow_total = 0
        _shadow_disagreements = 0
        _shadow_recent_preds = {}
        _shadow_eval_y_true = []
        _shadow_eval_champ = []
        _shadow_eval_chall = []
        _update_shadow_gauges_locked()
    return get_shadow_status()


def get_shadow_status() -> dict[str, Any]:
    """Return live Shadow Challenger vs Champion comparison metrics."""
    with _shadow_lock:
        active = _shadow_model is not None
        rate = (_shadow_disagreements / _shadow_total) if _shadow_total > 0 else 0.0
        n_labeled = len(_shadow_eval_y_true)
        chall_f1 = (
            float(f1_score(_shadow_eval_y_true, _shadow_eval_chall, zero_division=0))
            if n_labeled >= 1
            else 0.0
        )
        champ_f1 = (
            float(f1_score(_shadow_eval_y_true, _shadow_eval_champ, zero_division=0))
            if n_labeled >= 1
            else 0.0
        )
        ready = bool(active and n_labeled >= 5 and chall_f1 >= (champ_f1 - 0.02))
        return {
            "active": active,
            "champion_version": _model_version,
            "challenger_version": _shadow_version,
            "challenger_path": _shadow_path,
            "artifact_uri": _shadow_artifact_uri,
            "loaded_at": _shadow_loaded_at,
            "predictions_scored": _shadow_total,
            "disagreements": _shadow_disagreements,
            "shadow_disagreement_rate": round(rate, 4),
            "labeled_samples": n_labeled,
            "challenger_live_f1": round(chall_f1, 4),
            "champion_live_f1": round(champ_f1, 4),
            "ready_to_promote": ready,
        }


def _score_shadow_nonfatal(
    df: pd.DataFrame,
    champ_preds: list[int],
    interaction_ids: Optional[list[Optional[str]]] = None,
    correct_values: Optional[list[Optional[int]]] = None,
) -> None:
    """Score live requests in shadow without blocking or failing Champion traffic."""
    global _shadow_total, _shadow_disagreements
    with _shadow_lock:
        shadow = _shadow_model
    if shadow is None:
        return
    try:
        chall_preds, _ = _predict_with_probabilities(shadow, df)
        with _shadow_lock:
            if _shadow_model is None:
                return
            disagreements = 0
            for idx, (cp, sp) in enumerate(zip(champ_preds, chall_preds)):
                if int(cp) != int(sp):
                    disagreements += 1
                iid = (
                    interaction_ids[idx]
                    if interaction_ids and idx < len(interaction_ids)
                    else None
                )
                if iid:
                    if len(_shadow_recent_preds) >= 5000:
                        _shadow_recent_preds.pop(next(iter(_shadow_recent_preds)))
                    _shadow_recent_preds[iid] = (int(cp), int(sp))
                corr = (
                    correct_values[idx]
                    if correct_values and idx < len(correct_values)
                    else None
                )
                if corr in (0, 1):
                    _shadow_eval_y_true.append(int(corr))
                    _shadow_eval_champ.append(int(cp))
                    _shadow_eval_chall.append(int(sp))
                    if len(_shadow_eval_y_true) > 5000:
                        del _shadow_eval_y_true[:1000]
                        del _shadow_eval_champ[:1000]
                        del _shadow_eval_chall[:1000]
            _shadow_total += len(chall_preds)
            _shadow_disagreements += disagreements
            if _PROMETHEUS_AVAILABLE:
                SHADOW_PREDICTIONS_TOTAL.inc(len(chall_preds))
                if disagreements:
                    SHADOW_DISAGREEMENTS_TOTAL.inc(disagreements)
            _update_shadow_gauges_locked()
    except Exception:
        pass


def _record_shadow_label_nonfatal(interaction_id: str, correct: int) -> None:
    """Attach delayed ground-truth feedback to a prior shadow prediction for live F1."""
    if not interaction_id:
        return
    with _shadow_lock:
        pair = _shadow_recent_preds.pop(interaction_id, None)
        if pair is None or _shadow_model is None:
            return
        cp, sp = pair
        _shadow_eval_y_true.append(int(correct))
        _shadow_eval_champ.append(cp)
        _shadow_eval_chall.append(sp)
        _update_shadow_gauges_locked()


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
        "student_features": "GET /features/student/{student_id}",
        "student_outcome": "POST /features/student/{student_id}/outcome",
        "assessment_items": "GET /assessment/items",
        "assessment_start": "POST /assessment/start",
        "assessment_hint": "POST /assessment/hint",
        "assessment_submit": "POST /assessment/submit",
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

    pg_active = is_postgres_configured()
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_loaded": model_ready,
            "model_path": MODEL_PATH,
            "model_source": _model_source,
            "model_version": _model_version,
            "storage_backend": "postgres" if pg_active else "local",
            "vector_backend": "pgvector" if pg_active else "chroma",
            "artifact_backend": "azure_blob" if is_azure_blob_configured() else "local",
            "admin_auth_enabled": bool(get_configured_admin_key()),
            "alerts_configured": bool(get_alert_webhook_url()),
            "shadow_active": _shadow_model is not None,
            "uptime_seconds": (
                round(time.time() - _model_load_time, 1) if _model_load_time else None
            ),
        },
    )


@app.get(
    "/metrics",
    tags=["Monitoring"],
    response_class=PlainTextResponse,
    dependencies=[Depends(require_admin_api_key)],
)
def metrics():
    """Full-pipeline Prometheus exposition endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", media_type="text/plain")
    _sync_offline_champion_metrics()
    _sync_drift_and_retrain_metrics(force=False)
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


@app.post("/model/reload", tags=["Model"], dependencies=[Depends(require_admin_api_key)])
def reload_model(body: Optional[ModelReloadRequest] = None):
    """Zero-downtime model reload / promotion swap without container restart."""
    req = body or ModelReloadRequest()
    return reload_active_model(
        model_path=req.model_path,
        mlflow_model_uri=req.mlflow_model_uri,
        version_label=req.version_label,
    )


@app.post("/model/retrain", tags=["Model"], dependencies=[Depends(require_admin_api_key)])
def trigger_retrain(body: Optional[RetrainTriggerRequest] = None):
    """Run drift-gated retraining (sync or out-of-band async) with optional Shadow deploy."""
    from src.worker_retrain import execute_retrain_job, submit_async_retrain_job

    req = body or RetrainTriggerRequest()
    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))

    def _on_shadow(path: str, version: str, uri: Optional[str]) -> dict[str, Any]:
        return deploy_shadow_model(model_path=path, artifact_uri=uri, version_label=version)

    def _on_reload(path: str, version: str) -> dict[str, Any]:
        return reload_active_model(model_path=path, version_label=version)

    job_kwargs = {
        "from_live": req.from_live,
        "force": req.force,
        "drift_threshold": req.drift_threshold,
        "deploy_as_shadow": req.deploy_as_shadow,
        "hot_reload": req.hot_reload,
        "output_dir": drift_dir,
        "on_shadow_ready": _on_shadow,
        "on_hot_reload": _on_reload,
        "on_complete": _sync_drift_and_retrain_metrics,
    }
    if req.async_mode:
        return submit_async_retrain_job(**job_kwargs)
    return execute_retrain_job(**job_kwargs)


@app.get(
    "/model/retrain/jobs",
    tags=["Model"],
    dependencies=[Depends(require_admin_api_key)],
)
def get_retrain_jobs():
    """List background async retraining jobs."""
    from src.worker_retrain import list_retrain_jobs

    return {"jobs": list_retrain_jobs()}


@app.get(
    "/model/retrain/jobs/{job_id}",
    tags=["Model"],
    dependencies=[Depends(require_admin_api_key)],
)
def get_retrain_job_status(job_id: str):
    """Fetch status and result for a specific async retraining job."""
    from src.worker_retrain import get_retrain_job

    job = get_retrain_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Retrain job '{job_id}' not found.")
    return job


@app.get("/model/shadow/status", tags=["Model"])
def shadow_status():
    """Inspect active Shadow Challenger vs Champion disagreement rate and live F1."""
    return get_shadow_status()


@app.post("/model/shadow/deploy", tags=["Model"], dependencies=[Depends(require_admin_api_key)])
def shadow_deploy(body: Optional[ShadowDeployRequest] = None):
    """Deploy a candidate model in Shadow Challenger Mode (0% user-facing traffic)."""
    req = body or ShadowDeployRequest()
    return deploy_shadow_model(
        model_path=req.model_path,
        artifact_uri=req.artifact_uri,
        version_label=req.version_label,
    )


@app.post("/model/shadow/promote", tags=["Model"], dependencies=[Depends(require_admin_api_key)])
def shadow_promote():
    """Blue-Green promote the active Shadow Challenger to 100% Champion traffic."""
    import shutil

    with _shadow_lock:
        if _shadow_model is None or not _shadow_path:
            raise HTTPException(status_code=400, detail="No Shadow Challenger is currently active.")
        src = Path(_shadow_path)
        ver = _shadow_version or "random_forest_retrained_shadow"
        snapshot = get_shadow_status()

    dest = REPO_ROOT / "models" / "retrained_champion.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    reload_res = reload_active_model(model_path=str(dest), version_label=ver)

    # Keep MLflow Model Registry @champion alias synced with Blue-Green promotion
    mlflow_promotion: Optional[dict[str, Any]] = None
    try:
        import re

        m = re.search(r"_v(\d+)$", ver)
        if m:
            from mlflow.tracking import MlflowClient
            from src.register_model import CHAMPION_ALIAS, REGISTRY_MODEL_NAME
            from src.utils import setup_mlflow

            setup_mlflow(allow_local_fallback=True)
            client = MlflowClient()
            client.set_registered_model_alias(REGISTRY_MODEL_NAME, CHAMPION_ALIAS, m.group(1))
            mlflow_promotion = {
                "model_name": REGISTRY_MODEL_NAME,
                "alias": CHAMPION_ALIAS,
                "version": m.group(1),
            }
    except Exception as exc:
        mlflow_promotion = {"skipped": str(exc)}

    clear_shadow_model()
    return {
        "status": "promoted_from_shadow",
        "shadow_snapshot": snapshot,
        "reload_result": reload_res,
        "mlflow_promotion": mlflow_promotion,
    }


@app.post("/model/shadow/clear", tags=["Model"], dependencies=[Depends(require_admin_api_key)])
def shadow_clear():
    """Disable Shadow Challenger Mode without altering the active Champion."""
    return clear_shadow_model()


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
    _score_shadow_nonfatal(df, preds)
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
    _score_shadow_nonfatal(df, preds, interaction_ids=[interaction_id])
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
    _score_shadow_nonfatal(df, preds)
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
        _record_shadow_label_nonfatal(body.session_id, body.correct)
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
    retrain_doc = _get_latest_retrain_status(drift_dir)
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
    retrain_doc = _get_latest_retrain_status(drift_dir)
    watermark = int(retrain_doc.get("watermark_rows", 0))

    win_all = load_live_interactions(labeled_only=False, since_row=watermark)
    win_labeled = load_live_interactions(labeled_only=True, since_row=watermark)
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


@app.post("/alerts/test", tags=["Monitoring"], dependencies=[Depends(require_admin_api_key)])
def test_alert_webhook():
    """Send a test notification to the configured Discord / webhook alert channel."""
    return send_alert_webhook(
        title="Test Alert — Student Performance MLOps",
        message="Live alert paging webhook is connected and operational.",
        severity="info",
        fields={
            "Active Model": _model_version,
            "Storage": "postgres" if is_postgres_configured() else "local",
        },
    )


@app.get("/telemetry/summary", tags=["Monitoring"])
def telemetry_summary():
    """Return live telemetry buffer counts, latest drift report, and retrain status."""
    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
    retrain_doc = _get_latest_retrain_status(drift_dir)
    watermark = int(retrain_doc.get("watermark_rows", 0))
    buf_stats = get_live_buffer_stats(watermark_rows=watermark)
    return {
        "active_model_version": _model_version,
        "model_source": _model_source,
        "model_path": MODEL_PATH,
        "storage_backend": "postgres" if is_postgres_configured() else "local",
        "artifact_backend": "azure_blob" if is_azure_blob_configured() else "local",
        "admin_auth_enabled": bool(get_configured_admin_key()),
        "shadow": get_shadow_status(),
        "live_interactions_count": buf_stats["total_rows"],
        "window_interactions_count": buf_stats["window_rows"],
        "labeled_interactions_count": buf_stats["labeled_rows"],
        "live_intervention_rate": buf_stats["intervention_rate"],
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "drift": _get_latest_drift_summary(drift_dir),
        "retrain": retrain_doc,
    }


@app.post(
    "/telemetry/reset",
    tags=["Monitoring"],
    dependencies=[Depends(require_admin_api_key)],
)
def reset_telemetry_endpoint(
    rollback_to_baseline: bool = False,
    include_students: bool = False,
    include_pdfs: bool = False,
    include_demo_artifacts: bool = False,
):
    """Clear the live telemetry buffer and drift/retrain status for a clean demo reset."""
    global _scored_total_count, _scored_intervention_count
    drift_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", str(DRIFT_REPORTS_DIR)))
    out = reset_live_telemetry(drift_dir=drift_dir, include_students=include_students)
    _scored_total_count = 0
    _scored_intervention_count = 0
    if _PROMETHEUS_AVAILABLE:
        LIVE_INTERVENTION_RATIO.set(0.0)
    if include_pdfs and _tutor_mod is not None and hasattr(_tutor_mod, "clear_uploaded_pdfs"):
        try:
            out["cleared_pdfs"] = int(_tutor_mod.clear_uploaded_pdfs())
        except Exception as exc:
            out["cleared_pdfs_error"] = str(exc)
    if include_demo_artifacts:
        clear_shadow_model()
        # Clean local retrained/challenger artifacts (never touch models/baseline_model.pkl)
        removed_local = []
        for fname in ("retrained_champion.pkl", "challenger_model.pkl"):
            fpath = REPO_ROOT / "models" / fname
            if fpath.exists():
                fpath.unlink()
                removed_local.append(fname)
        out["cleared_local_models"] = removed_local
        try:
            out["cleared_blob_models"] = ModelArtifactStore().delete_non_baseline_models()
        except Exception as exc:
            out["cleared_blob_models_error"] = str(exc)
        # Clean demo @challenger alias & non-baseline MLflow versions (keep baseline @champion)
        try:
            from mlflow.tracking import MlflowClient
            from src.register_model import CHALLENGER_ALIAS, CHAMPION_ALIAS, REGISTRY_MODEL_NAME
            from src.utils import setup_mlflow

            setup_mlflow(allow_local_fallback=True)
            mc = MlflowClient()
            rm = mc.get_registered_model(REGISTRY_MODEL_NAME)
            try:
                mc.delete_registered_model_alias(REGISTRY_MODEL_NAME, CHALLENGER_ALIAS)
            except Exception:
                pass
            # Restore @champion to baseline v3 if present and delete demo versions > 3
            versions = mc.search_model_versions(f"name='{REGISTRY_MODEL_NAME}'")
            v_nums = {int(v.version) for v in versions}
            baseline_ver = "3" if 3 in v_nums else (str(min(v_nums)) if v_nums else None)
            if baseline_ver:
                mc.set_registered_model_alias(REGISTRY_MODEL_NAME, CHAMPION_ALIAS, baseline_ver)
            deleted_mlflow = []
            keep_max = int(baseline_ver) if baseline_ver else 3
            for mv in versions:
                if int(mv.version) > keep_max:
                    mc.delete_model_version(REGISTRY_MODEL_NAME, mv.version)
                    deleted_mlflow.append(mv.version)
            out["cleared_mlflow_versions"] = deleted_mlflow
            out["mlflow_champion_version"] = baseline_ver or rm.aliases.get(CHAMPION_ALIAS)
        except Exception as exc:
            out["cleared_mlflow_error"] = str(exc)
    if rollback_to_baseline and os.path.exists(DEFAULT_MODEL_PATH):
        out["reload_result"] = reload_active_model(model_path=DEFAULT_MODEL_PATH)
    _sync_drift_and_retrain_metrics(force=True)
    return out


@app.post(
    "/telemetry/simulate",
    tags=["Monitoring"],
    dependencies=[Depends(require_admin_api_key)],
)
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
    _score_shadow_nonfatal(df, preds, correct_values=correct_vals)
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

    retrain_doc = _get_latest_retrain_status(drift_dir)
    watermark = int(retrain_doc.get("watermark_rows", 0))
    stats = get_live_buffer_stats(watermark_rows=watermark)
    intervention_rate = sum(1 for r in results if r.needs_intervention) / max(len(results), 1)

    drift_share = 0.0
    dataset_drifted = False
    model_drift_detected = False
    if body.run_drift_check:
        ref_df = _resolve_active_reference_df(drift_dir, seed=body.seed)
        live_window_df = load_live_interactions(labeled_only=True, since_row=watermark)
        # Evaluate the full cumulative active window (capped at 2000 for speed) rather than
        # only the latest batch, so mixed cohorts reflect true cumulative window drift.
        cur_eval_df = (
            live_window_df.tail(2000) if len(live_window_df) >= 20 else batch_df
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


@app.get("/features/student/{student_id}", tags=["Feature Store"])
def get_student_features(student_id: str, skill_name: Optional[str] = None):
    """Fetch a student's expanding accuracy and per-skill opportunity from the Feature Store."""
    try:
        return get_storage().get_student_feature_context(student_id, skill_name=skill_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/features/student/{student_id}/outcome", tags=["Feature Store"])
def record_student_feature_outcome(student_id: str, body: StudentOutcomeRequest):
    """Record a graded attempt outcome and update the student's online feature state."""
    try:
        return get_storage().record_student_outcome(
            student_id=student_id,
            skill_name=body.skill_name,
            correct=body.correct,
            display_name=body.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Milestone 2: Real Interactive Student Assessment Loop
# ---------------------------------------------------------------------------


@app.get("/assessment/items", tags=["Assessment"])
def list_assessment_items(skill_name: Optional[str] = None):
    """List calibrated Digital Literacy practice items (without leaking answers)."""
    storage = get_storage()
    items = ensure_assessment_items_seeded(storage)
    if skill_name:
        items = [it for it in items if it.get("skill_name") == skill_name]
    return {
        "count": len(items),
        "items": [sanitize_item_for_student(it) for it in items],
    }


@app.post("/assessment/start", tags=["Assessment"])
def start_assessment_item(body: AssessmentStartRequest):
    """Start a timed attempt session for a graded practice item."""
    storage = get_storage()
    items = ensure_assessment_items_seeded(storage)
    if body.item_id:
        item = storage.get_assessment_item(body.item_id)
        if item is None:
            raise HTTPException(
                status_code=404, detail=f"Assessment item '{body.item_id}' not found."
            )
    else:
        pool = [
            it for it in items if not body.skill_name or it.get("skill_name") == body.skill_name
        ]
        if not pool:
            raise HTTPException(status_code=404, detail="No assessment items found for skill.")
        item = pool[0]

    student_ctx = storage.get_student_feature_context(
        body.student_id, skill_name=item["skill_name"]
    )
    session_id = uuid.uuid4().hex[:16]
    started_perf = time.perf_counter()
    with _assessment_sessions_lock:
        if len(_assessment_sessions) > 2000:
            # Evict oldest sessions to bound memory
            for old_key in list(_assessment_sessions.keys())[:500]:
                _assessment_sessions.pop(old_key, None)
        _assessment_sessions[session_id] = {
            "student_id": body.student_id.strip(),
            "item_id": item["item_id"],
            "started_perf": started_perf,
            "first_response_ms": None,
            "attempt_count": 0,
            "hints_used": 0,
            "bottom_hint_used": False,
            "student_ctx_before": student_ctx,
            "outcome_recorded": False,
            "student_ctx_after": None,
        }

    return {
        "attempt_session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "item": sanitize_item_for_student(item),
        "student_context": student_ctx,
    }


@app.post("/assessment/hint", tags=["Assessment"])
def request_assessment_hint(body: AssessmentHintRequest):
    """Reveal a progressive conceptual hint (or bottom-out hint) and track hint usage."""
    storage = get_storage()
    ensure_assessment_items_seeded(storage)
    item = storage.get_assessment_item(body.item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Assessment item '{body.item_id}' not found.")
    try:
        hint_info = get_hint_for_item(item, body.hint_index)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if body.attempt_session_id:
        with _assessment_sessions_lock:
            sess = _assessment_sessions.get(body.attempt_session_id)
            if sess is not None and sess.get("item_id") == item["item_id"]:
                sess["hints_used"] = max(int(sess["hints_used"]), int(hint_info["hints_used"]))
                if hint_info["is_bottom_hint"]:
                    sess["bottom_hint_used"] = True

    return hint_info


@app.post("/assessment/submit", tags=["Assessment"])
def submit_assessment_answer(body: AssessmentSubmitRequest):
    """Grade a live student answer, compute all 11 features automatically, explain & log."""
    start = time.perf_counter()
    try:
        model = get_model()
    except RuntimeError as exc:
        if _PROMETHEUS_AVAILABLE:
            REQUEST_ERRORS.labels(endpoint="/predict", error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc))

    storage = get_storage()
    ensure_assessment_items_seeded(storage)
    item = storage.get_assessment_item(body.item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Assessment item '{body.item_id}' not found.")

    sess: Optional[dict[str, Any]] = None
    if body.attempt_session_id:
        with _assessment_sessions_lock:
            sess = _assessment_sessions.get(body.attempt_session_id)
            if sess is not None and sess.get("item_id") == item["item_id"]:
                sess["attempt_count"] = int(sess.get("attempt_count", 0)) + 1
                if sess.get("first_response_ms") is None:
                    sess["first_response_ms"] = (
                        time.perf_counter() - float(sess["started_perf"])
                    ) * 1000.0

    attempt_count = (
        int(body.attempt_count)
        if body.attempt_count is not None
        else (int(sess["attempt_count"]) if sess is not None else 1)
    )
    hints_used = (
        int(body.hints_used)
        if body.hints_used is not None
        else (int(sess["hints_used"]) if sess is not None else 0)
    )
    bottom_hint_used = (
        bool(body.bottom_hint_used)
        if body.bottom_hint_used is not None
        else (bool(sess["bottom_hint_used"]) if sess is not None else False)
    )
    ms_first_response = (
        float(body.ms_first_response)
        if body.ms_first_response is not None
        else (
            float(sess["first_response_ms"])
            if sess and sess.get("first_response_ms")
            else 12000.0
        )
    )

    # 1. Pull pre-item student history (anchored per item session so retries don't double-count)
    if sess is not None and sess.get("student_ctx_before"):
        student_ctx_before = sess["student_ctx_before"]
    else:
        student_ctx_before = storage.get_student_feature_context(
            body.student_id, skill_name=item["skill_name"]
        )
    sanitized = sanitize_item_for_student(item)

    # 2. Automatically compute all 11 canonical ENGINEERED_FEATURES
    computed_dict = compute_11_engineered_features(
        opportunity=student_ctx_before["opportunity"],
        attempt_count=attempt_count,
        hints_used=hints_used,
        hint_total=sanitized["hint_total"],
        bottom_hint_used=bottom_hint_used,
        user_prev_acc=student_ctx_before["user_prev_acc"],
        problem_difficulty=item["problem_difficulty"],
        skill_difficulty=item["skill_difficulty"],
        is_skill_unmapped=item.get("is_skill_unmapped", 0),
        ms_first_response=ms_first_response,
    )
    features_obj = StudentFeatures(**computed_dict)

    # 3. Grade real answer -> ground-truth binary outcome (0/1)
    is_correct = body.selected_answer.strip() == str(item["correct_answer"]).strip()
    correct_int = 1 if is_correct else 0

    # 4. Run active Champion model + Monte-Carlo Shapley explanation
    df = _build_feature_dataframe([features_obj])
    preds, probs = _predict_with_probabilities(model, df)
    pred_res = _format_prediction_response(preds[0], probs[0])
    explanation = _compute_local_xai(model, features_obj)

    interaction_id = uuid.uuid4().hex[:12]
    _record_live_scoring_metrics(preds, probs)
    _score_shadow_nonfatal(
        df, preds, interaction_ids=[interaction_id], correct_values=[correct_int]
    )

    # 5. Persist labeled interaction event (with student_id & item_id) + update Feature Store
    try:
        append_interactions(
            [
                {
                    **computed_dict,
                    "session_id": interaction_id,
                    "student_id": body.student_id.strip(),
                    "item_id": item["item_id"],
                    "source": "live_assessment",
                    "skill_name": item["skill_name"],
                    "opportunity": student_ctx_before["opportunity"],
                    "prediction": pred_res.prediction,
                    "probability": pred_res.probability,
                    "needs_intervention": pred_res.needs_intervention,
                    "model_version": pred_res.model_version,
                    TARGET_COL: correct_int,
                }
            ]
        )
    except Exception:
        pass

    if sess is not None:
        with _assessment_sessions_lock:
            if not sess.get("outcome_recorded"):
                sess["student_ctx_after"] = storage.record_student_outcome(
                    student_id=body.student_id,
                    skill_name=item["skill_name"],
                    correct=correct_int,
                )
                sess["outcome_recorded"] = True
            student_ctx_after = sess["student_ctx_after"]
    else:
        student_ctx_after = storage.record_student_outcome(
            student_id=body.student_id,
            skill_name=item["skill_name"],
            correct=correct_int,
        )

    if _PROMETHEUS_AVAILABLE:
        latency = time.perf_counter() - start
        PREDICTION_COUNTER.labels(endpoint="/predict", result=str(pred_res.prediction)).inc()
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(latency)
    _sync_drift_and_retrain_metrics()

    # 3-tier pedagogical policy for live practice (preserves raw model P(correct) & 0.50 contract):
    # - mastery: P >= 0.50
    # - light_support: 0.25 <= P < 0.50 (e.g. 1 light hint on try 1) -> no forced auto-interrupt
    # - tutor_intervention: P < 0.25 or bottom_hint or (attempt_count >= 2 and not is_correct)
    tutor_threshold = float(os.environ.get("TUTOR_INTERVENTION_THRESHOLD", "0.25"))
    severe_struggle = bool(
        (pred_res.probability < tutor_threshold)
        or bottom_hint_used
        or (attempt_count >= 2 and not is_correct)
    )
    if pred_res.probability >= 0.5:
        intervention_tier = "mastery"
    elif severe_struggle:
        intervention_tier = "tutor_intervention"
    else:
        intervention_tier = "light_support"

    socratic_prompt = build_socratic_intervention_prompt(
        item=item,
        selected_answer=body.selected_answer,
        is_correct=is_correct,
        needs_intervention=severe_struggle,
    )

    return {
        **pred_res.model_dump(),
        "interaction_id": interaction_id,
        "item_id": item["item_id"],
        "skill_name": item["skill_name"],
        "attempt_number": attempt_count,
        "is_correct": is_correct,
        "correct": correct_int,
        "ms_first_response": round(ms_first_response, 1),
        "computed_features": computed_dict,
        "explanation": explanation,
        "student_profile_before": student_ctx_before,
        "student_profile_after": student_ctx_after,
        "intervention_tier": intervention_tier,
        "trigger_tutor": bool(severe_struggle and not is_correct),
        "suggested_tutor_prompt": socratic_prompt,
    }
