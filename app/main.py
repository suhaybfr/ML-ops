"""FastAPI inference service for the AI-Based Digital Learning Assistant.

Serves the Phase 2 Champion Random Forest model (``models/baseline_model.pkl``)
with optional MLflow Model Registry loading (``MLFLOW_MODEL_URI``) and clean
support for ``sklearn.pipeline.Pipeline`` comparison models
(``models/logreg_baseline.pkl``, ``models/svm_model.pkl``).

Endpoints:
  GET  /              -> service overview & endpoint directory
  GET  /health        -> liveness & model-readiness probe
  GET  /model/info    -> deployed model metadata, hyperparameters & 11-feature contract
  GET  /metrics       -> Prometheus exposition endpoint
  POST /predict       -> single student attempt prediction (11 engineered features)
  POST /predict/batch -> batch predictions (up to 1,000 instances)

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from src.features import ENGINEERED_FEATURES

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
    version="2.0.0",
    contact={"name": "MLOps Team", "url": "https://github.com/suhaybfr/ML-ops"},
)

# Optional Phase 3 LLM/RAG Tutor router (decoupled in Phase 2 unless ENABLE_TUTOR=1)
if os.environ.get("ENABLE_TUTOR", "0") == "1":
    for _tutor_module in ("app.tutor", "tutor"):
        try:
            _mod = __import__(_tutor_module, fromlist=["router"])
            app.include_router(_mod.router)
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

if _PROMETHEUS_AVAILABLE:
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

# ---------------------------------------------------------------------------
# Model loading (MLflow Model Registry optional + local DVC .pkl default)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "baseline_model.pkl")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
MLFLOW_MODEL_URI = os.environ.get("MLFLOW_MODEL_URI", "").strip()

FEATURE_COLS: list[str] = list(ENGINEERED_FEATURES)

_ARTIFACT_VERSION_MAP = {
    "baseline_model.pkl": "random_forest_baseline_v2",
    "logreg_baseline.pkl": "logistic_regression_baseline_v2",
    "svm_model.pkl": "svm_rbf_holdout_v2",
}

_model: Any = None
_model_load_time: Optional[float] = None
_model_source: str = "unloaded"
_model_version: str = "unknown"
_loaded_cache_key: Optional[tuple[str, str]] = None


def _infer_version_label(model_path: str, model_obj: Any, registry_uri: str = "") -> str:
    if registry_uri:
        return registry_uri.replace("models:/", "").replace("/", ":")
    basename = os.path.basename(model_path)
    if basename in _ARTIFACT_VERSION_MAP:
        return _ARTIFACT_VERSION_MAP[basename]
    est = model_obj.steps[-1][1] if isinstance(model_obj, Pipeline) else model_obj
    return f"{type(est).__name__.lower()}_custom"


def reset_model_cache() -> None:
    """Clear the cached model singleton (used by tests when swapping MODEL_PATH)."""
    global _model, _model_load_time, _model_source, _model_version, _loaded_cache_key
    _model = None
    _model_load_time = None
    _model_source = "unloaded"
    _model_version = "unknown"
    _loaded_cache_key = None


def get_model():
    """Load and cache the configured model (Registry URI first, then local .pkl)."""
    global _model, _model_load_time, _model_source, _model_version, _loaded_cache_key

    desired_key = (MLFLOW_MODEL_URI, os.path.abspath(MODEL_PATH))
    if _model is not None and _loaded_cache_key == desired_key:
        return _model

    loaded = None
    source = "local_artifact"
    version = "unknown"

    if MLFLOW_MODEL_URI:
        try:
            import mlflow.sklearn
            from src.utils import setup_mlflow

            setup_mlflow()
            loaded = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
            source = "mlflow_registry"
            version = _infer_version_label(MODEL_PATH, loaded, registry_uri=MLFLOW_MODEL_URI)
        except Exception as exc:
            print(
                f"[WARN] Could not load MLFLOW_MODEL_URI={MLFLOW_MODEL_URI!r} ({exc}); "
                f"falling back to MODEL_PATH={MODEL_PATH!r}.",
                file=sys.stderr,
            )

    if loaded is None:
        if not os.path.exists(MODEL_PATH):
            if _PROMETHEUS_AVAILABLE:
                MODEL_LOADED_GAUGE.set(0)
            raise RuntimeError(
                f"Model artifact not found at '{MODEL_PATH}'. "
                "Run 'dvc pull' to fetch the DVC-tracked model."
            )
        loaded = joblib.load(MODEL_PATH)
        source = "local_artifact"
        version = _infer_version_label(MODEL_PATH, loaded)

    _model = loaded
    _model_load_time = time.time()
    _model_source = source
    _model_version = version
    _loaded_cache_key = desired_key
    if _PROMETHEUS_AVAILABLE:
        MODEL_LOADED_GAUGE.set(1)
    return _model


# Attempt non-fatal warm load at startup
try:
    get_model()
except RuntimeError as exc:
    print(f"[WARN] {exc}", file=sys.stderr)


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["General"])
def root():
    """Service welcome message and endpoint guide."""
    return {
        "service": "AI-Based Digital Learning Assistant — Student Performance API",
        "version": app.version,
        "champion_model": "RandomForestClassifier (models/baseline_model.pkl)",
        "feature_count": len(FEATURE_COLS),
        "endpoints": {
            "health": "GET /health",
            "model_info": "GET /model/info",
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "metrics": "GET /metrics",
            "docs": "GET /docs",
        },
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
    """Prometheus exposition endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", media_type="text/plain")
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
        "target": "correct (0=struggling/needs_intervention, 1=correct)",
        "training_framework": "scikit-learn via DVC pipeline",
        "experiment_tracking": "MLflow on DAGsHub",
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

    if _PROMETHEUS_AVAILABLE:
        latency = time.perf_counter() - start
        PREDICTION_COUNTER.labels(endpoint="/predict", result=str(response.prediction)).inc()
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(latency)

    return response


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

    if _PROMETHEUS_AVAILABLE:
        latency = time.perf_counter() - start
        PREDICTION_COUNTER.labels(endpoint="/predict/batch", result="batch").inc()
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(latency)
        BATCH_SIZE_HISTOGRAM.observe(len(request.instances))

    return BatchPredictionResponse(predictions=results, count=len(results))
