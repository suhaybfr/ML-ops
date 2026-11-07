"""
FastAPI prediction service for the Student Performance SVM model.

Endpoints:
  GET  /          → welcome message
  GET  /health    → liveness check
  GET  /model/info → model metadata
  GET  /metrics   → Prometheus metrics (scraped by Prometheus)
  POST /predict   → single prediction
  POST /predict/batch → batch predictions

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

# Prometheus metrics (optional – gracefully skipped if not installed)
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Student Performance Predictor",
    description=(
        "MLOps-powered API that predicts whether a student will answer a "
        "knowledge-component question **correctly** (1) or **incorrectly** (0). "
        "Model: RBF-kernel SVM trained on the ASSISTments dataset via DVC + MLflow. "
        "Also includes an LLM-powered tutor via the /tutor endpoints."
    ),
    version="1.1.0",
    contact={"name": "MLOps Team", "url": "https://github.com/suhaybfr/ML-ops"},
)

# ── LLM/RAG Tutor router (optional — gracefully skipped if deps missing) ──
try:
    from tutor import router as tutor_router
    app.include_router(tutor_router)
except ImportError:
    pass

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
    PREDICTION_COUNTER = Counter(
        "prediction_requests_total",
        "Total number of prediction requests",
        ["endpoint", "result"],
    )
    PREDICTION_LATENCY = Histogram(
        "prediction_latency_seconds",
        "Prediction latency in seconds",
        ["endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    MODEL_LOADED_GAUGE = Gauge(
        "model_loaded",
        "1 if the ML model is loaded and ready, 0 otherwise",
    )
    REQUEST_ERRORS = Counter(
        "prediction_errors_total",
        "Total prediction errors",
        ["endpoint", "error_type"],
    )
    BATCH_SIZE_HISTOGRAM = Histogram(
        "batch_prediction_size",
        "Number of instances per batch prediction request",
        buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
    )

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "svm_model.pkl"),
)

_model = None
_model_load_time: Optional[float] = None


def get_model():
    global _model, _model_load_time
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found at '{MODEL_PATH}'. "
                "Run 'dvc pull' to fetch the model artifact."
            )
        _model = joblib.load(MODEL_PATH)
        _model_load_time = time.time()
        if _PROMETHEUS_AVAILABLE:
            MODEL_LOADED_GAUGE.set(1)
    return _model


# Attempt model load at startup (non-fatal if missing — tests may skip)
try:
    get_model()
except RuntimeError as e:
    print(f"[WARN] {e}", file=sys.stderr)
    if _PROMETHEUS_AVAILABLE:
        MODEL_LOADED_GAUGE.set(0)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "skill_id",
    "attempt_count",
    "ms_first_response",
    "hint_count",
    "overlap_time",
]


class StudentFeatures(BaseModel):
    """Input features for a single prediction request."""

    skill_id: float = Field(
        ..., ge=0, description="Encoded skill / knowledge-component ID."
    )
    attempt_count: int = Field(
        ..., ge=1, le=100, description="Number of attempts on this skill so far."
    )
    ms_first_response: float = Field(
        ..., ge=0, description="Milliseconds until the student's first response."
    )
    hint_count: int = Field(
        ..., ge=0, le=50, description="Number of hints requested."
    )
    overlap_time: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of question time overlapping with prior activity."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "skill_id": 7,
            "attempt_count": 2,
            "ms_first_response": 4500,
            "hint_count": 1,
            "overlap_time": 0.05,
        }
    }}


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = incorrect, 1 = correct")
    confidence: float = Field(..., description="Model decision score (higher = more confident)")
    label: str = Field(..., description="Human-readable label")
    model_version: str = Field(default="svm_rbf_v1")


class BatchPredictionRequest(BaseModel):
    instances: List[StudentFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root():
    """Welcome message and quick usage guide."""
    return {
        "message": "🎓 Student Performance Predictor API",
        "description": "RBF-SVM model trained on ASSISTments dataset via MLOps pipeline.",
        "endpoints": {
            "health": "GET /health",
            "model_info": "GET /model/info",
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "metrics": "GET /metrics (Prometheus)",
            "docs": "GET /docs",
        },
        "project": "https://github.com/suhaybfr/ML-ops",
        "dagshub": "https://dagshub.com/suhaybfr/ML-ops",
    }


@app.get("/health", tags=["General"])
def health():
    """Liveness probe — returns status and model readiness."""
    model_ready = _model is not None
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model_loaded": model_ready,
            "model_path": MODEL_PATH,
            "uptime_seconds": (
                round(time.time() - _model_load_time, 1) if _model_load_time else None
            ),
        },
    )


@app.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
def metrics():
    """Prometheus metrics endpoint — scraped by the Prometheus server."""
    if not _PROMETHEUS_AVAILABLE:
        return PlainTextResponse(
            "# prometheus_client not installed\n",
            media_type="text/plain",
        )
    return PlainTextResponse(
        generate_latest(REGISTRY).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata about the deployed model."""
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Inspect the sklearn Pipeline steps
    steps = [
        {"name": name, "type": type(step).__name__}
        for name, step in model.steps
    ]
    svc = model.steps[-1][1]  # last step is the SVC

    return {
        "model_type": "SVM (RBF Kernel)",
        "pipeline_steps": steps,
        "hyperparameters": {
            "kernel": getattr(svc, "kernel", "N/A"),
            "C": getattr(svc, "C", "N/A"),
            "gamma": str(getattr(svc, "gamma", "N/A")),
        },
        "feature_columns": FEATURE_COLS,
        "target": "correct (0=incorrect, 1=correct)",
        "training_framework": "scikit-learn via DVC pipeline",
        "experiment_tracking": "MLflow on DagsHub",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: StudentFeatures):
    """
    Predict whether a student will answer a question correctly.

    Returns:
        - **prediction**: 0 (incorrect) or 1 (correct)
        - **confidence**: raw decision-function score (higher = more confident correct)
        - **label**: human-readable result
    """
    start = time.perf_counter()
    try:
        model = get_model()
    except RuntimeError as exc:
        if _PROMETHEUS_AVAILABLE:
            REQUEST_ERRORS.labels(endpoint="/predict", error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc))

    row = pd.DataFrame([features.model_dump()])[FEATURE_COLS]

    prediction = int(model.predict(row)[0])

    # Use decision_function for confidence score
    if hasattr(model, "decision_function"):
        confidence = float(model.decision_function(row)[0])
    elif hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(row)[0][1])
    else:
        confidence = float(prediction)

    label = "✅ Correct" if prediction == 1 else "❌ Incorrect"

    if _PROMETHEUS_AVAILABLE:
        latency = time.perf_counter() - start
        PREDICTION_COUNTER.labels(endpoint="/predict", result=str(prediction)).inc()
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(latency)

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        label=label,
        model_version="svm_rbf_v1",
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple students at once.

    Accepts a list of feature objects and returns predictions for each.
    Maximum batch size: 1000 instances.
    """
    if len(request.instances) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 1000 instances.",
        )
    if len(request.instances) == 0:
        raise HTTPException(status_code=400, detail="Empty batch — provide at least one instance.")

    start = time.perf_counter()
    try:
        model = get_model()
    except RuntimeError as exc:
        if _PROMETHEUS_AVAILABLE:
            REQUEST_ERRORS.labels(endpoint="/predict/batch", error_type="model_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc))

    rows = pd.DataFrame([inst.model_dump() for inst in request.instances])[FEATURE_COLS]
    predictions = model.predict(rows).tolist()

    if hasattr(model, "decision_function"):
        confidences = model.decision_function(rows).tolist()
    elif hasattr(model, "predict_proba"):
        confidences = model.predict_proba(rows)[:, 1].tolist()
    else:
        confidences = predictions

    results = [
        PredictionResponse(
            prediction=int(p),
            confidence=round(float(c), 4),
            label="✅ Correct" if p == 1 else "❌ Incorrect",
            model_version="svm_rbf_v1",
        )
        for p, c in zip(predictions, confidences)
    ]

    if _PROMETHEUS_AVAILABLE:
        latency = time.perf_counter() - start
        PREDICTION_COUNTER.labels(endpoint="/predict/batch", result="batch").inc()
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(latency)
        BATCH_SIZE_HISTOGRAM.observe(len(request.instances))

    return BatchPredictionResponse(predictions=results, count=len(results))
