# =============================================================================
#  Dockerfile — AI-Based Digital Learning Assistant (FastAPI + Champion RF)
# =============================================================================
# Stage 1: Install lean serving dependencies only (no Jupyter/DVC/test bloat)
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements-serving.txt

# =============================================================================
# Stage 2: Non-root production runtime image
# =============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="MLOps Team <https://github.com/suhaybfr/ML-ops>" \
      org.opencontainers.image.title="student-performance-predictor" \
      org.opencontainers.image.description="Champion Random Forest student performance predictor served via FastAPI" \
      org.opencontainers.image.source="https://github.com/suhaybfr/ML-ops"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/baseline_model.pkl \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    HOME=/home/mlopsuser \
    PORT=8000

# UID/GID 1000 by default so the ./chroma_db bind mount (docker-compose) is writable
ARG APP_UID=1000
ARG APP_GID=1000

WORKDIR /app

# Copy installed serving packages from builder
COPY --from=builder /install /usr/local

# Copy inference service, canonical 11-feature schema, config, and DVC-pulled model
COPY app/        ./app/
COPY src/        ./src/
COPY models/     ./models/
COPY params.yaml ./params.yaml
# Course documents for `python -m app.tutor --index docs/` and POST /tutor/index
COPY docs/       ./docs/

# Run as unprivileged user
RUN addgroup --system --gid ${APP_GID} mlops \
    && adduser --system --uid ${APP_UID} --ingroup mlops --home /home/mlopsuser mlopsuser \
    && mkdir -p /app/chroma_db \
    && chown -R mlopsuser:mlops /app /home/mlopsuser

USER mlopsuser

# Pre-bake ChromaDB's ONNX all-MiniLM-L6-v2 embedding model (~80 MB, CPU-only) so the
# tutor works without a runtime download
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['warmup'])"

EXPOSE 8000

# Verify both HTTP liveness and loaded model readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import json, urllib.request; d = json.loads(urllib.request.urlopen('http://localhost:8000/health').read()); assert d.get('status') == 'ok' and d.get('model_loaded') is True" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
