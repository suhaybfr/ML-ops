# ============================================================
#  Dockerfile — Student Performance Predictor (FastAPI + SVM)
# ============================================================
# Stage 1: Build environment with all dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install \
        fastapi \
        "uvicorn[standard]" \
        httpx \
        prometheus_client

# ============================================================
# Stage 2: Lean runtime image
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="MLOps Team <https://github.com/suhaybfr/ML-ops>" \
      org.opencontainers.image.title="student-performance-predictor" \
      org.opencontainers.image.description="RBF-SVM model served via FastAPI" \
      org.opencontainers.image.source="https://github.com/suhaybfr/ML-ops"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/svm_model.pkl \
    PORT=8000

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/           ./app/
COPY src/           ./src/
COPY models/        ./models/
COPY params.yaml    ./params.yaml

# Create non-root user for security
RUN addgroup --system mlops && adduser --system --ingroup mlops mlopsuser \
    && chown -R mlopsuser:mlops /app

USER mlopsuser

EXPOSE 8000

# Health check — Docker will mark container unhealthy if /health fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
