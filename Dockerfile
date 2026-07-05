# ──────────────────────────────────────────────────────────────────────────────
# Docker — MLOps Platform (Ray-Integrated)
# ──────────────────────────────────────────────────────────────────────────────
# Multi-stage build: slim final image with Ray, MLflow, and runtime deps
#
# Build:  docker build -t mlops-platform-ray .
# Run:    docker run --rm -v mlops-data:/app/data -p 5000:5000 mlops-platform-ray
# Shell:  docker run --rm -it mlops-platform-ray bash
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Sushant Shambharkar <sam1064max@gmail.com>"
LABEL description="MLOps Platform (Ray) — 10K models, 750M records, Ray Tune + Actors"
LABEL version="2.0.0"

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=sqlite:///app/data/mlflow/mlflow.db \
    MLFLOW_ARTIFACT_ROOT=/app/data/mlflow/artifacts \
    PIPELINE_ENV=docker \
    RAY_OBJECT_STORE_MEMORY=1000000000

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config/ config/
COPY src/ src/
COPY orchestration/ orchestration/
COPY monitoring/ monitoring/
COPY tests/ tests/
COPY pyproject.toml .
COPY requirements.txt .
COPY README.md .

# Create data directories
RUN mkdir -p data/bronze data/silver data/gold \
    data/inference/input data/inference/output \
    data/monitoring data/mlflow/artifacts

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops -d /app mlops \
    && chown -R mlops:mlops /app
USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import ray; import src.utils; print('OK')" || exit 1

# Default: run full pipeline
ENTRYPOINT ["python", "-m", "orchestration.pipeline_runner"]

# Override entrypoint for other commands:
#   docker run mlops-platform-ray python -m pytest tests/ -v
#   docker run mlops-platform-ray mlflow ui --host 0.0.0.0
