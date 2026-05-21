# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p /app/storage/data /app/storage/runs

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

ENV DATABASE_URL="sqlite:////app/storage/energy_predictor.db" \
    DATA_DIR="/app/storage/data" \
    RUNS_DIR="/app/storage/runs" \
    CORS_ORIGINS="http://localhost,http://localhost:80,http://localhost:3000" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Suppress CUDA probe — container has no GPU, avoids "UNKNOWN ERROR (303)" log spam
    CUDA_VISIBLE_DEVICES="" \
    # Suppress TF C++ INFO/WARNING logs (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
    TF_CPP_MIN_LOG_LEVEL="2" \
    # Suppress XLA/CUDA init noise
    XLA_FLAGS="--xla_gpu_cuda_data_dir=/dev/null" \
    TF_ENABLE_ONEDNN_OPTS="0"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
