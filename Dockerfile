# SatQuery backend — API + orchestration image.
#
# By default the vision model is DISABLED (VLM_BACKEND=disabled): this image is
# small and CPU-only, suitable when the Qwen2-VL model runs elsewhere or you
# only need the orchestration API. To run the VLM in-container, uncomment the
# ML deps below, use a CUDA base image, and set VLM_BACKEND=local.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- optional: local vision model + geospatial tools ---
# COPY requirements-ml.txt requirements-geo.txt ./
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
#     && pip install --no-cache-dir -r requirements-ml.txt -r requirements-geo.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

ENV VLM_BACKEND=disabled \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "satquery"]
