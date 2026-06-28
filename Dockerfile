# =====================================================================
#  BAH 2026 — Hugging Face Spaces Docker deployment
# =====================================================================
#
#  CPU-only image optimized for HF Spaces free tier (2 vCPU, 16 GB RAM).
#  Smaller than Render — no apt cleanup tricks, just slim base.
# =====================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/tmp/torch_cache

# OS deps (HF Spaces needs libgl for opencv, libtiff for tifffile, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libtiff6 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch — HF Spaces free tier has no CUDA
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/requirements.txt

# Copy application code (datasets excluded via .dockerignore)
COPY src/           /app/src/
COPY webapp/        /app/webapp/
COPY outputs/       /app/outputs/

# Sanity check that model files are present
RUN ls -la /app/outputs/

# HF Spaces uses port 7860 by default
EXPOSE 7860
ENV PORT=7860

# Ensure huggingface_hub available (for runtime weight download)
RUN pip install --no-cache-dir huggingface_hub

# Run gunicorn on port 7860 (HF Spaces standard)
# Download model weights from HF Hub first, then start server
CMD exec python -m webapp.download_weights && \
        gunicorn \
        --bind 0.0.0.0:${PORT} \
        --workers 1 \
        --threads 4 \
        --timeout 180 \
        --graceful-timeout 60 \
        --access-logfile - \
        --error-logfile - \
        webapp.app:app