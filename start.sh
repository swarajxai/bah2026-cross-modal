#!/bin/bash
# HF Spaces startup script
# Downloads weights then starts gunicorn

set -e
echo "==========================================="
echo "[startup] BAH 2026 Cross-Modal Retrieval"
echo "==========================================="
echo "[startup] $(date)"

# 1. Ensure output directory exists (may be wiped on restart)
mkdir -p /app/outputs
chmod 777 /app/outputs

# 2. Download model weights from HF Hub
echo "[startup] downloading model weights..."
python -m webapp.download_weights || echo "[startup] WARNING: download had errors, continuing anyway"

# 3. Verify weights (don't fail if missing — gunicorn will report error)
echo "[startup] verifying outputs..."
ls -la /app/outputs/ 2>/dev/null || echo "[startup] no outputs dir"

# 3. Start gunicorn in foreground (HF Spaces requires this)
echo "[startup] starting gunicorn..."
exec gunicorn \
    --bind 0.0.0.0:${PORT:-7860} \
    --workers 1 \
    --threads 4 \
    --timeout 180 \
    --graceful-timeout 60 \
    --access-logfile - \
    --error-logfile - \
    webapp.app:app