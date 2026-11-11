FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8001 \
    HOME=/home/appuser \
    CLEARML_CACHE_DIR=/home/appuser/.clearml

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first to avoid pulling multi-GB CUDA wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch>=2.3.0

COPY inference_service/requirements.txt ./inference_service/requirements.txt
RUN pip install --no-cache-dir -r inference_service/requirements.txt

# Copy only shared ML modules and inference service code (no checkpoints, datasets, or .env)
COPY src/ ./src/
COPY inference_service/ ./inference_service/

# Remove any test files or dev artifacts that should not ship in runtime layers
RUN rm -rf inference_service/tests inference_service/.env inference_service/.env.* \
    src/test_*.py src/**/__pycache__ inference_service/**/__pycache__

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data_cache /home/appuser/.clearml \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8001

CMD ["sh", "-c", "exec uvicorn inference_service.app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
