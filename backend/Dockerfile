FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install backend + ML inference dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend API, ML pipeline source, and model artifacts
COPY backend/ ./backend/
COPY src/ ./src/
COPY models/registry/ ./models/registry/
COPY models/checkpoints/neural_layout/best_model.pt ./models/checkpoints/neural_layout/best_model.pt
COPY data/processed/features/v1/elements.parquet ./data/processed/features/v1/elements.parquet

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
