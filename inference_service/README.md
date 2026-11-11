# Infographix Standalone Inference Service

Independent, Dockerized **FastAPI** inference microservice for Infographix. It loads versioned layout models directly from the **ClearML Model Registry**, caches them in memory at startup, and serves both:
- **Engine A (`adaptation`)**: `LayoutRetriever` + `LayoutAdapter` + SciPy HiGHS linear programming optimization (`src/models/layout_retrieval.py`, `src/adaptation/`).
- **Engine B (`neural`)**: `LayoutRetriever` (top-1 conditioning template) + `TemplateConditionedLayoutModel` PyTorch inference (`src/neural_layout/`).

The service is stateless and performs **no** database writes (Supabase persistence remains owned by the application backend).

---

## 1. Architecture

```text
Application Backend ──HTTP──▶ inference_service (FastAPI)
                                ├── Startup Lifespan:
                                │     1. Authenticate ClearML via env vars
                                │     2. load_registered_retriever() + SHA-256 verify
                                │     3. load_registered_model() + SHA-256 verify
                                │     4. Load & verify elements.parquet
                                └── Request Handling (thread-safe):
                                      ├── GET  /health/live
                                      ├── GET  /health/ready
                                      ├── GET  /v1/models
                                      └── POST /v1/layouts/generate
```

All ML model code, feature engineering, schemas, and ClearML registry verification are imported directly from `src/` without duplication.

---

## 2. Environment Variables

Copy `inference_service/.env.example` to `inference_service/.env` for local runs (never commit `.env`):

| Variable | Required | Description |
| :--- | :---: | :--- |
| `CLEARML_API_ACCESS_KEY` | Yes | ClearML API access key |
| `CLEARML_API_SECRET_KEY` | Yes | ClearML API secret key |
| `CLEARML_API_HOST` | No | Default `https://api.clear.ml` |
| `CLEARML_WEB_HOST` | No | Default `https://app.clear.ml` |
| `CLEARML_FILES_HOST` | No | Default `https://files.clear.ml` |
| `CLEARML_PROJECT_NAME` | No | ClearML project (`Infographix`) |
| `CLEARML_FORCE_DOWNLOAD` | No | Force re-download from ClearML cache (`false`) |
| `RETRIEVER_MODEL_FAMILY` | No | Default `poster_layout_retriever` |
| `RETRIEVER_MODEL_VERSION` | Yes* | Explicit semantic version (e.g. `v2.0.0`) |
| `RETRIEVER_MODEL_ID` | Yes* | Explicit ClearML model ID (overrides version) |
| `RETRIEVER_LIFECYCLE_STAGE` | No | Optional unambiguous stage selector (fails if 0 or >1 matches) |
| `NEURAL_MODEL_FAMILY` | No | Default `poster_layout_generator` |
| `NEURAL_MODEL_VERSION` | Yes* | Explicit semantic version (e.g. `v0.1.0`) |
| `NEURAL_MODEL_ID` | Yes* | Explicit ClearML model ID (overrides version) |
| `NEURAL_LIFECYCLE_STAGE` | No | Optional unambiguous stage selector |
| `DEFAULT_ENGINE` | No | `neural` or `adaptation` (default `neural`) |
| `TORCH_DEVICE` | No | `cpu` (default) |
| `ELEMENTS_PARQUET_PATH` | Yes | Local/cached path to `elements.parquet` (default `/app/data_cache/elements.parquet`) |
| `AWS_ACCESS_KEY_ID` | If S3 | AWS access key ID for private S3 bucket |
| `AWS_SECRET_ACCESS_KEY` | If S3 | AWS secret access key for private S3 bucket |
| `AWS_DEFAULT_REGION` | If S3 | AWS region (e.g. `us-east-1`) |
| `S3_BUCKET_NAME` | If S3 | Private S3 bucket name (required with `S3_OBJECT_KEY`) |
| `S3_OBJECT_KEY` | If S3 | S3 object key (e.g. `data/processed/features/v1/elements.parquet`) |
| `ELEMENTS_PARQUET_URL` | No | HTTPS URL to download `elements.parquet` if not cached |
| `ELEMENTS_PARQUET_SHA256` | No | Expected SHA-256 hex digest for `elements.parquet` |
| `CLEARML_DATASET_ID` / `CLEARML_DATASET_NAME` | No | Optional ClearML Dataset source for `elements.parquet` |
| `MINIDVC_SNAPSHOT` | No | Optional Minidvc snapshot name (e.g. `v2`) if `minidvc` CLI is present |
| `PORT` | No | HTTP listen port (default `8001`) |
| `CORS_ORIGINS` | No | JSON list of allowed origins |

*\*Either `*_MODEL_ID` or `*_MODEL_VERSION` (or an unambiguous `*_LIFECYCLE_STAGE`) must be set. The service refuses to silently load `latest` or fall back to local checkpoints.*

---

## 3. ClearML Model Selection & Initialization Behavior

During FastAPI startup (`lifespan` in `inference_service/app/main.py`):
1. `configure_clearml_credentials` validates `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` and calls `Task.set_credentials(...)`.
2. `load_clearml_retriever` calls `src.registry.clearml_registry.load_registered_retriever(...)`, downloads the ClearML bundle, verifies all SHA-256 checksums in `manifest.json`, confirms `architecture == "LayoutRetriever"`, validates train-only fitting (`_validate_fitted_retriever`), and verifies schema columns.
3. `load_clearml_neural_model` calls `src.registry.clearml_registry.load_registered_model(...)`, verifies SHA-256 checksums, confirms `architecture == "TemplateConditionedLayoutModel"`, checks feature dimensions `(18, 16)`, and sets `.eval()` mode.
4. `load_elements_dataframe` loads and validates `elements.parquet`.
5. Only after all steps succeed does `/health/ready` transition to HTTP `200` (`status: "ready"`). If any step fails, `/health/ready` returns HTTP `503` with a sanitized error message (credentials redacted).

Models are loaded **once** into memory at startup. The service never registers or promotes models at runtime.

---

## 4. Adaptation & Template Data Requirement (`elements.parquet`)

Both engines require `elements.parquet` (containing `poster_id`, `element_id`, `category_slug`, `x0`, `y0`, `w_norm`, `h_norm`, `cx`, `cy`, `area_norm`, `raw_x`, `raw_y`, `raw_w`, `raw_h`):
- **Adaptation Engine**: `LayoutAdapter` reads the retrieved template poster's element rows to build a hierarchical column/section layout for HiGHS optimization.
- **Neural Engine**: Encodes the retrieved template poster's element rows into `[1, K, 16]` `TemplateSlotItem` tensors.

Because large datasets are excluded from the Docker image and Git, `data_loader.py` resolves `elements.parquet` strictly in this precedence order (without silent fallback if a selected remote source fails):
1. **Local / Cached File (`ELEMENTS_PARQUET_PATH`)**: If the file already exists on disk (persistent disk, local dev, or warm cache), verifies `ELEMENTS_PARQUET_SHA256` (if set) and loads it without re-downloading.
2. **Private AWS S3 (`S3_BUCKET_NAME` + `S3_OBJECT_KEY`)**: Downloads via `boto3` using `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` to a temp file in the cache directory, verifies `ELEMENTS_PARQUET_SHA256`, and atomically promotes it via `os.replace`. Both `S3_BUCKET_NAME` and `S3_OBJECT_KEY` are required when S3 is selected.
3. **HTTPS URL (`ELEMENTS_PARQUET_URL`)**: Streams download to a temp file, verifies `ELEMENTS_PARQUET_SHA256` (if set), and atomically promotes it.
4. **ClearML Dataset (`CLEARML_DATASET_ID` or `CLEARML_DATASET_NAME`)**: Downloads via `clearml.Dataset.get(...)` and verifies checksum.
5. **Minidvc (`MINIDVC_SNAPSHOT`)**: Pulls and checks out via `minidvc` CLI and verifies checksum.

---

## 5. Running Locally

From the repository root:

```bash
# 1. Configure environment variables (or export them directly)
cp inference_service/.env.example inference_service/.env
# Fill in CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, model versions, and ELEMENTS_PARQUET_PATH

# 2. Start the inference service
PYTHONPATH=. .venv/bin/uvicorn inference_service.app.main:app --host 0.0.0.0 --port 8001
```

### Run Automated Tests (No ClearML Credentials Required)
```bash
PYTHONPATH=. .venv/bin/python -m unittest inference_service/tests/test_inference_service.py -v
```

---

## 6. Building & Running with Docker

Build from the **repository root** so `src/` and `inference_service/` are in context (while `.dockerignore` / `Dockerfile.dockerignore` exclude checkpoints, datasets, `.venv`, and `.env`):

```bash
docker build -f inference_service/Dockerfile -t infographix-inference:latest .

docker run --rm -p 8001:8001 \
  -e CLEARML_API_ACCESS_KEY="your-key" \
  -e CLEARML_API_SECRET_KEY="your-secret" \
  -e RETRIEVER_MODEL_VERSION="v2.0.0" \
  -e NEURAL_MODEL_VERSION="v0.1.0" \
  -e ELEMENTS_PARQUET_PATH="/app/data_cache/elements.parquet" \
  -e ELEMENTS_PARQUET_URL="https://your-storage/elements.parquet" \
  -e ELEMENTS_PARQUET_SHA256="4a491581bd20a770fd244e5dfcce7a943ba2be97e368c11dc7dc8cf96bf0d651" \
  infographix-inference:latest
```

Or via Docker Compose:
```bash
cd inference_service
docker compose up --build
```

---

## 7. API Endpoints

### `GET /health/live`
Liveness check (`200 OK`):
```json
{
  "status": "alive",
  "service": "Infographix Inference Service",
  "version": "1.0.0"
}
```

### `GET /health/ready`
Readiness check (`200 OK` when initialized, `503 Service Unavailable` otherwise):
```json
{
  "status": "ready",
  "model_initialized": true,
  "retriever_initialized": true,
  "data_dependency_ready": true,
  "selected_models": {
    "retriever": {
      "model_name": "poster_layout_retriever-v2.0.0",
      "clearml_model_id": "...",
      "semantic_version": "2.0.0"
    },
    "neural_generator": {
      "model_name": "poster_layout_generator-v0.1.0",
      "clearml_model_id": "...",
      "semantic_version": "0.1.0"
    }
  },
  "supported_engines": ["adaptation", "neural"],
  "error": null
}
```

### `GET /v1/models`
Returns actual loaded ClearML models, SHA-256 digests, architectures, and dataset metadata.

### `POST /v1/layouts/generate`
**Request:**
```json
{
  "request": {
    "title": "Constraint-Aware Layout Synthesis for Scientific Posters",
    "authors": ["Elena Rostova", "Marcus Vance"],
    "affiliations": ["Infographix Research Lab"],
    "orientation": "landscape",
    "columns": 3,
    "content_blocks": [
      {
        "id": "intro",
        "type": "section",
        "heading": "1. Introduction",
        "text": "Research motivation and problem framing..."
      },
      {
        "id": "fig1",
        "type": "figure",
        "heading": "System Architecture",
        "aspectRatio": 1.45,
        "caption": "Figure 1: Overview."
      }
    ]
  },
  "engine": "neural"
}
```

**Response (`200 OK`):**
```json
{
  "engine": "neural",
  "model_name": "poster_layout_generator",
  "model_version": "0.1.0",
  "model_id": "...",
  "retriever_version": "2.0.0",
  "retriever_model_id": "...",
  "retrieved_template_id": "3449768008",
  "candidate_templates": [],
  "generation_metrics": {
    "predicted_macro_columns": 3,
    "num_elements": 7,
    "raw_overflow_rate": 0.0,
    "clamped_boxes_count": 0,
    "validated_overlap_area": 0.0,
    "latency_ms": 42.15
  },
  "request_features": {},
  "layout": {
    "id": "...",
    "title": "Constraint-Aware Layout Synthesis for Scientific Posters",
    "width": 1000,
    "height": 700,
    "orientation": "landscape",
    "background": "#ffffff",
    "accentColor": "#3458e8",
    "columns": 3,
    "elements": [],
    "updatedAt": "..."
  }
}
```

---

## 8. Deploying on Render

1. Create a new **Web Service** on Render connected to this repository.
2. Configure Docker settings:
   - **Language / Environment:** `Docker`
   - **Root Directory:** `.` *(leave at repository root so `src/` is accessible)*
   - **Dockerfile Path:** `inference_service/Dockerfile`
   - **Docker Build Context Directory:** `.`
   - **Health Check Path:** `/health/ready`
3. Attach a **Persistent Disk** (optional, e.g. mount path `/app/data_cache`) OR set `ELEMENTS_PARQUET_URL` + `ELEMENTS_PARQUET_SHA256` in Render Environment Variables so the container downloads and verifies `elements.parquet` on cold start.
4. Set Render **Environment Variables**:
   - `CLEARML_API_ACCESS_KEY`
   - `CLEARML_API_SECRET_KEY`
   - `CLEARML_API_HOST=https://api.clear.ml`
   - `CLEARML_WEB_HOST=https://app.clear.ml`
   - `CLEARML_FILES_HOST=https://files.clear.ml`
   - `CLEARML_PROJECT_NAME=Infographix`
   - `RETRIEVER_MODEL_VERSION=v2.0.0` *(or exact `RETRIEVER_MODEL_ID`)*
   - `NEURAL_MODEL_VERSION=v0.1.0` *(or exact `NEURAL_MODEL_ID`)*
   - `DEFAULT_ENGINE=neural`
   - `ELEMENTS_PARQUET_PATH=/app/data_cache/elements.parquet`
   - `ELEMENTS_PARQUET_URL=<signed-or-public-object-storage-url>`
   - `ELEMENTS_PARQUET_SHA256=4a491581bd20a770fd244e5dfcce7a943ba2be97e368c11dc7dc8cf96bf0d651`

---

## 9. Updating Model Versions Without Rebuilding Docker

1. Train, evaluate, and register the new model version in ClearML (e.g., `poster_layout_generator-v0.2.0` or `poster_layout_retriever-v2.1.0`).
2. In Render (or your container orchestrator), update `NEURAL_MODEL_VERSION` / `NEURAL_MODEL_ID` or `RETRIEVER_MODEL_VERSION` / `RETRIEVER_MODEL_ID`.
3. Trigger a service restart / deploy without rebuilding the image (`Restart Service` on Render).
4. Verify the active model IDs and versions via `GET /v1/models` and `GET /health/ready`.
