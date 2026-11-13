# Infographix — Backend Architecture (MLOps Inference & Telemetry)

## 1. Purpose
The backend is a streamlined **FastAPI** inference and MLOps logging service designed to test Infographix layout models end-to-end and collect `(input, model_output, human_edited_output)` pairs in **Supabase (PostgreSQL)** for future model retraining.

Authentication, multi-user projects, and server-side document/asset storage have been intentionally removed so the service remains stateless, lightweight, and easy to deploy.

---

## 2. Directory Structure

```text
backend/
├── Dockerfile
├── .env.example
├── requirements.txt
├── app/
│   ├── main.py                      # FastAPI entrypoint, CORS, lifespan table creation
│   ├── core/
│   │   └── config.py                # Pydantic settings (DATABASE_URL, CORS, artifact paths)
│   ├── db/
│   │   ├── base.py                  # SQLAlchemy DeclarativeBase
│   │   ├── session.py               # Engine & SessionLocal (PostgreSQL/Supabase & SQLite)
│   │   └── init_db.py               # Standalone schema initialization script
│   ├── models/
│   │   ├── __init__.py
│   │   └── generation_run.py        # GenerationRun SQLAlchemy model (`generation_runs`)
│   ├── schemas/
│   │   └── poster.py                # Pydantic request/response schemas
│   ├── services/
│   │   ├── layout_pipeline.py       # ML inference wrapper (Retrieval+LP & Neural Model)
│   │   └── pdf_export_service.py    # Pillow PNG-to-PDF export converter
│   └── api/
│       └── v1/
│           ├── router.py            # /api/v1 router aggregation
│           ├── health.py            # GET /health (DB + model readiness)
│           └── generation.py        # /generate, /runs, /runs/{id}, /export/pdf
└── tests/
    ├── conftest.py                  # Isolated temporary SQLite fixture
    └── test_generation_mlops.py     # Inference, telemetry, export, and health tests
```

---

## 3. Remote Inference Delegation (`LayoutPipelineService`)

`backend/app/services/layout_pipeline.py` delegates layout generation over HTTP to the standalone ClearML-backed `inference_service` configured via `INFERENCE_SERVICE_URL`:

1. Strips large client-side `imageDataUrl` blobs before sending `POST {INFERENCE_SERVICE_URL}/v1/layouts/generate` (retaining `aspectRatio` for layout geometry).
2. Re-attaches any client figure images onto the returned `figure` Konva slots locally before returning `PosterDocument` to the frontend.
3. Persists the sanitized input payload, engineered `request_features`, ClearML model provenance, `generated_layout`, and subsequent human edits (`edited_layout`) in Supabase PostgreSQL (`generation_runs`).

---

## 4. MLOps Data Loop

Every generation request creates a row in `generation_runs` containing:
- Sanitized raw user input (`input_payload`) and engineered `PosterRequest` features (`request_features`)
- Model provenance (`engine`, `model_name`, `model_version`, `retrieved_template_id`, `candidate_templates`, `generation_metrics`)
- Initial AI prediction (`generated_layout`)
- Subsequent user edits from the Konva editor (`edited_layout`, `was_edited=True`), export actions (`exported_formats`), and optional rating (`feedback_rating`, `feedback_notes`)

Collected pairs can be exported directly via `GET /api/v1/runs/export/training-data?format=jsonl`.
