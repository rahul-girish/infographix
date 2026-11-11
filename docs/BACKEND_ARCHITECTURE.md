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

## 3. Inference Pipeline (`LayoutPipelineService`)

`backend/app/services/layout_pipeline.py` lazily loads artifacts from the repository root:

1. **Candidate Template Retrieval (`models/registry/poster_layout_retriever/v2.0.0/pipeline.joblib`)**:
   - Converts `PosterGenerateRequest` (title, authors, orientation, columns, and content blocks with word counts, aspect ratios, and table dimensions) into a `PosterRequest` (`src/features/request_schema.py`).
   - Queries `LayoutRetriever.retrieve_for_request(..., top_k=5, diversify=True)`.
2. **Engine Selection (`engine`)**:
   - **`adaptation` (default)**: Uses `LayoutAdapter` (`src/adaptation/layout_adapter.py`) + `data/processed/features/v1/elements.parquet` to solve a HiGHS linear program (`scipy.optimize.linprog`) guaranteeing non-overlapping, typography-aware bounding boxes.
   - **`neural`**: Encodes user blocks into `[1, M, 18]` tensors (`UserContentItem`) and the top-1 retrieved template into `[1, K, 16]` tensors (`TemplateSlotItem`), then runs `TemplateConditionedLayoutModel` (`models/checkpoints/neural_layout/best_model.pt`).
3. **Canvas Mapping**:
   - Maps generated elements into Konva-compatible `PosterElement` dictionaries (`x`, `y`, `width`, `height`, styling, plus `sourceBlockId`, `semanticRole`, and `bboxNorm` provenance).

---

## 4. MLOps Data Loop

Every generation request creates a row in `generation_runs` containing:
- Sanitized raw user input (`input_payload`) and engineered `PosterRequest` features (`request_features`)
- Model provenance (`engine`, `model_name`, `model_version`, `retrieved_template_id`, `candidate_templates`, `generation_metrics`)
- Initial AI prediction (`generated_layout`)
- Subsequent user edits from the Konva editor (`edited_layout`, `was_edited=True`), export actions (`exported_formats`), and optional rating (`feedback_rating`, `feedback_notes`)

Collected pairs can be exported directly via `GET /api/v1/runs/export/training-data?format=jsonl`.
