# Infographix — Database Design (Supabase PostgreSQL)

## 1. Overview
The database is intentionally minimal: a single **`generation_runs`** table in Supabase PostgreSQL (using `JSONB` for structured payloads and layouts, with an automatic `JSON` fallback for local SQLite tests).

Its purpose is **MLOps observability and training data collection**:
1. Record every user input (`input_payload`) and engineered model input (`request_features`).
2. Record model provenance (`engine`, `model_name`, `model_version`, `retrieved_template_id`, `candidate_templates`, `generation_metrics`).
3. Store both the raw AI layout (`generated_layout`) and any human corrections made in the Konva editor (`edited_layout`, `was_edited`) so edited runs form supervised preference/fine-tuning pairs.

---

## 2. Table Schema: `generation_runs`

Defined in [`backend/app/models/generation_run.py`](../backend/app/models/generation_run.py):

| Column | Type (PostgreSQL) | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` (PK) | No | Unique run identifier |
| `created_at` | `TIMESTAMPTZ` (indexed) | No | Inference timestamp (UTC) |
| `updated_at` | `TIMESTAMPTZ` | No | Last edit/export timestamp (UTC) |
| `title` | `VARCHAR(255)` | No | Poster title |
| `authors` | `VARCHAR(500)` | No | Author string |
| `orientation` | `VARCHAR(32)` | No | `landscape` or `portrait` |
| `engine` | `VARCHAR(64)` | No | `adaptation` or `neural` |
| `model_name` | `VARCHAR(128)` | No | Model identifier |
| `model_version` | `VARCHAR(64)` | No | Semantic version |
| `retrieved_template_id` | `VARCHAR(64)` | Yes | Selected SciPostLayout template `poster_id` |
| `candidate_templates` | `JSONB` | No | Top-$k$ retrieved template IDs and relevance scores |
| `generation_metrics` | `JSONB` | No | Quality score, overflow/overlap, latency (`latency_ms`) |
| `input_payload` | `JSONB` | No | Sanitized `PosterGenerateRequest` (inline image blobs stripped) |
| `request_features` | `JSONB` | No | `PosterRequest.to_summary_dict()` + per-block demand features |
| `generated_layout` | `JSONB` | No | Raw AI-predicted `PosterDocument` with `bboxNorm` |
| `edited_layout` | `JSONB` | Yes | Human-edited `PosterDocument` synced from Konva editor |
| `was_edited` | `BOOLEAN` (indexed) | No | `true` once user edits the generated layout |
| `exported_formats` | `JSONB` | No | List of exported formats (e.g. `["png", "pdf"]`) |
| `feedback_rating` | `INTEGER` | Yes | Optional `1..5` layout quality rating |
| `feedback_notes` | `TEXT` | Yes | Optional qualitative feedback |

---

## 3. Notes on Inline Images
When a user attaches an image from disk in the frontend, the browser computes its exact `aspectRatio` for the AI model and keeps the data URL locally for Konva rendering/export. Before persisting `input_payload`, `generated_layout`, and `edited_layout` to Supabase, the backend replaces inline `data:` image strings with `[inline-image-omitted]` so database rows remain small and fast to query/export.
