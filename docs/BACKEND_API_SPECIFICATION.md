# Infographix — Backend API Specification

**Base Paths:** `/api/v1` (also mounted at `/api` for frontend proxy convenience)  
**Interactive Docs:** `GET /docs` (Swagger UI) and `GET /redoc`

---

## 1. Health & Readiness

### `GET /api/v1/health`
Checks database connectivity (Supabase PostgreSQL or local SQLite) and ML artifact status.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "database": "connected",
  "models": {
    "retrieverLoaded": true,
    "retrieverVersion": "v2.0.0",
    "adapterReady": true,
    "neuralCheckpointExists": true,
    "neuralLoaded": false
  },
  "version": "2.0.0"
}
```

---

## 2. Poster Generation & MLOps Runs

### `POST /api/v1/generate`
Runs the AI layout model on user content, persists the run in `generation_runs`, and returns the editable `PosterDocument`.

**Request Body (`PosterGenerateRequest`):**
```json
{
  "title": "Constraint-Aware Layout Synthesis for Scientific Posters",
  "authors": "Elena Rostova, Marcus Vance",
  "affiliations": "Infographix Research Lab",
  "orientation": "landscape",
  "preferredColumns": 3,
  "accentColor": "#3458e8",
  "background": "#ffffff",
  "engine": "adaptation",
  "blocks": [
    {
      "id": "intro",
      "type": "section",
      "heading": "1. Introduction",
      "text": "Research motivation and core contributions..."
    },
    {
      "id": "fig_1",
      "type": "figure",
      "heading": "Architecture",
      "aspectRatio": 1.45,
      "caption": "Figure 1: Pipeline overview."
    }
  ]
}
```

**Response `201 Created` (`GenerationRunOut`):**
Returns `id`, `createdAt`, `updatedAt`, `engine`, `modelName`, `modelVersion`, `retrievedTemplateId`, `candidateTemplates`, `generationMetrics`, `poster` (`PosterDocumentSchema`), `wasEdited`, `exportedFormats`, and `feedbackRating`.

---

### `GET /api/v1/runs`
Lists recent runs for quick inspection or reopening in the editor.

**Query Params:**
- `limit` (int, `1..200`, default `25`)
- `edited_only` (bool, default `false`)

---

### `GET /api/v1/runs/{run_id}`
Fetches a single run by UUID, returning `edited_layout` if present or `generated_layout` otherwise.

---

### `PATCH /api/v1/runs/{run_id}`
Logs human edits from the Konva editor, export events, or quality feedback into Supabase.

**Request Body (`RunUpdateRequest`):**
```json
{
  "editedLayout": { "...": "PosterDocument JSON" },
  "exportedFormat": "png",
  "feedbackRating": 5,
  "feedbackNotes": "Clean 3-column alignment"
}
```

---

### `GET /api/v1/runs/export/training-data`
Exports collected MLOps training records (`input_payload`, `request_features`, `generated_layout`, `edited_layout`, `was_edited`, `generation_metrics`, `feedback_rating`).

**Query Params:**
- `format`: `jsonl` (default, downloadable `.jsonl`) or `json`
- `edited_only`: bool (default `false`) — filter to human-corrected runs
- `limit`: int (`1..5000`, default `500`)

---

## 3. PDF Export

### `POST /api/v1/export/pdf`
Converts the high-resolution Konva canvas PNG snapshot into a downloadable single-page PDF (`application/pdf`) and appends `"pdf"` to `exported_formats` on the associated run if `runId` is provided.

**Request Body (`PdfExportRequest`):**
```json
{
  "title": "My Poster",
  "width": 1000,
  "height": 700,
  "imageDataUrl": "data:image/png;base64,...",
  "runId": "optional-uuid"
}
```
