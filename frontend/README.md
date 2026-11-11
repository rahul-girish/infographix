# Infographix Frontend

Streamlined React + TypeScript + Konva interface for testing the Infographix AI poster layout pipeline and recording human-in-the-loop edits for MLOps retraining.

## Routes
- `/` — Enter poster title, authors, orientation, columns, AI engine (`adaptation` or `neural`), and content blocks, then generate a layout.
- `/editor/:runId` — Interactive Konva canvas editor with auto-save of edits to Supabase (`edited_layout`), 1–5 layout rating, and direct **PNG** / **PDF** download.

## Development
```bash
npm install
npm run dev
```

## Build & Lint
```bash
npm run build
npm run lint
```

Set `VITE_API_BASE_URL` in `.env` when deploying separately from the FastAPI backend (defaults to `/api` via Vite dev proxy).
