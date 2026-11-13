# Infographix — Frontend Architecture

## 1. Overview
The frontend is a focused **React + TypeScript + Vite** application using **Konva (`react-konva`)** and **Zustand** with two core routes:

1. **`/` (`CreatePosterPage.tsx`)**:
   - Poster details form: title, authors, affiliations, orientation (`landscape` / `portrait`), preferred columns (`Auto` / `2` / `3` / `4`), accent color, and AI engine selector (`Retrieval + HiGHS LP` vs. `Neural Layout Model`).
   - Dynamic content blocks (`section`, `figure` with optional client-side image upload & automatic aspect ratio calculation, and `table` with rows/columns).
   - Recent Supabase runs sidebar to reopen past runs and a direct `.jsonl` training-data download button.
2. **`/editor/:runId` (`EditorPage.tsx`)**:
   - Full interactive Konva canvas (`PosterCanvas.tsx`) with drag, resize, rotate, snap-to-grid, alignment guides, undo/redo, layers panel, and element properties (`EditorPanels.tsx`).
   - MLOps provenance pill showing engine, retrieved template ID, quality score, and inference latency.
   - Auto-saves canvas modifications back to Supabase (`PATCH /api/v1/runs/:runId`) so human edits are captured in `edited_layout`.
   - Quick 1–5 star rating control for dataset labeling.
   - Direct **PNG** download (clean 2x Konva stage snapshot with selection overlays hidden) and **PDF** download (via `POST /api/v1/export/pdf`).

---

## 2. Directory Structure

```text
frontend/src/
├── App.tsx                       # Minimal router: `/` and `/editor/:runId`
├── main.tsx                      # Entry point
├── index.css                     # Application & editor styling
├── types/index.ts                # Shared TypeScript interfaces
├── stores/editorStore.ts         # Zustand canvas, undo/redo, and stage export store
├── services/
│   ├── apiClient.ts              # Axios client (VITE_API_BASE_URL || "/api")
│   ├── assets.ts                 # Client-side image FileReader + aspect ratio helper
│   └── posters.ts                # /v1/generate, /v1/runs, /v1/export/pdf API methods
├── components/editor/
│   ├── PosterCanvas.tsx          # Konva Stage + clean overlay layer for exports
│   └── EditorPanels.tsx          # Left tools/layers sidebar & right properties inspector
└── pages/
    ├── CreatePosterPage.tsx      # Input form + recent runs + training data export
    └── EditorPage.tsx            # Interactive editor, auto-save edits, PNG/PDF export
```
