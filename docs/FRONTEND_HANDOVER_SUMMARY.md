# Infographix frontend handover

## Delivered

- Responsive landing page, supported-format callout, template gallery, authentication and dashboard.
- Create/upload/review/configure workflow with editable research sections and five poster templates.
- Three-panel React-Konva editor with text, image, shape, line and arrow elements; properties, layer ordering, visibility/lock controls, multi-select/grouping, snapping/alignment guides, grid, zoom, fit, copy/paste, undo/redo and autosave.
- Structured poster JSON, browser-local project CRUD and reopen/duplicate/rename/delete/status flows.
- Read-only preview, SVG export, print and mock submit behavior.
- Separate mock service modules plus configurable Axios base URL.

## Run locally

```powershell
cd frontend
npm run dev
```

Set `VITE_API_BASE_URL` using `frontend/.env.example` when a backend is ready. Current services do not make HTTP requests.

## Important mock boundaries

- Authentication accepts demo credentials and stores only a user identity in session storage. Passwords are not stored.
- PDF/DOCX upload validates extension and size and simulates progress; it does not transfer or extract document contents.
- Projects and structured poster JSON persist in local storage. Browser quota may limit larger projects.
- Image elements embed data URLs and are limited to 1 MB each in this demo.
- SVG export outputs the individual poster elements; preview submission only changes local project status.
- No FastAPI endpoints, database access or AI extraction/generation are claimed as implemented.

## Verification

See `FRONTEND_TEST_REPORT.md` for the executed build/lint checks and browser smoke-test results. The only build warning is the advisory large main JavaScript bundle from the editor dependencies.
