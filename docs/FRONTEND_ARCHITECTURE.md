# Infographix frontend architecture

## Runtime and entry points

- React 19, TypeScript, Vite 8, React Router, Tailwind CSS v4, Zustand, Axios and React-Konva run from `frontend/`.
- `src/main.tsx` mounts `src/App.tsx`; the router owns the public landing/auth routes and the project workflow routes.
- The frontend is a static SPA. The current mock workflow can run without an API server.

## Application routes

| Route | Responsibility |
|---|---|
| `/` | Product landing page and template/format overview |
| `/login`, `/register`, `/forgot-password` | Mock authentication and account recovery |
| `/dashboard` | Project search, status filters and project CRUD actions |
| `/projects/:projectId/upload` | PDF/DOCX validation and simulated upload progress |
| `/projects/:projectId/review` | Editable, reorderable, includable research sections |
| `/projects/:projectId/configure` | Orientation, template, columns and background |
| `/projects/:projectId/editor` | React-Konva editor and automatic local save |
| `/projects/:projectId/preview` | Read-only preview, SVG export, print and mock submission |

## Source organization

- `src/pages/` contains route-level screens; authentication modes share `pages/auth/AuthPage.tsx`.
- `src/components/layout/` contains public and workspace navigation.
- `src/components/editor/` separates the Konva canvas from element tools, layers and properties.
- `src/types/index.ts` defines the persisted project, research-section and poster JSON contracts.
- `src/data/templates.ts` contains the five starter templates and independently editable starter layers.
- `src/stores/editorStore.ts` is the transient Zustand editor state, including selection, view settings, undo and redo.
- `src/services/` is the mock/backend seam. `projects.ts` persists projects; the other service modules isolate authentication, documents, extraction, posters, assets and submissions. `apiClient.ts` supplies a configurable Axios instance for later HTTP services.

## Persistence and data flow

Projects and poster JSON are stored under `infographix.projects.v1` in browser local storage. Projects hold editable section text and a `PosterDocument`, whose element array stores element type, geometry, rotation, layer order, visibility, lock state and type-specific properties. Uploaded editor images are data URLs in poster elements. The project service is the single persistence boundary, so replacing it with API calls does not require the pages or canvas to own storage details.

The editor store is in-memory and receives the persisted poster when a project is opened. Changes update that structured document, enter the undo history and are debounced to the project service. Shift-click canvas or layer rows to select multiple elements, then group/ungroup from the properties panel. Preview/export renders the individual elements; it does not flatten the saved poster model.

Mock auth stores only the current user identity in `sessionStorage`. Passwords are accepted by the mock form/service but are not written to browser storage. Document upload and extraction are simulations: the selected paper's contents are not sent to a server or parsed.

## Local development

From `frontend/`, run `npm install` only when dependencies need restoring, then `npm run dev`. Set `VITE_API_BASE_URL` in the environment (see `frontend/.env.example`) to configure the Axios base URL when implementing real HTTP services. The current mock services do not call it.
