# Frontend test report

## Automated checks

| Check | Result |
|---|---|
| `cd frontend && npm run build` | Passed: TypeScript project build and Vite production bundle |
| `cd frontend && npm run lint` | Passed: ESLint |

The production build reports a non-fatal bundle-size warning: the main JavaScript chunk is approximately 694 kB minified (about 211 kB gzip), above Vite's 500 kB advisory threshold. React-Konva and Konva contribute significantly; route-level code splitting can be considered if bundle performance becomes a priority.

## Build issue resolved

The initial build failed because `App.tsx` imported missing login and registration page modules. Authentication now uses the shared `AuthPage` with explicit login, registration and forgot-password routes. The final TypeScript/Vite build completes.

## Coverage and limitations

There is no unit/integration test runner configured in the frontend package at handover. Build and lint checks do not replace browser interaction testing. The browser checks below exercise the UI against the local mock services; document extraction, account recovery and submission are not validated against a backend.

## Browser smoke checks

Using the local Vite server in a desktop-sized browser viewport:

- Loaded the landing page and confirmed the product sections and route links rendered.
- Signed in with demo values; the dashboard loaded and created a project.
- Continued through the no-file path, opened content/configuration and selected portrait orientation.
- Opened the Konva editor, verified poster canvas/layers, added a heading, exercised Ctrl+Z/Ctrl+Y, and reloaded to confirm the new layer persisted.
- Shift-selected two layer rows, grouped and ungrouped them from the properties panel.
- Duplicated a project, verified the copy title, and used the dashboard delete confirmation to remove the test projects.
- Opened preview and submitted; the UI changed to `Submitted`.
- Uploaded a mock `.pdf` fixture; simulated completion and accepted state appeared. A `.png` selection displayed the PDF/DOCX validation error.
- Confirmed registration and forgot-password routes render their forms.

These are manual browser smoke checks, not a substitute for automated component/service tests or backend integration tests. A small temporary upload fixture was used from the session workspace and removed after testing.
