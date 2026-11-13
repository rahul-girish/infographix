import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Check,
  Download,
  FileDown,
  Grid2X2,
  Loader2,
  Minus,
  Plus,
  Redo2,
  Save,
  Star,
  Undo2,
} from "lucide-react";
import { EditorProperties, EditorSidebar } from "../components/editor/EditorPanels";
import PosterCanvas from "../components/editor/PosterCanvas";
import { postersService } from "../services/posters";
import { useEditorStore } from "../stores/editorStore";
import type { PosterElement } from "../types";

export default function EditorPage() {
  const { runId = "" } = useParams();
  const navigate = useNavigate();

  const run = useEditorStore((state) => state.run);
  const document = useEditorStore((state) => state.document);
  const selectedId = useEditorStore((state) => state.selectedId);
  const zoom = useEditorStore((state) => state.zoom);
  const undoStack = useEditorStore((state) => state.undoStack);
  const redoStack = useEditorStore((state) => state.redoStack);
  const showGrid = useEditorStore((state) => state.showGrid);
  const snapToGrid = useEditorStore((state) => state.snapToGrid);
  const stageExporter = useEditorStore((state) => state.stageExporter);
  const setRun = useEditorStore((state) => state.setRun);
  const setRunMeta = useEditorStore((state) => state.setRunMeta);
  const setZoom = useEditorStore((state) => state.setZoom);
  const undo = useEditorStore((state) => state.undo);
  const redo = useEditorStore((state) => state.redo);

  const [loading, setLoading] = useState(() =>
    Boolean(runId && (useEditorStore.getState().run?.id !== runId || !useEditorStore.getState().document))
  );
  const [error, setError] = useState("");
  const [saveLabel, setSaveLabel] = useState("Synced to Supabase");
  const [exportingPdf, setExportingPdf] = useState(false);

  const canvasAreaRef = useRef<HTMLDivElement>(null);
  const copyBuffer = useRef<PosterElement | null>(null);
  const loadedRunId = useRef(
    runId && useEditorStore.getState().run?.id === runId && useEditorStore.getState().document
      ? runId
      : ""
  );
  const didFitForRun = useRef("");
  const lastSavedDocument = useRef("");

  useEffect(() => {
    if (!runId) return;
    const state = useEditorStore.getState();
    if (state.run?.id === runId && state.document) {
      loadedRunId.current = runId;
      lastSavedDocument.current = JSON.stringify(state.document);
      return;
    }
    let active = true;
    void postersService
      .getRun(runId)
      .then((fetched) => {
        if (!active) return;
        setRun(fetched);
        loadedRunId.current = runId;
        lastSavedDocument.current = JSON.stringify(fetched.poster);
        setLoading(false);
      })
      .catch((loadError: unknown) => {
        if (!active) return;
        setError(loadError instanceof Error ? loadError.message : "Could not load poster run.");
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [runId, setRun]);

  const saveEdits = useCallback(async () => {
    if (!document || !runId || loadedRunId.current !== runId) return;
    try {
      setSaveLabel("Logging edits…");
      const updated = await postersService.updateRun(runId, { editedLayout: document });
      setRunMeta(updated);
      lastSavedDocument.current = JSON.stringify(document);
      setSaveLabel("Edits saved to Supabase");
    } catch (saveError) {
      setSaveLabel("Save failed");
      setError(saveError instanceof Error ? saveError.message : "Could not save edits.");
    }
  }, [document, runId, setRunMeta]);

  useEffect(() => {
    if (loading || !document || !runId || loadedRunId.current !== runId) return;
    if (JSON.stringify(document) === lastSavedDocument.current) return;
    setSaveLabel("Unsaved edits");
    const timeout = window.setTimeout(() => {
      void saveEdits();
    }, 800);
    return () => window.clearTimeout(timeout);
  }, [document, loading, runId, saveEdits]);

  useEffect(() => {
    function handleKeys(event: KeyboardEvent) {
      const target = event.target as HTMLElement;
      const editing =
        ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable;
      if (editing) return;
      const command = event.ctrlKey || event.metaKey;
      if (command && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) useEditorStore.getState().redo();
        else useEditorStore.getState().undo();
      } else if (command && event.key.toLowerCase() === "y") {
        event.preventDefault();
        useEditorStore.getState().redo();
      } else if (command && event.key.toLowerCase() === "c" && selectedId && document) {
        const element = document.elements.find((item) => item.id === selectedId);
        if (element) copyBuffer.current = structuredClone(element);
      } else if (command && event.key.toLowerCase() === "v" && copyBuffer.current) {
        event.preventDefault();
        const element = copyBuffer.current;
        useEditorStore.getState().addElement({
          ...element,
          id: crypto.randomUUID(),
          name: `${element.name} copy`,
          x: element.x + 18,
          y: element.y + 18,
          locked: false,
        });
      } else if ((event.key === "Delete" || event.key === "Backspace") && selectedId) {
        useEditorStore.getState().removeElement(selectedId);
      }
    }
    window.addEventListener("keydown", handleKeys);
    return () => window.removeEventListener("keydown", handleKeys);
  }, [document, selectedId]);

  const fitToScreen = useCallback(() => {
    if (!document || !canvasAreaRef.current) return;
    const { clientWidth, clientHeight } = canvasAreaRef.current;
    setZoom(
      Math.min(1.1, (clientWidth - 40) / document.width, (clientHeight - 105) / document.height)
    );
  }, [document, setZoom]);

  useEffect(() => {
    if (loading || !document || didFitForRun.current === (runId || "local")) return;
    const frame = window.requestAnimationFrame(() => {
      fitToScreen();
      didFitForRun.current = runId || "local";
    });
    return () => window.cancelAnimationFrame(frame);
  }, [document, fitToScreen, loading, runId]);

  function getSafeSlug() {
    return (
      (document?.title || "infographix")
        .replace(/[^a-z0-9]+/gi, "-")
        .replace(/^-|-$/g, "")
        .toLowerCase() || "infographix"
    );
  }

  async function handleDownloadPng() {
    if (!document || !stageExporter) return;
    const dataUrl = stageExporter();
    if (!dataUrl) {
      setError("Canvas is not ready for export yet.");
      return;
    }
    postersService.downloadDataUrl(dataUrl, `${getSafeSlug()}-poster.png`);
    if (runId) {
      void postersService.updateRun(runId, { exportedFormat: "png" }).catch(() => {});
    }
  }

  async function handleDownloadPdf() {
    if (!document || !stageExporter) return;
    const dataUrl = stageExporter();
    if (!dataUrl) {
      setError("Canvas is not ready for export yet.");
      return;
    }
    setExportingPdf(true);
    try {
      const pdfBlob = await postersService.exportPdf({
        title: document.title,
        width: document.width,
        height: document.height,
        imageDataUrl: dataUrl,
        runId: runId || undefined,
      });
      postersService.downloadBlob(pdfBlob, `${getSafeSlug()}-poster.pdf`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not export PDF.");
    } finally {
      setExportingPdf(false);
    }
  }

  async function handleRate(rating: number) {
    if (!runId) return;
    try {
      const updated = await postersService.updateRun(runId, { feedbackRating: rating });
      setRunMeta(updated);
    } catch {
      /* non-fatal */
    }
  }

  const qualityScore = run?.generationMetrics?.composite_quality_score;
  const latencyMs = run?.generationMetrics?.latency_ms;

  return (
    <div className="editor-page">
      <header className="editor-topbar">
        <button
          className="editor-back"
          onClick={() => navigate("/")}
          title="Create new poster"
          aria-label="Back to poster input"
        >
          <ArrowLeft size={17} />
        </button>

        <div className="editor-document-name">
          <span className="editor-brand-mark">i</span>
          <input
            aria-label="Poster title"
            value={document?.title || run?.title || "Poster"}
            onChange={(event) =>
              useEditorStore.getState().updatePoster({ title: event.target.value })
            }
          />
        </div>

        {run && (
          <div className="mlops-run-pill" title="Inference provenance">
            <span>{run.engine}</span>
            {run.retrievedTemplateId && <span>tmpl #{run.retrievedTemplateId}</span>}
            {typeof qualityScore === "number" && <span>score {qualityScore}</span>}
            {typeof latencyMs === "number" && <span>{latencyMs}ms</span>}
          </div>
        )}

        <span className={`save-state ${saveLabel === "Save failed" ? "save-error" : ""}`}>
          <span />
          {saveLabel}
        </span>

        <div className="editor-top-actions">
          <button title="Undo (Ctrl+Z)" onClick={undo} disabled={!undoStack.length}>
            <Undo2 size={17} />
          </button>
          <button title="Redo (Ctrl+Y)" onClick={redo} disabled={!redoStack.length}>
            <Redo2 size={17} />
          </button>
          {runId && (
            <button
              className="preview-button"
              onClick={() => void saveEdits()}
              title="Save edits to Supabase training dataset"
            >
              <Save size={15} /> Save
            </button>
          )}
          <button className="preview-button" onClick={() => void handleDownloadPng()}>
            <Download size={15} /> PNG
          </button>
          <button
            className="editor-save-button"
            onClick={() => void handleDownloadPdf()}
            disabled={exportingPdf}
          >
            {exportingPdf ? <Loader2 size={15} className="spin" /> : <FileDown size={15} />} PDF
          </button>
        </div>
      </header>

      <div className="editor-toolbar">
        <div className="toolbar-left">
          <span className="tool-mode">EDITOR</span>
          <span className="toolbar-separator" />
          <button
            className={showGrid ? "toolbar-option active" : "toolbar-option"}
            onClick={() => useEditorStore.getState().toggleGrid()}
          >
            <Grid2X2 size={15} /> Grid
          </button>
          <button
            className={snapToGrid ? "toolbar-option active" : "toolbar-option"}
            onClick={() => useEditorStore.getState().toggleSnap()}
          >
            <span className="snap-icon">⌗</span> Snap
          </button>
          {runId && (
            <>
              <span className="toolbar-separator" />
              <div className="rating-strip" title="Rate generated layout for MLOps dataset">
                <span>Layout rating:</span>
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    type="button"
                    className={(run?.feedbackRating || 0) >= star ? "star-btn active" : "star-btn"}
                    onClick={() => void handleRate(star)}
                  >
                    <Star size={13} />
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="toolbar-zoom">
          <button title="Zoom out" onClick={() => setZoom(zoom - 0.1)}>
            <Minus size={15} />
          </button>
          <span>{Math.round(zoom * 100)}%</span>
          <button title="Zoom in" onClick={() => setZoom(zoom + 0.1)}>
            <Plus size={15} />
          </button>
          <button className="fit-button" onClick={fitToScreen}>
            Fit
          </button>
        </div>
      </div>

      {error && (
        <div className="editor-error" role="alert">
          {error}
          <button onClick={() => setError("")}>Dismiss</button>
        </div>
      )}

      <div className="editor-workspace">
        <EditorSidebar onError={setError} />
        <main className="editor-canvas-area" ref={canvasAreaRef}>
          <div className="canvas-area-heading">
            <div>
              <strong>Poster canvas</strong>
              <small>
                {document
                  ? `${document.width} × ${document.height} px · ${document.orientation}`
                  : "Loading poster"}
              </small>
            </div>
            <span className="canvas-hint">Click an element to edit · Drag to move</span>
          </div>
          <div className="canvas-viewport">
            {loading ? (
              <div className="loading-state">Loading poster run…</div>
            ) : document ? (
              <PosterCanvas />
            ) : (
              <div className="loading-state">
                No poster loaded.{" "}
                <button className="inline-link" onClick={() => navigate("/")}>
                  Create a poster
                </button>
              </div>
            )}
          </div>
          <div className="canvas-footer">
            <span>
              <span className="canvas-ready-dot" /> <Check size={13} /> {saveLabel}
            </span>
            <span>Edits are recorded against the raw AI layout for model retraining</span>
          </div>
        </main>
        <EditorProperties onError={setError} />
      </div>
    </div>
  );
}
