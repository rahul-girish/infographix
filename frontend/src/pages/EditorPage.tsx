import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Check, ChevronDown, Eye, Grid2X2, Minus, Plus, Redo2, Save, Undo2 } from "lucide-react";
import { EditorProperties, EditorSidebar } from "../components/editor/EditorPanels";
import PosterCanvas from "../components/editor/PosterCanvas";
import { projectsService } from "../services/projects";
import { useEditorStore } from "../stores/editorStore";
import type { PosterElement, Project } from "../types";

export default function EditorPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const document = useEditorStore((state) => state.document);
  const selectedId = useEditorStore((state) => state.selectedId);
  const zoom = useEditorStore((state) => state.zoom);
  const undoStack = useEditorStore((state) => state.undoStack);
  const redoStack = useEditorStore((state) => state.redoStack);
  const showGrid = useEditorStore((state) => state.showGrid);
  const snapToGrid = useEditorStore((state) => state.snapToGrid);
  const setDocument = useEditorStore((state) => state.setDocument);
  const setZoom = useEditorStore((state) => state.setZoom);
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [saveLabel, setSaveLabel] = useState("All changes saved");
  const canvasAreaRef = useRef<HTMLDivElement>(null);
  const copyBuffer = useRef<PosterElement | null>(null);
  const loadedId = useRef("");
  const didFitForProject = useRef("");
  const lastSavedDocument = useRef("");

  useEffect(() => {
    let active = true;
    void projectsService.get(projectId).then((value) => {
      if (!active) return;
      if (!value?.poster) throw new Error("This project has no poster to edit. Reopen it from the dashboard.");
      setProject(value);
      setDocument(value.poster);
      loadedId.current = projectId;
      lastSavedDocument.current = JSON.stringify(value.poster);
      setLoading(false);
    }).catch((loadError: unknown) => {
      if (!active) return;
      setError(loadError instanceof Error ? loadError.message : "Could not open this poster.");
      setLoading(false);
    });
    return () => { active = false; };
  }, [projectId, setDocument]);

  const save = useCallback(async () => {
    if (!document || loadedId.current !== projectId) return;
    try {
      setSaveLabel("Saving…");
      await projectsService.savePoster(projectId, document);
      lastSavedDocument.current = JSON.stringify(document);
      setSaveLabel("All changes saved");
    } catch (saveError) {
      setSaveLabel("Save failed");
      setError(saveError instanceof Error ? saveError.message : "Could not save your poster.");
    }
  }, [document, projectId]);

  useEffect(() => {
    if (loading || !document || loadedId.current !== projectId) return;
    if (JSON.stringify(document) === lastSavedDocument.current) return;
    setSaveLabel("Unsaved changes");
    const timeout = window.setTimeout(() => { void save(); }, 650);
    return () => window.clearTimeout(timeout);
  }, [document, loading, projectId, save]);

  useEffect(() => {
    function handleKeys(event: KeyboardEvent) {
      const target = event.target as HTMLElement;
      const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable;
      if (editing) return;
      const command = event.ctrlKey || event.metaKey;
      if (command && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) useEditorStore.getState().redo();
        else useEditorStore.getState().undo();
      } else if (command && event.key.toLowerCase() === "y") {
        event.preventDefault(); useEditorStore.getState().redo();
      } else if (command && event.key.toLowerCase() === "c" && selectedId && document) {
        const element = document.elements.find((item) => item.id === selectedId);
        if (element) copyBuffer.current = structuredClone(element);
      } else if (command && event.key.toLowerCase() === "v" && copyBuffer.current) {
        event.preventDefault();
        const element = copyBuffer.current;
        useEditorStore.getState().addElement({ ...element, id: crypto.randomUUID(), name: `${element.name} copy`, x: element.x + 18, y: element.y + 18, locked: false });
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
    setZoom(Math.min(1.1, (clientWidth - 40) / document.width, (clientHeight - 105) / document.height));
  }, [document, setZoom]);

  useEffect(() => {
    if (loading || !document || loadedId.current !== projectId || didFitForProject.current === projectId) return;
    const frame = window.requestAnimationFrame(() => {
      fitToScreen();
      didFitForProject.current = projectId;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [document, fitToScreen, loading, projectId]);

  const undo = useEditorStore((state) => state.undo);
  const redo = useEditorStore((state) => state.redo);
  return <div className="editor-page">
    <header className="editor-topbar">
      <button className="editor-back" onClick={() => navigate("/dashboard")} aria-label="Back to dashboard"><ArrowLeft size={17} /></button>
      <div className="editor-document-name"><span className="editor-brand-mark">i</span><input aria-label="Poster title" value={document?.title || project?.title || "Poster"} onChange={(event) => useEditorStore.getState().updatePoster({ title: event.target.value })} /><ChevronDown size={14} /></div>
      <span className={`save-state ${saveLabel === "Save failed" ? "save-error" : ""}`}><span />{saveLabel}</span>
      <div className="editor-top-actions"><button title="Undo (Ctrl+Z)" onClick={undo} disabled={!undoStack.length}><Undo2 size={17} /></button><button title="Redo (Ctrl+Y)" onClick={redo} disabled={!redoStack.length}><Redo2 size={17} /></button><button className="preview-button" onClick={() => navigate(`/projects/${projectId}/preview`)}><Eye size={16} /> Preview</button><button className="editor-save-button" onClick={() => void save()}><Save size={16} /> Save</button></div>
    </header>
    <div className="editor-toolbar"><div className="toolbar-left"><span className="tool-mode">DESIGN MODE</span><span className="toolbar-separator" /><button className={showGrid ? "toolbar-option active" : "toolbar-option"} onClick={() => useEditorStore.getState().toggleGrid()}><Grid2X2 size={15} /> Grid</button><button className={snapToGrid ? "toolbar-option active" : "toolbar-option"} onClick={() => useEditorStore.getState().toggleSnap()}><span className="snap-icon">⌗</span> Snap</button></div><div className="toolbar-zoom"><button title="Zoom out" onClick={() => setZoom(zoom - 0.1)}><Minus size={15} /></button><span>{Math.round(zoom * 100)}%</span><button title="Zoom in" onClick={() => setZoom(zoom + 0.1)}><Plus size={15} /></button><button className="fit-button" onClick={fitToScreen}>Fit</button></div></div>
    {error && <div className="editor-error" role="alert">{error}<button onClick={() => setError("")}>Dismiss</button></div>}
    <div className="editor-workspace">
      <EditorSidebar onError={setError} />
      <main className="editor-canvas-area" ref={canvasAreaRef}>
        <div className="canvas-area-heading"><div><strong>Poster canvas</strong><small>{document ? `${document.width} × ${document.height} px · ${document.orientation}` : "Loading poster"}</small></div><span className="canvas-hint">Click an element to edit · Drag to move</span></div>
        <div className="canvas-viewport">{loading ? <div className="loading-state">Opening your poster…</div> : document ? <PosterCanvas /> : <div className="loading-state">No poster to display.</div>}</div>
        <div className="canvas-footer"><span><span className="canvas-ready-dot" /> <Check size={13} /> {saveLabel}</span><span>Changes save automatically</span></div>
      </main>
      <EditorProperties onError={setError} />
    </div>
  </div>;
}
