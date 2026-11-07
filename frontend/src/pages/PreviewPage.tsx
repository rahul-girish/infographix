import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Check, Download, Printer, Send, Share2 } from "lucide-react";
import PosterCanvas from "../components/editor/PosterCanvas";
import { submissionsService } from "../services/submissions";
import { projectsService } from "../services/projects";
import { postersService } from "../services/posters";
import { useEditorStore } from "../stores/editorStore";

export default function PreviewPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const document = useEditorStore((state) => state.document);
  const setDocument = useEditorStore((state) => state.setDocument);
  const setZoom = useEditorStore((state) => state.setZoom);
  const [error, setError] = useState("");
  const [submitted, setSubmitted] = useState(false);

  useEffect(() => {
    let active = true;
    void projectsService.get(projectId).then((project) => {
      if (!active) return;
      if (!project?.poster) throw new Error("This project has no poster to preview.");
      setDocument(project.poster);
      setZoom(Math.min(0.78, 880 / project.poster.width));
      setSubmitted(project.status === "submitted");
    }).catch((loadError: unknown) => { if (active) setError(loadError instanceof Error ? loadError.message : "Could not load preview."); });
    return () => { active = false; };
  }, [projectId, setDocument, setZoom]);

  async function submitPoster() {
    try { await submissionsService.submit(projectId); setSubmitted(true); setError(""); }
    catch (submitError) { setError(submitError instanceof Error ? submitError.message : "Could not submit this poster."); }
  }

  return <div className="preview-page"><header className="preview-topbar"><button className="back-button" onClick={() => navigate(`/projects/${projectId}/editor`)}><ArrowLeft size={16} /> Back to editor</button><div><span className="preview-ready"><Check size={14} /> {submitted ? "Submitted" : "Preview ready"}</span><button className="secondary-button" onClick={() => window.print()}><Printer size={16} /> Print</button><button className="secondary-button" onClick={() => { if (document) { const data = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(postersService.toSvg(document))}`; const anchor = window.document.createElement("a"); anchor.href = data; anchor.download = `${document.title.replace(/[^a-z0-9]/gi, "-").toLowerCase()}-poster.svg`; anchor.click(); } }}><Download size={16} /> Export SVG</button><button className="primary-button" onClick={() => void submitPoster()} disabled={submitted}><Send size={15} /> {submitted ? "Submitted" : "Submit poster"}</button></div></header>
    {error && <div className="form-alert error-alert preview-error" role="alert">{error}</div>}
    <main className="preview-content"><div className="preview-heading"><span className="section-kicker">FINAL REVIEW</span><h1>{document?.title || "Poster preview"}</h1><p>Review your poster at presentation scale before submitting.</p></div><div className="preview-canvas-wrap">{document ? <PosterCanvas readOnly /> : <div className="loading-state">Loading preview…</div>}</div><div className="preview-checklist"><span><Check size={15} /> All elements remain editable</span><span><Share2 size={15} /> Saved to your local workspace</span></div></main>
  </div>;
}
