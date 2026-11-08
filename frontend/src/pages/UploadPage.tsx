import { useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, Check, FileText, FileUp, LoaderCircle, ShieldCheck, X } from "lucide-react";
import WorkspaceHeader from "../components/layout/WorkspaceHeader";
import { documentsService } from "../services/documents";
import { extractionService } from "../services/extraction";
import { projectsService } from "../services/projects";

export default function UploadPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState("");
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  async function chooseFile(file?: File) {
    if (!file) return;
    const validationError = documentsService.validate(file);
    if (validationError) { setError(validationError); return; }
    setError(""); setBusy(true); setFileName(file.name); setProgress(0);
    try {
      await documentsService.upload(file, setProgress);
      await projectsService.attachDocument(projectId, file.name);
      await projectsService.saveSections(projectId, await extractionService.extract(file.name));
    } catch (uploadError) {
      setFileName("");
      setError(uploadError instanceof Error ? uploadError.message : "Could not upload this document.");
    } finally { setBusy(false); }
  }

  async function continueToReview() {
    try {
      if (!await projectsService.get(projectId)) throw new Error("This project could not be found.");
      navigate(`/projects/${projectId}/review`);
    } catch (continueError) { setError(continueError instanceof Error ? continueError.message : "Could not continue."); }
  }

  return <div className="workspace-page"><WorkspaceHeader /><main className="flow-main">
    <div className="flow-topline"><button className="back-button" onClick={() => navigate("/dashboard")}><ArrowLeft size={16} /> Dashboard</button><span>STEP 1 OF 3</span></div>
    <div className="flow-progress"><i className="active" /><i /><i /></div>
    <div className="flow-heading"><div className="flow-icon"><FileUp size={22} /></div><h1>Start with your research</h1><p>Upload a paper and we’ll prepare a structured starting point for your poster. You can also begin with a blank canvas.</p></div>
    <div className={`drop-zone ${dragging ? "drag-active" : ""} ${fileName ? "file-selected" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); void chooseFile(event.dataTransfer.files[0]); }}>
      <input ref={inputRef} type="file" accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document" className="visually-hidden" onChange={(event) => void chooseFile(event.target.files?.[0])} />
      {fileName ? <div className="uploaded-file"><span className="uploaded-file-icon"><FileText size={22} /></span><div><strong>{fileName}</strong><small>{busy ? `Preparing document · ${progress}%` : "Ready to continue"}</small>{busy && <span className="upload-progress"><i style={{ width: `${progress}%` }} /></span>}</div><button onClick={(event) => { event.stopPropagation(); setFileName(""); setProgress(0); if (inputRef.current) inputRef.current.value = ""; }} aria-label="Remove document" disabled={busy}><X size={18} /></button></div> : <><div className="drop-icon"><FileUp size={24} /></div><h2>Drop your research paper here</h2><p>or <button className="inline-link" onClick={() => inputRef.current?.click()}>browse files</button> from your computer</p><span className="file-format-tags"><b>PDF</b><b>DOCX</b><small>Maximum file size: 20 MB</small></span></>}
    </div>
    {error && <div className="form-alert error-alert" role="alert">{error}</div>}
    <div className="upload-privacy"><ShieldCheck size={17} /><span><strong>Your research stays yours.</strong> Files are only mocked locally in this demo and are not sent to a server.</span></div>
    <div className="flow-actions"><button className="text-button" onClick={() => void continueToReview()}>Continue without uploading</button><button className="primary-button" onClick={() => void continueToReview()} disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : fileName ? <Check size={17} /> : null} Continue <ArrowRight size={17} /></button></div>
  </main></div>;
}
