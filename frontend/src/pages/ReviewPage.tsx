import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Check, Eye, EyeOff, Plus, Trash2 } from "lucide-react";
import WorkspaceHeader from "../components/layout/WorkspaceHeader";
import { extractionService } from "../services/extraction";
import { projectsService } from "../services/projects";
import type { ResearchSection } from "../types";

export default function ReviewPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const [sections, setSections] = useState<ResearchSection[]>([]);
  const [projectTitle, setProjectTitle] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const project = await projectsService.get(projectId);
        if (!project) throw new Error("This project could not be found.");
        if (!active) return;
        setProjectTitle(project.title);
        setSections(project.sections.length ? project.sections : await extractionService.extract(project.documentName));
      } catch (loadError) { if (active) setError(loadError instanceof Error ? loadError.message : "Could not load extracted content."); }
      finally { if (active) setLoading(false); }
    })();
    return () => { active = false; };
  }, [projectId]);

  function updateSection(id: string, update: Partial<ResearchSection>) {
    setSections((current) => current.map((section) => section.id === id ? { ...section, ...update } : section));
  }
  function moveSection(index: number, direction: -1 | 1) {
    const target = index + direction;
    if (target < 0 || target >= sections.length) return;
    setSections((current) => { const copy = [...current]; [copy[index], copy[target]] = [copy[target], copy[index]]; return copy; });
  }
  async function continueToConfigure() {
    try { await projectsService.saveSections(projectId, sections); navigate(`/projects/${projectId}/configure`); }
    catch (saveError) { setError(saveError instanceof Error ? saveError.message : "Could not save your content."); }
  }

  return <div className="workspace-page"><WorkspaceHeader /><main className="flow-main review-main">
    <div className="flow-topline"><button className="back-button" onClick={() => navigate(`/projects/${projectId}/upload`)}><ArrowLeft size={16} /> Back</button><span>STEP 2 OF 3</span></div><div className="flow-progress"><i className="complete" /><i className="active" /><i /></div>
    <div className="review-title-row"><div><div className="flow-heading compact-heading"><span className="section-kicker">RESEARCH CONTENT</span><h1>Review your content</h1><p>Check, refine and arrange the information that will shape your poster.</p></div></div><div className="review-project-title"><small>PROJECT</small><strong>{projectTitle}</strong></div></div>
    {error && <div className="form-alert error-alert" role="alert">{error}</div>}
    {loading ? <div className="loading-state">Preparing your research sections…</div> : <div className="review-sections">
      {sections.map((section, index) => <article className={`review-section ${section.included ? "" : "section-excluded"}`} key={section.id}>
        <div className="section-order"><span>{String(index + 1).padStart(2, "0")}</span><button aria-label={`Move ${section.title} up`} disabled={index === 0} onClick={() => moveSection(index, -1)}><ArrowUp size={14} /></button><button aria-label={`Move ${section.title} down`} disabled={index === sections.length - 1} onClick={() => moveSection(index, 1)}><ArrowDown size={14} /></button></div>
        <div className="section-fields"><input className="section-title-input" aria-label="Section title" value={section.title} onChange={(event) => updateSection(section.id, { title: event.target.value })} /><textarea aria-label={`${section.title} content`} value={section.content} onChange={(event) => updateSection(section.id, { content: event.target.value })} rows={section.content.length > 130 ? 4 : 2} /></div>
        <div className="section-actions"><button title={section.included ? "Exclude from poster" : "Include in poster"} aria-label={section.included ? "Exclude section" : "Include section"} className={section.included ? "include-toggle selected" : "include-toggle"} onClick={() => updateSection(section.id, { included: !section.included })}>{section.included ? <Eye size={16} /> : <EyeOff size={16} />}</button><button title="Delete section" aria-label={`Delete ${section.title}`} className="remove-section" onClick={() => setSections((current) => current.filter((item) => item.id !== section.id))}><Trash2 size={16} /></button></div>
      </article>)}
      <button className="add-section-button" onClick={() => setSections((current) => [...current, { id: crypto.randomUUID(), title: "New section", content: "", included: true }])}><Plus size={16} /> Add a section</button>
    </div>}
    <div className="flow-actions review-flow-actions"><span><Check size={15} /> {sections.filter((section) => section.included).length} sections included</span><button className="primary-button" onClick={() => void continueToConfigure()} disabled={loading}>Choose a design <ArrowRight size={17} /></button></div>
  </main></div>;
}
