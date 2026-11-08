import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, Check, Columns3, Frame, Palette, Type } from "lucide-react";
import WorkspaceHeader from "../components/layout/WorkspaceHeader";
import { posterTemplates } from "../data/templates";
import { projectsService } from "../services/projects";
import type { Orientation, Project } from "../types";

export default function ConfigurePage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [templateId, setTemplateId] = useState("research");
  const [orientation, setOrientation] = useState<Orientation>("landscape");
  const [background, setBackground] = useState("#ffffff");
  const [columns, setColumns] = useState(3);
  const [error, setError] = useState("");
  useEffect(() => { void projectsService.get(projectId).then((value) => { if (!value) setError("This project could not be found."); else { setProject(value); setTemplateId(value.templateId); setBackground(value.poster?.background || "#ffffff"); setColumns(value.poster?.columns || 3); } }).catch((loadError: unknown) => setError(loadError instanceof Error ? loadError.message : "Could not load project.")); }, [projectId]);
  const template = posterTemplates.find((item) => item.id === templateId) || posterTemplates[0];

  async function openEditor() {
    if (!project?.poster) { setError("The poster could not be initialized. Please reopen this project."); return; }
    const dimensions = orientation === "landscape" ? { width: 1000, height: 700 } : { width: 700, height: 1000 };
    const sections = project.sections.filter((section) => section.included);
    const contentFor = (id: string, fallback: string) => sections.find((section) => section.id === id)?.content || fallback;
    const horizontalScale = dimensions.width / project.poster.width;
    const verticalScale = dimensions.height / project.poster.height;
    const elements = project.poster.elements.map((element) => {
      const updated = { ...element, x: element.x * horizontalScale, width: element.width * horizontalScale, y: element.y * verticalScale, height: element.height * verticalScale };
      if (updated.name === "Header background") updated.fill = template.accent;
      if (updated.name === "Poster title") updated.text = contentFor("title", project.title).toUpperCase();
      if (updated.name === "Authors") updated.text = [contentFor("authors", ""), contentFor("affiliations", "")].filter(Boolean).join(" · ");
      if (updated.name === "Introduction copy") updated.text = contentFor("introduction", contentFor("abstract", updated.text || ""));
      if (updated.name === "Figure caption") updated.text = contentFor("methodology", "Add a chart, figure or diagram");
      if (updated.name === "Results copy") updated.text = contentFor("results", updated.text || "");
      if (updated.name === "Conclusion") updated.text = `CONCLUSION   ·   ${contentFor("conclusion", "Add the takeaway your audience should remember.")}`;
      return updated;
    });
    const supplemental = sections.filter((section) => ["problem", "objectives", "discussion", "references", "keywords"].includes(section.id));
    const width = dimensions.width / columns - 45;
    const additions = supplemental.map((section, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = 35 + column * (dimensions.width / columns);
      const y = dimensions.height * (orientation === "landscape" ? 0.64 : 0.52) + row * 74;
      return [
        { id: crypto.randomUUID(), type: "text" as const, name: `${section.title} heading`, text: section.title.toUpperCase(), x, y, width, height: 24, rotation: 0, layer: elements.length + index * 2, visible: true, locked: false, fill: template.accent, fontFamily: "Arial", fontSize: 14, fontWeight: "bold" as const },
        { id: crypto.randomUUID(), type: "text" as const, name: `${section.title} content`, text: section.content, x, y: y + 26, width, height: 46, rotation: 0, layer: elements.length + index * 2 + 1, visible: true, locked: false, fill: "#334155", fontFamily: "Arial", fontSize: 11 },
      ];
    }).flat();
    const poster = { ...project.poster, ...dimensions, orientation, templateId, background, columns, elements: [...elements, ...additions] };
    try { await projectsService.save({ ...project, templateId, poster }); navigate(`/projects/${projectId}/editor`); }
    catch (saveError) { setError(saveError instanceof Error ? saveError.message : "Could not save poster settings."); }
  }

  return <div className="workspace-page"><WorkspaceHeader /><main className="flow-main configure-main">
    <div className="flow-topline"><button className="back-button" onClick={() => navigate(`/projects/${projectId}/review`)}><ArrowLeft size={16} /> Back to content</button><span>STEP 3 OF 3</span></div><div className="flow-progress"><i className="complete" /><i className="complete" /><i className="active" /></div>
    <div className="flow-heading compact-heading"><span className="section-kicker">MAKE IT YOURS</span><h1>Set up your poster</h1><p>Choose the format and visual direction. You can refine every detail in the editor.</p></div>
    <section className="config-block"><div className="config-block-heading"><span className="config-icon"><Frame size={18} /></span><div><h2>Poster orientation</h2><p>Pick the format for your print or presentation.</p></div></div><div className="orientation-options"><button className={orientation === "landscape" ? "orientation-card chosen" : "orientation-card"} onClick={() => setOrientation("landscape")}><span className="orientation-preview landscape-preview" /><strong>Landscape</strong><small>48 × 36 in · 4:3</small>{orientation === "landscape" && <Check size={16} />}</button><button className={orientation === "portrait" ? "orientation-card chosen" : "orientation-card"} onClick={() => setOrientation("portrait")}><span className="orientation-preview portrait-preview" /><strong>Portrait</strong><small>36 × 48 in · 3:4</small>{orientation === "portrait" && <Check size={16} />}</button></div></section>
    <section className="config-block"><div className="config-block-heading"><span className="config-icon"><Columns3 size={18} /></span><div><h2>Choose a template</h2><p>Each template is a fully editable starting point.</p></div></div><div className="template-options">{posterTemplates.map((item, index) => <button key={item.id} className={templateId === item.id ? "template-option chosen" : "template-option"} onClick={() => { setTemplateId(item.id); setBackground(item.background); setColumns(item.columns); }}><span className={`template-swatch template-swatch-${index}`}><i /><i /><i /></span><span><strong>{item.name}</strong><small>{item.description}</small><em>{item.columns} columns · {item.category}</em></span>{templateId === item.id && <Check size={17} />}</button>)}</div><div className="column-control"><span><strong>Column layout</strong><small>Organize your sections across the page</small></span><div>{[1, 2, 3, 4].map((count) => <button key={count} className={columns === count ? "active" : ""} onClick={() => setColumns(count)}>{count}</button>)}</div></div></section>
    <section className="config-block config-small-grid"><div><div className="config-block-heading"><span className="config-icon"><Palette size={18} /></span><div><h2>Background</h2><p>Pick a calm canvas color.</p></div></div><div className="color-options">{["#ffffff", "#fbfaff", "#f7fcfd", "#f8fcfa", "#f8fafc"].map((color) => <button aria-label={`Background ${color}`} key={color} className={`color-swatch ${background === color ? "active" : ""}`} style={{ backgroundColor: color }} onClick={() => setBackground(color)} />)}</div></div><div><div className="config-block-heading"><span className="config-icon"><Type size={18} /></span><div><h2>Typography</h2><p>Change fonts anytime in the editor.</p></div></div><div className="font-note"><strong>Modern & readable</strong><small>Inter · Arial · Georgia</small></div></div></section>
    {error && <div className="form-alert error-alert" role="alert">{error}</div>}
    <div className="flow-actions"><span className="config-selected"><Check size={15} /> {template.name}</span><button className="primary-button" onClick={() => void openEditor()}>Open the editor <ArrowRight size={17} /></button></div>
  </main></div>;
}
