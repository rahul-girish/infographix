import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Clock3, FilePlus2, FileText, MoreHorizontal, Plus, Search, Sparkles, Trash2, Copy, PencilLine, Layers3 } from "lucide-react";
import WorkspaceHeader from "../components/layout/WorkspaceHeader";
import { projectsService } from "../services/projects";
import type { Project } from "../types";

export default function DashboardPage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [error, setError] = useState("");
  const [menuId, setMenuId] = useState<string | null>(null);

  async function refresh() {
    try { setProjects(await projectsService.list()); setError(""); }
    catch (loadError) { setError(loadError instanceof Error ? loadError.message : "Could not load projects."); }
  }
  useEffect(() => {
    let active = true;
    void projectsService.list().then((items) => { if (active) setProjects(items); }).catch((loadError: unknown) => {
      if (active) setError(loadError instanceof Error ? loadError.message : "Could not load projects.");
    });
    return () => { active = false; };
  }, []);

  const visibleProjects = useMemo(() => projects.filter((project) => {
    const matchesSearch = project.title.toLowerCase().includes(query.toLowerCase());
    const matchesFilter = filter === "all" || project.status === filter;
    return matchesSearch && matchesFilter;
  }), [projects, query, filter]);

  async function createProject() {
    try {
      const project = await projectsService.create();
      navigate(`/projects/${project.id}/upload`);
    } catch (createError) { setError(createError instanceof Error ? createError.message : "Could not create project."); }
  }

  async function act(action: "rename" | "duplicate" | "delete", project: Project) {
    setMenuId(null);
    try {
      if (action === "rename") {
        const title = window.prompt("Choose a new project name", project.title)?.trim();
        if (title) await projectsService.rename(project.id, title);
      } else if (action === "duplicate") await projectsService.duplicate(project.id);
      else if (window.confirm(`Delete “${project.title}”? This cannot be undone.`)) await projectsService.delete(project.id);
      await refresh();
    } catch (actionError) { setError(actionError instanceof Error ? actionError.message : "Could not update project."); }
  }

  return (
    <div className="workspace-page">
      <WorkspaceHeader />
      <main className="dashboard-main">
        <div className="dashboard-welcome">
          <div><div className="eyebrow light-eyebrow"><Sparkles size={14} /> YOUR CREATIVE WORKSPACE</div><h1>Good ideas deserve <span>a great canvas.</span></h1><p>Pick up where you left off or start shaping something new.</p></div>
          <button className="primary-button create-project-button" onClick={() => void createProject()}><Plus size={19} /> Create a poster</button>
        </div>
        <section className="dashboard-stat-row">
          <div className="dashboard-stat"><span className="stat-icon blue-stat"><Layers3 size={19} /></span><div><strong>{projects.length}</strong><small>Total projects</small></div></div>
          <div className="dashboard-stat"><span className="stat-icon violet-stat"><FileText size={19} /></span><div><strong>{projects.filter((project) => project.status === "draft").length}</strong><small>In progress</small></div></div>
          <div className="dashboard-stat"><span className="stat-icon green-stat"><Clock3 size={19} /></span><div><strong>{projects.filter((project) => project.status === "submitted").length}</strong><small>Submitted</small></div></div>
        </section>
        <section className="projects-section">
          <div className="projects-heading"><div><h2>Your projects</h2><p>All your research posters, in one place.</p></div><div className="project-controls"><label className="search-field"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search projects" /></label><select aria-label="Filter projects" value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">All projects</option><option value="draft">Drafts</option><option value="submitted">Submitted</option></select></div></div>
          {error && <div className="form-alert error-alert" role="alert">{error}</div>}
          {visibleProjects.length === 0 ? <div className="empty-projects"><div className="empty-icon"><FilePlus2 size={26} /></div><h3>{query || filter !== "all" ? "No matching projects" : "Your next big idea starts here"}</h3><p>{query || filter !== "all" ? "Try another search or filter." : "Create a poster to begin turning your research into a clear visual story."}</p>{!query && filter === "all" && <button className="primary-button" onClick={() => void createProject()}><Plus size={18} /> Create your first poster</button>}</div> :
            <div className="project-grid">{visibleProjects.map((project, index) => <article className="project-card" key={project.id}>
              <button className="project-open" onClick={() => navigate(`/projects/${project.id}/editor`)} aria-label={`Open ${project.title}`}>
                <div className={`project-thumbnail thumb-${index % 5}`}><div className="mini-poster"><div className="mini-poster-heading" /><div className="mini-poster-title" /><div className="mini-poster-columns"><i /><i /><i /></div><div className="mini-poster-caption" /></div><span className={`status-pill ${project.status}`}>{project.status === "submitted" ? "Submitted" : "Draft"}</span></div>
                <div className="project-card-copy"><h3>{project.title}</h3><p><Clock3 size={13} /> Edited {new Date(project.updatedAt).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</p></div>
              </button>
              <div className="project-card-actions"><button onClick={() => navigate(`/projects/${project.id}/editor`)}>Open project <ArrowRight size={15} /></button><button className="project-more" aria-label={`More options for ${project.title}`} onClick={() => setMenuId(menuId === project.id ? null : project.id)}><MoreHorizontal size={19} /></button>
                {menuId === project.id && <div className="project-menu"><button onClick={() => void act("rename", project)}><PencilLine size={15} /> Rename</button><button onClick={() => void act("duplicate", project)}><Copy size={15} /> Duplicate</button><button className="delete-action" onClick={() => void act("delete", project)}><Trash2 size={15} /> Delete</button></div>}
              </div>
            </article>)}</div>}
        </section>
      </main>
    </div>
  );
}
