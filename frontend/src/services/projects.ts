import { createStarterElements, posterTemplates } from "../data/templates";
import type { PosterDocument, Project, ResearchSection } from "../types";
import { readStorage, writeStorage } from "./storage";

const PROJECTS_KEY = "infographix.projects.v1";

export const defaultSections: ResearchSection[] = [
  ["title", "Title", "A new perspective on research"],
  ["authors", "Authors", "Researcher Name, Co-author Name"],
  ["affiliations", "Affiliations", "Department of Research, University"],
  ["abstract", "Abstract", "A concise summary of the research question, approach and key findings."],
  ["introduction", "Introduction", "Introduce the topic and relevant background."],
  ["problem", "Problem statement", "Describe the challenge this work addresses."],
  ["objectives", "Objectives", "List the primary aims of this research."],
  ["methodology", "Methodology", "Explain the methods, materials and process."],
  ["results", "Results", "Present the key observations and data."],
  ["discussion", "Discussion", "Interpret the findings and their implications."],
  ["conclusion", "Conclusion", "Summarize the main takeaway."],
  ["references", "References", "Add sources cited in this poster."],
  ["keywords", "Keywords", "research, science, innovation"],
].map(([id, title, content]) => ({ id, title, content, included: true }));

function getAll(): Project[] {
  return readStorage<Project[]>(PROJECTS_KEY, []);
}

function persist(projects: Project[]): Project[] {
  writeStorage(PROJECTS_KEY, projects);
  return projects;
}

export const projectsService = {
  async list(): Promise<Project[]> {
    return getAll().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  },
  async get(id: string): Promise<Project | undefined> {
    return getAll().find((project) => project.id === id);
  },
  async create(title = "Untitled research poster"): Promise<Project> {
    const now = new Date().toISOString();
    const template = posterTemplates[0];
    const project: Project = {
      id: crypto.randomUUID(),
      title,
      status: "draft",
      updatedAt: now,
      templateId: template.id,
      sections: defaultSections.map((section) => ({ ...section })),
      poster: {
        id: crypto.randomUUID(),
        projectId: "",
        title,
        width: 1000,
        height: 700,
        orientation: "landscape",
        background: template.background,
        templateId: template.id,
        columns: template.columns,
        elements: createStarterElements(title.toUpperCase(), template.accent),
        updatedAt: now,
      },
    };
    project.poster!.projectId = project.id;
    persist([...getAll(), project]);
    return project;
  },
  async save(project: Project): Promise<Project> {
    const updated = { ...project, updatedAt: new Date().toISOString() };
    const projects = getAll();
    const index = projects.findIndex((item) => item.id === project.id);
    if (index < 0) throw new Error("This project no longer exists.");
    projects[index] = updated;
    persist(projects);
    return updated;
  },
  async rename(id: string, title: string): Promise<void> {
    const project = await this.get(id);
    if (!project) throw new Error("This project no longer exists.");
    await this.save({ ...project, title, poster: project.poster ? { ...project.poster, title } : undefined });
  },
  async duplicate(id: string): Promise<Project> {
    const source = await this.get(id);
    if (!source) throw new Error("This project no longer exists.");
    const title = `${source.title} (copy)`;
    const copy: Project = {
      ...source,
      id: crypto.randomUUID(),
      title,
      status: "draft",
      updatedAt: new Date().toISOString(),
      poster: source.poster ? {
        ...source.poster,
        id: crypto.randomUUID(),
        title,
        projectId: "",
        elements: source.poster.elements.map((element) => ({ ...element, id: crypto.randomUUID() })),
      } : undefined,
    };
    if (copy.poster) copy.poster.projectId = copy.id;
    persist([...getAll(), copy]);
    return copy;
  },
  async delete(id: string): Promise<void> {
    const projects = getAll();
    if (!projects.some((project) => project.id === id)) throw new Error("This project no longer exists.");
    persist(projects.filter((project) => project.id !== id));
  },
  async updateStatus(id: string, status: Project["status"]): Promise<void> {
    const project = await this.get(id);
    if (!project) throw new Error("This project no longer exists.");
    await this.save({ ...project, status });
  },
  async savePoster(id: string, poster: PosterDocument): Promise<void> {
    const project = await this.get(id);
    if (!project) throw new Error("This project no longer exists.");
    await this.save({ ...project, poster, title: poster.title });
  },
  async saveSections(id: string, sections: ResearchSection[]): Promise<void> {
    const project = await this.get(id);
    if (!project) throw new Error("This project no longer exists.");
    await this.save({ ...project, sections });
  },
  async attachDocument(id: string, name: string): Promise<void> {
    const project = await this.get(id);
    if (!project) throw new Error("This project no longer exists.");
    await this.save({ ...project, documentName: name });
  },
};
