import { apiClient } from "./apiClient";
import type { PosterDocument, Project, ResearchSection } from "../types";

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

export const projectsService = {
  async list(): Promise<Project[]> {
    const res = await apiClient.get<Project[]>("/projects");
    return res.data;
  },
  async get(id: string): Promise<Project | undefined> {
    try {
      const res = await apiClient.get<Project>(`/projects/${id}`);
      return res.data;
    } catch {
      return undefined;
    }
  },
  async create(title = "Untitled research poster"): Promise<Project> {
    const res = await apiClient.post<Project>("/projects", { title, templateId: "research" });
    return res.data;
  },
  async save(project: Project): Promise<Project> {
    if (project.poster) {
      await this.savePoster(project.id, project.poster);
    }
    if (project.sections) {
      await this.saveSections(project.id, project.sections);
    }
    const updated = await this.get(project.id);
    if (!updated) throw new Error("This project no longer exists.");
    return updated;
  },
  async rename(id: string, title: string): Promise<void> {
    await apiClient.patch(`/projects/${id}/rename`, { title });
  },
  async duplicate(id: string): Promise<Project> {
    const res = await apiClient.post<Project>(`/projects/${id}/duplicate`);
    return res.data;
  },
  async delete(id: string): Promise<void> {
    await apiClient.delete(`/projects/${id}`);
  },
  async updateStatus(id: string, status: Project["status"]): Promise<void> {
    await apiClient.patch(`/projects/${id}/status`, { status });
  },
  async savePoster(id: string, poster: PosterDocument): Promise<void> {
    await apiClient.put(`/projects/${id}/poster`, poster);
  },
  async saveSections(id: string, sections: ResearchSection[]): Promise<void> {
    await apiClient.put(`/projects/${id}/sections`, sections);
  },
  async attachDocument(_id: string, _name: string): Promise<void> {
    // Handled automatically on backend during document upload
  },
};
