import { apiClient } from "./apiClient";
import { defaultSections } from "./projects";
import type { ResearchSection } from "../types";

export const extractionService = {
  async extract(documentName?: string, projectId?: string): Promise<ResearchSection[]> {
    if (projectId) {
      try {
        const res = await apiClient.post<ResearchSection[]>(`/projects/${projectId}/extract`);
        if (res.data && res.data.length > 0) {
          return res.data;
        }
      } catch {
        // Fall back to template sections if project has no document yet
      }
    }
    const sections = defaultSections.map((section) => ({ ...section }));
    if (documentName) sections[0].content = documentName.replace(/\.(pdf|docx)$/i, "").replace(/[-_]/g, " ");
    return sections;
  },
};
