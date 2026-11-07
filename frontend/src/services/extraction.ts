import { defaultSections } from "./projects";
import type { ResearchSection } from "../types";

export const extractionService = {
  async extract(_documentName?: string): Promise<ResearchSection[]> {
    await new Promise((resolve) => window.setTimeout(resolve, 250));
    const sections = defaultSections.map((section) => ({ ...section }));
    if (_documentName) sections[0].content = _documentName.replace(/\.(pdf|docx)$/i, "").replace(/[-_]/g, " ");
    return sections;
  },
};
