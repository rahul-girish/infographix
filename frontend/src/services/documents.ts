import { apiClient } from "./apiClient";
import type { ResearchSection } from "../types";

export interface DocumentUploadResult {
  name: string;
  sections?: ResearchSection[];
}

export const documentsService = {
  validate(file: File): string | null {
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!extension || !["pdf", "docx", "doc"].includes(extension)) return "Choose a PDF or DOCX document.";
    if (file.size > 20 * 1024 * 1024) return "The document must be smaller than 20 MB.";
    return null;
  },
  async upload(file: File, onProgress?: (progress: number) => void, projectId?: string): Promise<DocumentUploadResult> {
    const error = this.validate(file);
    if (error) throw new Error(error);

    if (projectId) {
      const formData = new FormData();
      formData.append("file", file);
      const res = await apiClient.post<{ name: string; sections: ResearchSection[] }>(
        `/projects/${projectId}/documents`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: (progressEvent) => {
            if (progressEvent.total) {
              const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              onProgress?.(percent);
            }
          },
        }
      );
      onProgress?.(100);
      return { name: res.data.name, sections: res.data.sections };
    }

    // Fallback if no project ID passed
    for (const progress of [18, 42, 68, 91, 100]) {
      await new Promise((resolve) => window.setTimeout(resolve, 80));
      onProgress?.(progress);
    }
    return { name: file.name };
  },
};
