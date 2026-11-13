import { apiClient } from "./apiClient";
import type {
  GenerationRun,
  GenerationRunSummary,
  PosterDocument,
  PosterGenerateRequest,
} from "../types";

export const postersService = {
  async generate(payload: PosterGenerateRequest): Promise<GenerationRun> {
    const res = await apiClient.post<GenerationRun>("/v1/generate", payload);
    return res.data;
  },

  async getRun(runId: string): Promise<GenerationRun> {
    const res = await apiClient.get<GenerationRun>(`/v1/runs/${runId}`);
    return res.data;
  },

  async listRuns(limit = 15): Promise<GenerationRunSummary[]> {
    const res = await apiClient.get<GenerationRunSummary[]>("/v1/runs", {
      params: { limit },
    });
    return res.data;
  },

  async updateRun(
    runId: string,
    update: {
      editedLayout?: PosterDocument;
      exportedFormat?: "png" | "pdf" | "svg";
      feedbackRating?: number;
      feedbackNotes?: string;
    }
  ): Promise<GenerationRun> {
    const res = await apiClient.patch<GenerationRun>(`/v1/runs/${runId}`, update);
    return res.data;
  },

  async exportPdf(payload: {
    title: string;
    width: number;
    height: number;
    imageDataUrl: string;
    runId?: string;
  }): Promise<Blob> {
    const res = await apiClient.post("/v1/export/pdf", payload, {
      responseType: "blob",
    });
    return res.data as Blob;
  },

  getTrainingExportUrl(format: "jsonl" | "json" = "jsonl"): string {
    const base = apiClient.defaults.baseURL || "/api";
    return `${base}/v1/runs/export/training-data?format=${format}`;
  },

  downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const anchor = window.document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  },

  downloadDataUrl(dataUrl: string, filename: string): void {
    const anchor = window.document.createElement("a");
    anchor.href = dataUrl;
    anchor.download = filename;
    anchor.click();
  },
};
