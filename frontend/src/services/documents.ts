export const documentsService = {
  validate(file: File): string | null {
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!extension || !["pdf", "docx"].includes(extension)) return "Choose a PDF or DOCX document.";
    if (file.size > 20 * 1024 * 1024) return "The document must be smaller than 20 MB.";
    return null;
  },
  async upload(file: File, onProgress?: (progress: number) => void): Promise<{ name: string }> {
    const error = this.validate(file);
    if (error) throw new Error(error);
    for (const progress of [18, 42, 68, 91, 100]) {
      await new Promise((resolve) => window.setTimeout(resolve, 110));
      onProgress?.(progress);
    }
    return { name: file.name };
  },
};
