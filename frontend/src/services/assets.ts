import { apiClient } from "./apiClient";

export const assetsService = {
  async readImage(file: File, projectId?: string): Promise<string> {
    if (!file.type.startsWith("image/")) throw new Error("Choose an image file.");
    
    try {
      const formData = new FormData();
      formData.append("file", file);
      if (projectId) formData.append("project_id", projectId);

      const res = await apiClient.post<{ url: string }>("/assets", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return res.data.url;
    } catch {
      // Fallback to local Data URL if offline or network error
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => typeof reader.result === "string" ? resolve(reader.result) : reject(new Error("Could not read the selected image."));
        reader.onerror = () => reject(new Error("Could not read the selected image."));
        reader.readAsDataURL(file);
      });
    }
  },
};
