export interface LoadedImageResult {
  dataUrl: string;
  width: number;
  height: number;
  aspectRatio: number;
}

export const assetsService = {
  readImage(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      if (!file.type.startsWith("image/")) {
        reject(new Error("Please select a valid image file (PNG, JPG, WEBP, or SVG)."));
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        reject(new Error("Image must be smaller than 10 MB."));
        return;
      }
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("Could not read this image file."));
      reader.readAsDataURL(file);
    });
  },

  async readImageWithMetadata(file: File): Promise<LoadedImageResult> {
    const dataUrl = await this.readImage(file);
    return new Promise((resolve, reject) => {
      const img = new window.Image();
      img.onload = () => {
        const w = Math.max(1, img.naturalWidth);
        const h = Math.max(1, img.naturalHeight);
        resolve({
          dataUrl,
          width: w,
          height: h,
          aspectRatio: Number((w / h).toFixed(3)),
        });
      };
      img.onerror = () => reject(new Error("Could not decode this image."));
      img.src = dataUrl;
    });
  },
};
