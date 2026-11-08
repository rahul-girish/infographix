export const assetsService = {
  async readImage(file: File): Promise<string> {
    if (!file.type.startsWith("image/")) throw new Error("Choose an image file.");
    if (file.size > 1024 * 1024) throw new Error("Images must be smaller than 1 MB to fit browser storage.");
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => typeof reader.result === "string" ? resolve(reader.result) : reject(new Error("Could not read the selected image."));
      reader.onerror = () => reject(new Error("Could not read the selected image."));
      reader.readAsDataURL(file);
    });
  },
};
