import type { PosterDocument } from "../types";

export const postersService = {
  serialize(poster: PosterDocument): string {
    return JSON.stringify(poster);
  },
  deserialize(serialized: string): PosterDocument {
    const parsed: unknown = JSON.parse(serialized);
    if (!parsed || typeof parsed !== "object" || !("elements" in parsed) || !Array.isArray(parsed.elements)) {
      throw new Error("This saved poster is not valid.");
    }
    return parsed as PosterDocument;
  },
  toSvg(poster: PosterDocument): string {
    const escape = (value: string) => value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    const elements = poster.elements.filter((element) => element.visible).sort((a, b) => a.layer - b.layer).map((element) => {
      const transform = `translate(${element.x} ${element.y}) rotate(${element.rotation})`;
      const opacity = element.opacity ?? 1;
      if (element.type === "text") {
        const fontSize = element.fontSize || 16;
        const maxChars = Math.max(1, Math.floor(element.width / (fontSize * 0.55)));
        const lines = (element.text || "").split("\n").flatMap((line) => {
          const words = line.split(/\s+/);
          const wrapped: string[] = [];
          let current = "";
          for (const word of words) {
            if (current && `${current} ${word}`.length > maxChars) { wrapped.push(current); current = word; }
            else current = current ? `${current} ${word}` : word;
          }
          wrapped.push(current);
          return wrapped;
        });
        const anchor = element.align === "center" ? "middle" : element.align === "right" ? "end" : "start";
        const x = anchor === "middle" ? element.width / 2 : anchor === "end" ? element.width : 0;
        const tspans = lines.map((line, index) => `<tspan x="${x}" dy="${index === 0 ? fontSize : fontSize * (element.lineHeight || 1.2)}">${escape(line)}</tspan>`).join("");
        return `<text transform="${transform}" y="0" fill="${escape(element.fill || "#172033")}" opacity="${opacity}" font-family="${escape(element.fontFamily || "Arial")}" font-size="${fontSize}" font-weight="${element.fontWeight || "normal"}" font-style="${element.fontStyle || "normal"}" text-decoration="${element.underline ? "underline" : "none"}" text-anchor="${anchor}">${tspans}</text>`;
      }
      if (element.type === "image" && element.src) return `<image transform="${transform}" href="${escape(element.src)}" width="${element.width}" height="${element.height}" opacity="${opacity}" preserveAspectRatio="${element.imageFit === "cover" ? "xMidYMid slice" : "xMidYMid meet"}"/>`;
      if (element.type === "circle") return `<ellipse transform="${transform}" cx="${element.width / 2}" cy="${element.height / 2}" rx="${element.width / 2}" ry="${element.height / 2}" fill="${escape(element.fill || "#3458e8")}" stroke="${escape(element.stroke || "none")}" stroke-width="${element.strokeWidth || 0}" opacity="${opacity}"/>`;
      if (element.type === "line" || element.type === "arrow") {
        const points = element.points || [0, 0, element.width, element.height];
        const marker = element.type === "arrow" ? ' marker-end="url(#arrowhead)"' : "";
        return `<polyline transform="${transform}" points="${points.map((point) => point).join(" ")}" fill="none" stroke="${escape(element.stroke || element.fill || "#3458e8")}" stroke-width="${element.strokeWidth || 4}" stroke-linecap="round" stroke-linejoin="round"${marker}/>`;
      }
      const radius = element.name.includes("background") ? 8 : 2;
      return `<rect transform="${transform}" width="${element.width}" height="${element.height}" rx="${radius}" fill="${escape(element.fill || "#3458e8")}" stroke="${escape(element.stroke || "none")}" stroke-width="${element.strokeWidth || 0}" opacity="${opacity}"/>`;
    }).join("");
    return `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${poster.width}" height="${poster.height}" viewBox="0 0 ${poster.width} ${poster.height}"><defs><marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#3458e8"/></marker></defs><rect width="100%" height="100%" fill="${escape(poster.background)}"/>${elements}</svg>`;
  },
};
