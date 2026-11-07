import type { PosterElement, PosterTemplate } from "../types";

export const posterTemplates: PosterTemplate[] = [
  { id: "research", name: "Research Paper Poster", category: "Academic", description: "A balanced layout for academic research.", accent: "#3458e8", background: "#ffffff", columns: 3 },
  { id: "conference", name: "Scientific Conference Poster", category: "Science", description: "A clean, high-contrast conference layout.", accent: "#7048db", background: "#fbfaff", columns: 3 },
  { id: "engineering", name: "Engineering Project Poster", category: "Engineering", description: "A structured technical project showcase.", accent: "#087f9b", background: "#f7fcfd", columns: 2 },
  { id: "ai-ml", name: "AI/ML Research Poster", category: "Artificial intelligence", description: "A modern visual system for data-led work.", accent: "#8a45cf", background: "#fcfaff", columns: 3 },
  { id: "experimental", name: "Experimental Results Poster", category: "Results", description: "Bring methods, results and findings forward.", accent: "#14866d", background: "#f8fcfa", columns: 3 },
];

export function createStarterElements(title = "A NEW PERSPECTIVE ON RESEARCH", accent = "#3458e8"): PosterElement[] {
  return [
    { id: crypto.randomUUID(), type: "rect", name: "Header background", x: 40, y: 35, width: 920, height: 128, rotation: 0, layer: 0, visible: true, locked: false, fill: accent, opacity: 1 },
    { id: crypto.randomUUID(), type: "text", name: "Poster title", text: title, x: 64, y: 54, width: 865, height: 52, rotation: 0, layer: 1, visible: true, locked: false, fill: "#ffffff", fontFamily: "Arial", fontSize: 32, fontWeight: "bold", align: "center", lineHeight: 1.1 },
    { id: crypto.randomUUID(), type: "text", name: "Authors", text: "Researcher Name · Research Institute", x: 70, y: 112, width: 850, height: 30, rotation: 0, layer: 2, visible: true, locked: false, fill: "#e7ecff", fontFamily: "Arial", fontSize: 16, align: "center" },
    { id: crypto.randomUUID(), type: "text", name: "Introduction heading", text: "INTRODUCTION", x: 55, y: 200, width: 260, height: 34, rotation: 0, layer: 3, visible: true, locked: false, fill: accent, fontFamily: "Arial", fontSize: 19, fontWeight: "bold" },
    { id: crypto.randomUUID(), type: "text", name: "Introduction copy", text: "Set the context for your research question and explain why this work matters.", x: 55, y: 244, width: 260, height: 115, rotation: 0, layer: 4, visible: true, locked: false, fill: "#334155", fontFamily: "Arial", fontSize: 15, lineHeight: 1.45 },
    { id: crypto.randomUUID(), type: "text", name: "Methodology heading", text: "METHODOLOGY", x: 370, y: 200, width: 260, height: 34, rotation: 0, layer: 5, visible: true, locked: false, fill: accent, fontFamily: "Arial", fontSize: 19, fontWeight: "bold" },
    { id: crypto.randomUUID(), type: "rect", name: "Figure placeholder", x: 370, y: 244, width: 260, height: 170, rotation: 0, layer: 6, visible: true, locked: false, fill: "#eef2ff", stroke: "#d7ddf6", strokeWidth: 2, opacity: 1 },
    { id: crypto.randomUUID(), type: "text", name: "Figure caption", text: "Add a chart, figure or diagram", x: 390, y: 311, width: 220, height: 40, rotation: 0, layer: 7, visible: true, locked: false, fill: "#64748b", fontFamily: "Arial", fontSize: 15, align: "center" },
    { id: crypto.randomUUID(), type: "text", name: "Results heading", text: "KEY FINDINGS", x: 685, y: 200, width: 260, height: 34, rotation: 0, layer: 8, visible: true, locked: false, fill: accent, fontFamily: "Arial", fontSize: 19, fontWeight: "bold" },
    { id: crypto.randomUUID(), type: "text", name: "Results copy", text: "Summarize your most important results. Use concise statements and supporting evidence.", x: 685, y: 244, width: 260, height: 115, rotation: 0, layer: 9, visible: true, locked: false, fill: "#334155", fontFamily: "Arial", fontSize: 15, lineHeight: 1.45 },
    { id: crypto.randomUUID(), type: "rect", name: "Footer accent", x: 55, y: 620, width: 890, height: 5, rotation: 0, layer: 10, visible: true, locked: false, fill: accent, opacity: 0.8 },
    { id: crypto.randomUUID(), type: "text", name: "Conclusion", text: "CONCLUSION   ·   Add the takeaway your audience should remember.", x: 55, y: 648, width: 890, height: 38, rotation: 0, layer: 11, visible: true, locked: false, fill: "#26334b", fontFamily: "Arial", fontSize: 17, fontWeight: "bold" },
  ];
}
