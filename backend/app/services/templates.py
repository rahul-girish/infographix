import uuid
from typing import Any, List, Dict

POSTER_TEMPLATES: List[Dict[str, Any]] = [
    {"id": "research", "name": "Research Paper Poster", "category": "Academic", "description": "A balanced layout for academic research.", "accent": "#3458e8", "background": "#ffffff", "columns": 3},
    {"id": "conference", "name": "Scientific Conference Poster", "category": "Science", "description": "A clean, high-contrast conference layout.", "accent": "#7048db", "background": "#fbfaff", "columns": 3},
    {"id": "engineering", "name": "Engineering Project Poster", "category": "Engineering", "description": "A structured technical project showcase.", "accent": "#087f9b", "background": "#f7fcfd", "columns": 2},
    {"id": "ai-ml", "name": "AI/ML Research Poster", "category": "Artificial intelligence", "description": "A modern visual system for data-led work.", "accent": "#8a45cf", "background": "#fcfaff", "columns": 3},
    {"id": "experimental", "name": "Experimental Results Poster", "category": "Results", "description": "Bring methods, results and findings forward.", "accent": "#14866d", "background": "#f8fcfa", "columns": 3},
]

DEFAULT_SECTIONS: List[Dict[str, str]] = [
    {"id": "title", "title": "Title", "content": "A new perspective on research"},
    {"id": "authors", "title": "Authors", "content": "Researcher Name, Co-author Name"},
    {"id": "affiliations", "title": "Affiliations", "content": "Department of Research, University"},
    {"id": "abstract", "title": "Abstract", "content": "A concise summary of the research question, approach and key findings."},
    {"id": "introduction", "title": "Introduction", "content": "Introduce the topic and relevant background."},
    {"id": "problem", "title": "Problem statement", "content": "Describe the challenge this work addresses."},
    {"id": "objectives", "title": "Objectives", "content": "List the primary aims of this research."},
    {"id": "methodology", "title": "Methodology", "content": "Explain the methods, materials and process."},
    {"id": "results", "title": "Results", "content": "Present the key observations and data."},
    {"id": "discussion", "title": "Discussion", "content": "Interpret the findings and their implications."},
    {"id": "conclusion", "title": "Conclusion", "content": "Summarize the main takeaway."},
    {"id": "references", "title": "References", "content": "Add sources cited in this poster."},
    {"id": "keywords", "title": "Keywords", "content": "research, science, innovation"},
]

def create_starter_elements(title: str = "A NEW PERSPECTIVE ON RESEARCH", accent: str = "#3458e8") -> List[Dict[str, Any]]:
    return [
        {"id": str(uuid.uuid4()), "type": "rect", "name": "Header background", "x": 40.0, "y": 35.0, "width": 920.0, "height": 128.0, "rotation": 0.0, "layer": 0, "visible": True, "locked": False, "fill": accent, "opacity": 1.0},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Poster title", "text": title, "x": 64.0, "y": 54.0, "width": 865.0, "height": 52.0, "rotation": 0.0, "layer": 1, "visible": True, "locked": False, "fill": "#ffffff", "fontFamily": "Arial", "fontSize": 32.0, "fontWeight": "bold", "align": "center", "lineHeight": 1.1},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Authors", "text": "Researcher Name · Research Institute", "x": 70.0, "y": 112.0, "width": 850.0, "height": 30.0, "rotation": 0.0, "layer": 2, "visible": True, "locked": False, "fill": "#e7ecff", "fontFamily": "Arial", "fontSize": 16.0, "align": "center"},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Introduction heading", "text": "INTRODUCTION", "x": 55.0, "y": 200.0, "width": 260.0, "height": 34.0, "rotation": 0.0, "layer": 3, "visible": True, "locked": False, "fill": accent, "fontFamily": "Arial", "fontSize": 19.0, "fontWeight": "bold"},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Introduction copy", "text": "Set the context for your research question and explain why this work matters.", "x": 55.0, "y": 244.0, "width": 260.0, "height": 115.0, "rotation": 0.0, "layer": 4, "visible": True, "locked": False, "fill": "#334155", "fontFamily": "Arial", "fontSize": 15.0, "lineHeight": 1.45},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Methodology heading", "text": "METHODOLOGY", "x": 370.0, "y": 200.0, "width": 260.0, "height": 34.0, "rotation": 0.0, "layer": 5, "visible": True, "locked": False, "fill": accent, "fontFamily": "Arial", "fontSize": 19.0, "fontWeight": "bold"},
        {"id": str(uuid.uuid4()), "type": "rect", "name": "Figure placeholder", "x": 370.0, "y": 244.0, "width": 260.0, "height": 170.0, "rotation": 0.0, "layer": 6, "visible": True, "locked": False, "fill": "#eef2ff", "stroke": "#d7ddf6", "strokeWidth": 2.0, "opacity": 1.0},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Figure caption", "text": "Add a chart, figure or diagram", "x": 390.0, "y": 311.0, "width": 220.0, "height": 40.0, "rotation": 0.0, "layer": 7, "visible": True, "locked": False, "fill": "#64748b", "fontFamily": "Arial", "fontSize": 15.0, "align": "center"},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Results heading", "text": "KEY FINDINGS", "x": 685.0, "y": 200.0, "width": 260.0, "height": 34.0, "rotation": 0.0, "layer": 8, "visible": True, "locked": False, "fill": accent, "fontFamily": "Arial", "fontSize": 19.0, "fontWeight": "bold"},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Results copy", "text": "Summarize your most important results. Use concise statements and supporting evidence.", "x": 685.0, "y": 244.0, "width": 260.0, "height": 115.0, "rotation": 0.0, "layer": 9, "visible": True, "locked": False, "fill": "#334155", "fontFamily": "Arial", "fontSize": 15.0, "lineHeight": 1.45},
        {"id": str(uuid.uuid4()), "type": "rect", "name": "Footer accent", "x": 55.0, "y": 620.0, "width": 890.0, "height": 5.0, "rotation": 0.0, "layer": 10, "visible": True, "locked": False, "fill": accent, "opacity": 0.8},
        {"id": str(uuid.uuid4()), "type": "text", "name": "Conclusion", "text": "CONCLUSION   ·   Add the takeaway your audience should remember.", "x": 55.0, "y": 648.0, "width": 890.0, "height": 38.0, "rotation": 0.0, "layer": 11, "visible": True, "locked": False, "fill": "#26334b", "fontFamily": "Arial", "fontSize": 17.0, "fontWeight": "bold"},
    ]

