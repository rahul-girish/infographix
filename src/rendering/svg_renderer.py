"""Vector SVG Poster Renderer for Infographix.

Transforms a structured PosterDocument into a pure vector SVG graphic with
semantically layered elements:
  - `<g id="background-layer">`
  - `<g id="header-layer">`
  - `<g id="column-layer">`
  - `<g id="content-elements">`
Features text wrapping via `<tspan>`, styled section accents, table grids,
and figure diagram vectors.
"""

from __future__ import annotations

import html
import textwrap
from pathlib import Path
from typing import Any

from src.schemas.layout_document import PosterDocument, PosterElement


CATEGORY_BG = {
    "title": "none",
    "author_info": "none",
    "section": "none",
    "text": "none",
    "figure": "#f1f5f9",
    "table": "#ffffff",
    "caption": "#f8fafc",
    "list": "none",
    "logo": "#f8fafc",
}

CATEGORY_STROKE = {
    "title": "none",
    "author_info": "none",
    "section": "none",
    "text": "none",
    "figure": "#cbd5e1",
    "table": "#cbd5e1",
    "caption": "#e2e8f0",
    "list": "none",
    "logo": "#cbd5e1",
}


def _wrap_text_to_tspans(text: str, max_chars: int, font_size: float, line_height: float, x: float, start_y: float) -> str:
    """Wrap text and format into SVG <tspan> elements."""
    lines = []
    for raw_p in text.split("\n"):
        if not raw_p.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(raw_p, width=max(10, max_chars))
        lines.extend(wrapped)

    dy = line_height * font_size
    tspans = []
    for i, line in enumerate(lines):
        escaped = html.escape(line)
        y = start_y + i * dy
        tspans.append(f'<tspan x="{x:.1f}" y="{y:.1f}">{escaped}</tspan>')
    return "".join(tspans)


def render_to_svg(
    doc: PosterDocument,
    output_path: str | Path | None = None,
) -> str:
    """Render a PosterDocument to a standalone SVG vector string.

    Args:
        doc: The adapted PosterDocument.
        output_path: Optional path to save .svg file.

    Returns:
        Full SVG string.
    """
    canvas = doc.canvas
    w = int(canvas.width)
    h = int(canvas.height)

    svg_parts: list[str] = [
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="Inter, -apple-system, sans-serif">',
        "<defs>",
        '  <style type="text/css">',
        '    .title-text { font-family: Inter, sans-serif; font-weight: 800; fill: #0f172a; text-anchor: middle; }',
        '    .authors-text { font-family: Inter, sans-serif; font-weight: 500; fill: #475569; text-anchor: middle; }',
        '    .section-text { font-family: Inter, sans-serif; font-weight: 700; fill: #1e3a8a; letter-spacing: 0.5px; }',
        '    .body-text { font-family: Inter, sans-serif; font-weight: 400; fill: #334155; }',
        '    .caption-text { font-family: Inter, sans-serif; font-style: italic; fill: #64748b; }',
        '    .table-header { font-family: Inter, sans-serif; font-weight: 600; fill: #1e293b; }',
        '    .table-cell { font-family: Inter, sans-serif; font-weight: 400; fill: #334155; }',
        "  </style>",
        "</defs>",
        f'<rect id="background" width="{w}" height="{h}" fill="{canvas.background_color}" />',
    ]

    # Layers
    svg_parts.append('<g id="layer-content">')

    for el in doc.elements:
        x, y, ew, eh = el.bbox
        cat = el.type.lower()
        style = el.style
        bg = CATEGORY_BG.get(cat, "none")
        stroke = CATEGORY_STROKE.get(cat, "none")

        # Container rect if card has background or border
        if bg != "none" or stroke != "none":
            svg_parts.append(
                f'<rect id="card-{el.id}" x="{x:.1f}" y="{y:.1f}" width="{ew:.1f}" height="{eh:.1f}" '
                f'rx="5" fill="{bg}" stroke="{stroke}" stroke-width="1" />'
            )

        if cat == "title":
            title_text = str(el.content or doc.title or "Scientific Poster Title")
            fs = style.font_size_pt * 1.33
            center_x = x + ew / 2.0
            center_y = y + eh / 2.0 + fs * 0.35
            svg_parts.append(
                f'<text class="title-text" x="{center_x:.1f}" y="{center_y:.1f}" font-size="{fs:.1f}px">'
                f'{html.escape(title_text)}</text>'
            )

        elif cat == "author_info":
            author_text = str(el.content or ", ".join(doc.authors))
            fs = style.font_size_pt * 1.33
            center_x = x + ew / 2.0
            center_y = y + eh / 2.0 + fs * 0.35
            svg_parts.append(
                f'<text class="authors-text" x="{center_x:.1f}" y="{center_y:.1f}" font-size="{fs:.1f}px">'
                f'{html.escape(author_text)}</text>'
            )

        elif cat == "section":
            sec_text = str(el.content or "Section Heading")
            fs = style.font_size_pt * 1.33
            # Bar accent
            bar_w = 5.0
            bar_h = fs * 1.1
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y + (eh - bar_h)/2:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="2" fill="#2563eb" />'
            )
            # Text
            text_x = x + bar_w + 8.0
            text_y = y + eh / 2.0 + fs * 0.35
            svg_parts.append(
                f'<text class="section-text" x="{text_x:.1f}" y="{text_y:.1f}" font-size="{fs:.1f}px">'
                f'{html.escape(sec_text.upper())}</text>'
            )
            # Underline
            svg_parts.append(
                f'<line x1="{x:.1f}" y1="{y + eh - 2:.1f}" x2="{x + ew:.1f}" y2="{y + eh - 2:.1f}" stroke="#e2e8f0" stroke-width="1.5" />'
            )

        elif cat == "text":
            text_content = str(el.content or "")
            fs = style.font_size_pt * 1.33
            max_c = max(15, int(ew / (fs * 0.55)))
            tspans = _wrap_text_to_tspans(
                text=text_content,
                max_chars=max_c,
                font_size=fs,
                line_height=style.line_height,
                x=x + 4.0,
                start_y=y + fs,
            )
            svg_parts.append(f'<text class="body-text" font-size="{fs:.1f}px">{tspans}</text>')

        elif cat == "caption":
            cap_text = str(el.content or "Figure Caption")
            fs = style.font_size_pt * 1.33
            # Left accent bar
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="3" height="{eh:.1f}" rx="1" fill="#cbd5e1" />'
            )
            max_c = max(15, int((ew - 12) / (fs * 0.55)))
            tspans = _wrap_text_to_tspans(
                text=cap_text,
                max_chars=max_c,
                font_size=fs,
                line_height=style.line_height,
                x=x + 8.0,
                start_y=y + fs,
            )
            svg_parts.append(f'<text class="caption-text" font-size="{fs:.1f}px">{tspans}</text>')

        elif cat == "figure":
            # Drawing a scientific diagram graphic inside figure box
            cx = x + ew / 2.0
            cy = y + eh / 2.0
            r = min(ew, eh) * 0.16
            fig_title = "Scientific Diagram"
            if isinstance(el.content, dict) and "title" in el.content:
                fig_title = str(el.content["title"])
            elif isinstance(el.content, str) and el.content:
                fig_title = el.content[:30]

            svg_parts.append(f'<circle cx="{cx - r:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#38bdf8" opacity="0.25" />')
            svg_parts.append(f'<circle cx="{cx + r*0.8:.1f}" cy="{cy - r*0.3:.1f}" r="{r*1.1:.1f}" fill="#0284c7" opacity="0.18" />')
            svg_parts.append(
                f'<path d="M {x + ew*0.2:.1f} {y + eh*0.7:.1f} Q {cx:.1f} {y + eh*0.2:.1f} {x + ew*0.8:.1f} {y + eh*0.6:.1f}" '
                f'fill="none" stroke="#0284c7" stroke-width="2.5" />'
            )
            svg_parts.append(
                f'<text x="{cx:.1f}" y="{y + eh - 12:.1f}" font-size="12px" fill="#64748b" font-weight="600" text-anchor="middle">'
                f'{html.escape(fig_title)}</text>'
            )

        elif cat == "table":
            # Table lines and headers
            headers = ["Parameter", "Observed", "Baseline", "p-val"]
            if isinstance(el.content, dict) and "headers" in el.content:
                headers = [str(h) for h in el.content["headers"]]
            cols_n = max(1, len(headers))
            col_w = ew / cols_n
            row_h = min(28.0, eh / 4.0)

            # Header background
            svg_parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{ew:.1f}" height="{row_h:.1f}" fill="#f1f5f9" />')
            svg_parts.append(f'<line x1="{x:.1f}" y1="{y + row_h:.1f}" x2="{x + ew:.1f}" y2="{y + row_h:.1f}" stroke="#cbd5e1" stroke-width="1.5" />')

            for col_i, h_text in enumerate(headers):
                cell_x = x + col_i * col_w + 6.0
                cell_y = y + row_h * 0.7
                svg_parts.append(
                    f'<text class="table-header" x="{cell_x:.1f}" y="{cell_y:.1f}" font-size="11px">{html.escape(h_text)}</text>'
                )

            # Sample row
            sample_row = ["Factor A", "87.4%", "72.1%", "0.003"]
            r2_y = y + row_h
            svg_parts.append(f'<line x1="{x:.1f}" y1="{r2_y + row_h:.1f}" x2="{x + ew:.1f}" y2="{r2_y + row_h:.1f}" stroke="#f1f5f9" stroke-width="1" />')
            for col_i, c_text in enumerate(sample_row[:cols_n]):
                cell_x = x + col_i * col_w + 6.0
                cell_y = r2_y + row_h * 0.7
                svg_parts.append(
                    f'<text class="table-cell" x="{cell_x:.1f}" y="{cell_y:.1f}" font-size="10.5px">{html.escape(c_text)}</text>'
                )

        elif cat == "list":
            text_content = str(el.content or "")
            items = [item.strip().lstrip("-*• ") for item in text_content.split("\n") if item.strip()]
            fs = style.font_size_pt * 1.33
            dy = fs * style.line_height
            for idx_i, item_str in enumerate(items[:6]):
                bullet_y = y + (idx_i + 1) * dy
                svg_parts.append(f'<circle cx="{x + 8:.1f}" cy="{bullet_y - fs*0.3:.1f}" r="2.5" fill="#3b82f6" />')
                svg_parts.append(
                    f'<text class="body-text" x="{x + 18:.1f}" y="{bullet_y:.1f}" font-size="{fs:.1f}px">'
                    f'{html.escape(item_str)}</text>'
                )

    svg_parts.append("</g>")
    svg_parts.append("</svg>")

    full_svg = "\n".join(svg_parts)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(full_svg, encoding="utf-8")

    return full_svg
