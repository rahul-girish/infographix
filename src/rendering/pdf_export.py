"""Vector PDF Exporter for Infographix.

Renders a structured PosterDocument to high-resolution, publication-ready PDF
using Matplotlib's vector PDF engine (PdfPages / Figure.savefig).
Handles:
  - Exact canvas aspect ratio and DPI scaling.
  - Multi-line text wrapping for scientific abstract and body text.
  - Distinct colored card backgrounds and accent borders.
  - Styled figure placeholders and tabular grid structures.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

# Ensure matplotlib uses a writable temp cache directory to avoid permissions warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.schemas.layout_document import PosterDocument, PosterElement


CATEGORY_FILL = {
    "title": "#ffffff",
    "author_info": "#ffffff",
    "section": "#e0e7ff",
    "text": "#ffffff",
    "figure": "#f1f5f9",
    "table": "#f8fafc",
    "caption": "#f1f5f9",
    "list": "#ffffff",
    "logo": "#f8fafc",
}

CATEGORY_EDGE = {
    "title": "none",
    "author_info": "none",
    "section": "#2563eb",
    "text": "none",
    "figure": "#cbd5e1",
    "table": "#94a3b8",
    "caption": "#cbd5e1",
    "list": "none",
    "logo": "#94a3b8",
}


def export_to_pdf(
    doc: PosterDocument,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Export a PosterDocument directly to a vector PDF file.

    Args:
        doc: The adapted PosterDocument.
        output_path: Destination path for .pdf file.
        dpi: Dots per inch for raster elements (default 150).

    Returns:
        Path to the saved PDF file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = doc.canvas
    w_px = float(canvas.width)
    h_px = float(canvas.height)

    # Convert canvas pixel dimensions to physical inches
    fig_w_in = w_px / 100.0
    fig_h_in = h_px / 100.0

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)  # Invert Y to match web/screen coordinate system (top-left is (0,0))
    ax.axis("off")

    # Canvas background
    bg_rect = patches.Rectangle(
        (0, 0), w_px, h_px,
        facecolor=canvas.background_color,
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(bg_rect)

    for el in doc.elements:
        x, y, ew, eh = el.bbox
        cat = el.type.lower()
        fill_color = CATEGORY_FILL.get(cat, "#ffffff")
        edge_color = CATEGORY_EDGE.get(cat, "none")
        ec = "none" if edge_color == "none" else edge_color

        # Card container patch
        if fill_color != "#ffffff" or ec != "none":
            card = patches.FancyBboxPatch(
                (x, y), ew, eh,
                boxstyle="round,pad=2,rounding_size=4",
                facecolor=fill_color,
                edgecolor=ec,
                linewidth=1.0 if ec != "none" else 0.0,
                zorder=1,
            )
            ax.add_patch(card)

        # Content rendering
        if cat == "title":
            title_text = str(el.content or doc.title or "Poster Title")
            ax.text(
                x + ew / 2.0, y + eh / 2.0,
                title_text,
                color="#0f172a",
                fontsize=max(12, int(eh * 0.32)),
                fontweight="bold",
                ha="center",
                va="center",
                zorder=2,
                wrap=True,
            )

        elif cat == "author_info":
            author_text = str(el.content or ", ".join(doc.authors))
            ax.text(
                x + ew / 2.0, y + eh / 2.0,
                author_text,
                color="#475569",
                fontsize=max(9, int(eh * 0.40)),
                fontweight="medium",
                ha="center",
                va="center",
                zorder=2,
            )

        elif cat == "section":
            sec_text = str(el.content or "SECTION HEADING").upper()
            # Left accent bar
            bar = patches.Rectangle(
                (x, y + 2), 6.0, eh - 4,
                facecolor="#2563eb", edgecolor="none", zorder=3,
            )
            ax.add_patch(bar)
            ax.text(
                x + 14.0, y + eh / 2.0,
                sec_text,
                color="#1e3a8a",
                fontsize=max(9, int(eh * 0.45)),
                fontweight="bold",
                ha="left",
                va="center",
                zorder=3,
            )

        elif cat == "text":
            text_str = str(el.content or "")
            # Estimate wrap width
            avg_char_w = 6.0
            wrap_chars = max(20, int(ew / avg_char_w))
            wrapped = textwrap.fill(text_str, width=wrap_chars)
            ax.text(
                x + 6.0, y + 6.0,
                wrapped,
                color="#334155",
                fontsize=8.5,
                ha="left",
                va="top",
                zorder=2,
            )

        elif cat == "caption":
            cap_str = str(el.content or "Figure Caption")
            wrap_chars = max(20, int(ew / 5.5))
            wrapped = textwrap.fill(cap_str, width=wrap_chars)
            ax.text(
                x + 8.0, y + eh / 2.0,
                wrapped,
                color="#64748b",
                fontsize=7.5,
                fontstyle="italic",
                ha="left",
                va="center",
                zorder=2,
            )

        elif cat == "figure":
            # Draw schematic diagram icon
            cx = x + ew / 2.0
            cy = y + eh / 2.0
            fig_title = "Figure"
            if isinstance(el.content, dict) and "title" in el.content:
                fig_title = str(el.content["title"])
            elif isinstance(el.content, str) and el.content:
                fig_title = el.content[:25]

            # Circle graphic
            circle = plt.Circle((cx, cy), min(ew, eh) * 0.22, color="#38bdf8", alpha=0.3, zorder=2)
            ax.add_patch(circle)
            ax.text(
                cx, cy,
                fig_title,
                color="#0284c7",
                fontsize=9,
                fontweight="semibold",
                ha="center",
                va="center",
                zorder=3,
            )

        elif cat == "table":
            headers = ["Variable", "Value", "Baseline"]
            if isinstance(el.content, dict) and "headers" in el.content:
                headers = [str(h) for h in el.content["headers"]]
            cols_n = max(1, len(headers))
            col_w = ew / cols_n
            # Header line
            ax.text(
                x + 8.0, y + 10.0,
                " | ".join(headers),
                color="#1e293b",
                fontsize=8.0,
                fontweight="bold",
                ha="left",
                va="top",
                zorder=2,
            )
            ax.text(
                x + 8.0, y + 24.0,
                "Data Matrix / Statistical Summary",
                color="#64748b",
                fontsize=7.5,
                ha="left",
                va="top",
                zorder=2,
            )

    fig.tight_layout(pad=0)
    fig.savefig(out_path, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_to_image(
    doc: PosterDocument,
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Export a PosterDocument directly to a PNG raster image."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use export_to_pdf-like rendering logic for PNG
    w_px = float(doc.canvas.width)
    h_px = float(doc.canvas.height)
    fig_w_in = w_px / 100.0
    fig_h_in = h_px / 100.0

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)
    ax.axis("off")

    bg_rect = patches.Rectangle((0, 0), w_px, h_px, facecolor=doc.canvas.background_color, edgecolor="none", zorder=0)
    ax.add_patch(bg_rect)

    for el in doc.elements:
        x, y, ew, eh = el.bbox
        cat = el.type.lower()
        fill_color = CATEGORY_FILL.get(cat, "#ffffff")
        edge_color = CATEGORY_EDGE.get(cat, "none")
        ec = "none" if edge_color == "none" else edge_color

        if fill_color != "#ffffff" or ec != "none":
            card = patches.FancyBboxPatch(
                (x, y), ew, eh,
                boxstyle="round,pad=2,rounding_size=4",
                facecolor=fill_color,
                edgecolor=ec,
                linewidth=1.0 if ec != "none" else 0.0,
                zorder=1,
            )
            ax.add_patch(card)

        if cat == "title":
            title_text = str(el.content or doc.title or "Poster Title")
            ax.text(x + ew / 2.0, y + eh / 2.0, title_text, color="#0f172a", fontsize=max(11, int(eh * 0.3)), fontweight="bold", ha="center", va="center", zorder=2)
        elif cat == "author_info":
            author_text = str(el.content or ", ".join(doc.authors))
            ax.text(x + ew / 2.0, y + eh / 2.0, author_text, color="#475569", fontsize=max(8, int(eh * 0.35)), ha="center", va="center", zorder=2)
        elif cat == "section":
            sec_text = str(el.content or "SECTION").upper()
            bar = patches.Rectangle((x, y + 2), 6.0, eh - 4, facecolor="#2563eb", edgecolor="none", zorder=3)
            ax.add_patch(bar)
            ax.text(x + 14.0, y + eh / 2.0, sec_text, color="#1e3a8a", fontsize=max(9, int(eh * 0.45)), fontweight="bold", ha="left", va="center", zorder=3)
        elif cat == "text":
            text_str = str(el.content or "")
            wrap_chars = max(20, int(ew / 6.0))
            wrapped = textwrap.fill(text_str, width=wrap_chars)
            ax.text(x + 6.0, y + 6.0, wrapped, color="#334155", fontsize=8.0, ha="left", va="top", zorder=2)
        elif cat == "caption":
            cap_str = str(el.content or "Figure Caption")
            wrap_chars = max(20, int(ew / 5.5))
            wrapped = textwrap.fill(cap_str, width=wrap_chars)
            ax.text(x + 8.0, y + eh / 2.0, wrapped, color="#64748b", fontsize=7.5, fontstyle="italic", ha="left", va="center", zorder=2)
        elif cat == "figure":
            cx = x + ew / 2.0
            cy = y + eh / 2.0
            fig_title = "Figure"
            if isinstance(el.content, dict) and "title" in el.content:
                fig_title = str(el.content["title"])
            elif isinstance(el.content, str) and el.content:
                fig_title = el.content[:25]
            circle = plt.Circle((cx, cy), min(ew, eh) * 0.22, color="#38bdf8", alpha=0.3, zorder=2)
            ax.add_patch(circle)
            ax.text(cx, cy, fig_title, color="#0284c7", fontsize=9, fontweight="semibold", ha="center", va="center", zorder=3)
        elif cat == "table":
            ax.text(x + 8.0, y + 10.0, "Data Table", color="#1e293b", fontsize=8.0, fontweight="bold", ha="left", va="top", zorder=2)

    fig.tight_layout(pad=0)
    fig.savefig(out_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
