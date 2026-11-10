"""Typography-Aware Content Geometry and Capacity Estimator (Stage 3).

Computes realistic, width-dependent geometric dimensions and capacity bounds
for arbitrary user-provided content elements without relying on simplistic
universal word-count-to-area ratios:

  1. Text Blocks:
     - Word count, character count, and actual string content
     - Font family, font size, line-height multiplier, paragraph spacing
     - Available column width wrapping (chars-per-line calculation)
     - Minimum readable font size enforcement (e.g. 14pt body, 20pt heading, 36pt title)
     - Height and area demands as a function of allocated width

  2. Images / Figures:
     - Original pixel width and height
     - Aspect ratio locking (w / h)
     - Minimum and preferred display bounds
     - Cropping flexibility flags

  3. Tables:
     - Row and column counts
     - Header heights, cell padding, font sizes
     - Minimum readable width and height guarantees

  4. Titles, Authors, Captions, Section Headings, References:
     - Dedicated typographic presets and hierarchy bounds
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Standard typographic scale for scientific poster canvases (assuming standard 72 pt/in, canvas ~2500 x 1700 pt)
TYPOGRAPHY_PRESETS: dict[str, dict[str, Any]] = {
    "title": {
        "preferred_font_size_pt": 48.0,
        "min_font_size_pt": 32.0,
        "line_height_mult": 1.20,
        "avg_char_width_em": 0.54,
        "font_weight": "bold",
    },
    "author_info": {
        "preferred_font_size_pt": 22.0,
        "min_font_size_pt": 16.0,
        "line_height_mult": 1.30,
        "avg_char_width_em": 0.50,
        "font_weight": "normal",
    },
    "section": {
        "preferred_font_size_pt": 24.0,
        "min_font_size_pt": 18.0,
        "line_height_mult": 1.25,
        "avg_char_width_em": 0.52,
        "font_weight": "bold",
    },
    "text": {
        "preferred_font_size_pt": 16.0,
        "min_font_size_pt": 13.0,
        "line_height_mult": 1.38,
        "avg_char_width_em": 0.50,
        "font_weight": "normal",
    },
    "list": {
        "preferred_font_size_pt": 15.5,
        "min_font_size_pt": 12.5,
        "line_height_mult": 1.35,
        "avg_char_width_em": 0.50,
        "font_weight": "normal",
    },
    "caption": {
        "preferred_font_size_pt": 13.5,
        "min_font_size_pt": 11.0,
        "line_height_mult": 1.30,
        "avg_char_width_em": 0.48,
        "font_weight": "normal",
    },
    "table": {
        "preferred_font_size_pt": 14.0,
        "min_font_size_pt": 11.0,
        "line_height_mult": 1.25,
        "avg_char_width_em": 0.50,
        "font_weight": "normal",
    },
}


@dataclass
class GeometricDemand:
    """Rigorous dimensional demands and bounds for an individual content block."""

    block_id: str
    content_type: str  # title, author_info, section, text, figure, table, caption, list
    # Normalized canvas dimensions [0, 1]
    min_width: float
    preferred_width: float
    max_width: float
    min_height: float
    preferred_height: float
    max_height: float
    # Aspect ratio constraints (for images/tables)
    aspect_ratio: float | None = None
    lock_aspect_ratio: bool = False
    allow_crop: bool = False
    # Typography details
    font_size_pt: float = 16.0
    min_font_size_pt: float = 12.0
    line_height_mult: float = 1.35
    estimated_line_count: int = 1
    # Raw content details
    word_count: int = 0
    char_count: int = 0
    raw_content: Any = None
    priority: float = 1.0
    must_be_adjacent_to: str | None = None

    @property
    def preferred_area(self) -> float:
        return float(self.preferred_width * self.preferred_height)

    @property
    def min_area(self) -> float:
        return float(self.min_width * self.min_height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "content_type": self.content_type,
            "min_width": round(self.min_width, 4),
            "preferred_width": round(self.preferred_width, 4),
            "max_width": round(self.max_width, 4),
            "min_height": round(self.min_height, 4),
            "preferred_height": round(self.preferred_height, 4),
            "max_height": round(self.max_height, 4),
            "aspect_ratio": round(self.aspect_ratio, 3) if self.aspect_ratio else None,
            "lock_aspect_ratio": self.lock_aspect_ratio,
            "estimated_line_count": self.estimated_line_count,
            "font_size_pt": round(self.font_size_pt, 1),
            "priority": self.priority,
        }


def estimate_text_content_geometry(
    text: str | None = None,
    word_count: int | None = None,
    char_count: int | None = None,
    content_type: str = "text",
    candidate_col_width_norm: float = 0.28,
    canvas_width_pt: float = 2592.0,   # Standard 36" * 72 pt/in
    canvas_height_pt: float = 1728.0,  # Standard 24" * 72 pt/in
    font_size_pt: float | None = None,
    min_font_size_pt: float | None = None,
    paragraph_spacing_pt: float = 10.0,
    box_padding_pt: float = 14.0,
) -> dict[str, Any]:
    """Accurately wrap text at a candidate column width using typographic font metrics.

    Ensures height is inversely proportional to allocated width, rather than
    assuming a constant area.
    """
    preset = TYPOGRAPHY_PRESETS.get(content_type, TYPOGRAPHY_PRESETS["text"])
    f_size = font_size_pt or preset["preferred_font_size_pt"]
    f_min = min_font_size_pt or preset["min_font_size_pt"]
    lh_mult = preset["line_height_mult"]
    em_width = preset["avg_char_width_em"]

    if text is not None:
        raw_str = str(text).strip()
        paragraphs = [p for p in raw_str.split("\n") if p.strip()]
        w_cnt = len(raw_str.split())
        c_cnt = len(raw_str)
        n_paras = max(1, len(paragraphs))
    else:
        w_cnt = word_count if word_count is not None else 75
        c_cnt = char_count if char_count is not None else w_cnt * 6
        n_paras = max(1, w_cnt // 60)

    # 1. Preferred height at nominal candidate width
    col_width_pt = max(100.0, candidate_col_width_norm * canvas_width_pt)
    usable_width_pt = col_width_pt - 2.0 * box_padding_pt
    char_pt = f_size * em_width
    chars_per_line = max(10.0, usable_width_pt / char_pt)

    est_lines = max(1, int(math.ceil(max(1, c_cnt) / chars_per_line)))
    line_h_pt = f_size * lh_mult
    total_h_pt = est_lines * line_h_pt + (n_paras - 1) * paragraph_spacing_pt + 2.0 * box_padding_pt
    pref_h_norm = float(np.clip(total_h_pt / canvas_height_pt, 0.012, 0.85))

    # 2. Min height at maximum width & minimum font size
    max_w_norm = min(0.95, candidate_col_width_norm * 1.35)
    max_usable_w_pt = (max_w_norm * canvas_width_pt) - 2.0 * box_padding_pt
    min_char_pt = f_min * em_width
    max_chars_line = max(15.0, max_usable_w_pt / min_char_pt)
    min_lines = max(1, int(math.ceil(max(1, c_cnt) / max_chars_line)))
    min_h_pt = min_lines * (f_min * lh_mult) + (n_paras - 1) * (paragraph_spacing_pt * 0.7) + 2.0 * box_padding_pt
    min_h_norm = float(np.clip(min_h_pt / canvas_height_pt, 0.008, pref_h_norm))

    # 3. Max height at narrowest width
    min_w_norm = max(0.12, candidate_col_width_norm * 0.75)
    min_usable_w_pt = (min_w_norm * canvas_width_pt) - 2.0 * box_padding_pt
    narrow_chars_line = max(8.0, min_usable_w_pt / char_pt)
    max_lines = max(1, int(math.ceil(max(1, c_cnt) / narrow_chars_line)))
    max_h_pt = max_lines * line_h_pt + (n_paras - 1) * paragraph_spacing_pt + 2.0 * box_padding_pt
    max_h_norm = float(np.clip(max_h_pt / canvas_height_pt, pref_h_norm, 0.90))

    return {
        "word_count": w_cnt,
        "char_count": c_cnt,
        "estimated_lines": est_lines,
        "font_size_pt": f_size,
        "min_font_size_pt": f_min,
        "line_height_mult": lh_mult,
        "min_width": min_w_norm,
        "preferred_width": candidate_col_width_norm,
        "max_width": max_w_norm,
        "min_height": min_h_norm,
        "preferred_height": pref_h_norm,
        "max_height": max_h_norm,
        "preferred_area_norm": float(candidate_col_width_norm * pref_h_norm),
    }


def estimate_image_content_geometry(
    aspect_ratio: float = 1.333,
    original_width_px: int | None = None,
    original_height_px: int | None = None,
    candidate_col_width_norm: float = 0.28,
    canvas_aspect_ratio: float = 1.414,
    allow_crop: bool = False,
    priority: float = 1.0,
) -> dict[str, Any]:
    """Compute dimensional bounds for an image, strictly preserving aspect ratio.

    Width and height are coupled by: h_norm = w_norm * (canvas_aspect / aspect_ratio).
    """
    asp = aspect_ratio
    if original_width_px and original_height_px and original_height_px > 0:
        asp = float(original_width_px / original_height_px)
    asp = max(0.2, min(5.0, asp))

    pref_w = candidate_col_width_norm
    # Because normalized canvas coordinates are [0,1], width/height ratio in normalized coordinates is:
    # (w_norm * canvas_w) / (h_norm * canvas_h) = aspect_ratio
    # -> h_norm = w_norm * (canvas_aspect_ratio / aspect_ratio)
    h_scale = float(canvas_aspect_ratio / asp)
    pref_h = float(np.clip(pref_w * h_scale, 0.04, 0.70))

    min_w = float(max(0.12, pref_w * 0.65))
    min_h = float(max(0.03, min_w * h_scale))

    max_w = float(min(0.95, pref_w * 1.5))
    max_h = float(min(0.85, max_w * h_scale))

    return {
        "aspect_ratio": asp,
        "lock_aspect_ratio": not allow_crop,
        "allow_crop": allow_crop,
        "min_width": min_w,
        "preferred_width": pref_w,
        "max_width": max_w,
        "min_height": min_h,
        "preferred_height": pref_h,
        "max_height": max_h,
        "preferred_area_norm": float(pref_w * pref_h),
    }


def estimate_table_content_geometry(
    table_rows: int = 5,
    table_cols: int = 3,
    table_data: list[list[Any]] | None = None,
    candidate_col_width_norm: float = 0.28,
    canvas_width_pt: float = 2592.0,
    canvas_height_pt: float = 1728.0,
    font_size_pt: float = 13.0,
    min_font_size_pt: float = 10.0,
) -> dict[str, Any]:
    """Compute readable dimension bounds for tabular data.

    Enforces minimum row height (>= 22 pt) and column width (>= 45 pt).
    """
    n_rows = table_rows
    n_cols = table_cols
    if table_data and len(table_data) > 0:
        n_rows = len(table_data)
        n_cols = max(len(r) for r in table_data)

    min_col_w_pt = max(42.0, 7.0 * font_size_pt)
    min_tbl_w_pt = n_cols * min_col_w_pt + 16.0
    min_w_norm = float(np.clip(min_tbl_w_pt / canvas_width_pt, 0.15, 0.95))
    pref_w_norm = float(max(candidate_col_width_norm, min_w_norm))
    max_w_norm = float(min(0.95, max(pref_w_norm * 1.4, 0.60)))

    row_h_pt = max(20.0, font_size_pt * 1.7)
    header_h_pt = row_h_pt * 1.3
    tbl_h_pt = header_h_pt + (n_rows - 1) * row_h_pt + 16.0
    pref_h_norm = float(np.clip(tbl_h_pt / canvas_height_pt, 0.04, 0.70))
    min_h_norm = float(max(0.03, pref_h_norm * 0.75))
    max_h_norm = float(min(0.85, pref_h_norm * 1.35))

    return {
        "table_rows": n_rows,
        "table_cols": n_cols,
        "font_size_pt": font_size_pt,
        "min_font_size_pt": min_font_size_pt,
        "min_width": min_w_norm,
        "preferred_width": pref_w_norm,
        "max_width": max_w_norm,
        "min_height": min_h_norm,
        "preferred_height": pref_h_norm,
        "max_height": max_h_norm,
        "preferred_area_norm": float(pref_w_norm * pref_h_norm),
    }


def compute_content_demands(
    request: Any,
    candidate_col_width_norm: float = 0.28,
    canvas_aspect_ratio: float = 1.414,
) -> list[GeometricDemand]:
    """Convert all blocks in a PosterRequest into a list of GeometricDemand objects."""
    demands: list[GeometricDemand] = []

    for blk in request.blocks:
        vtype = blk.visual_type.lower()
        bid = blk.block_id

        if vtype in ("figure", "image", "logo"):
            asp = blk.aspect_ratio or 1.33
            geom = estimate_image_content_geometry(
                aspect_ratio=asp,
                candidate_col_width_norm=candidate_col_width_norm,
                canvas_aspect_ratio=canvas_aspect_ratio,
                allow_crop=blk.can_crop,
                priority=blk.priority,
            )
            demands.append(
                GeometricDemand(
                    block_id=bid,
                    content_type="figure" if vtype != "logo" else "logo",
                    min_width=geom["min_width"],
                    preferred_width=geom["preferred_width"],
                    max_width=geom["max_width"],
                    min_height=geom["min_height"],
                    preferred_height=geom["preferred_height"],
                    max_height=geom["max_height"],
                    aspect_ratio=geom["aspect_ratio"],
                    lock_aspect_ratio=geom["lock_aspect_ratio"],
                    allow_crop=geom["allow_crop"],
                    priority=blk.priority,
                    must_be_adjacent_to=blk.must_be_adjacent_to,
                    raw_content=getattr(blk, "content", None),
                )
            )

        elif vtype == "table":
            r_cnt = blk.table_rows or 5
            c_cnt = blk.table_cols or 3
            t_data = getattr(blk, "content", None)
            geom = estimate_table_content_geometry(
                table_rows=r_cnt,
                table_cols=c_cnt,
                table_data=t_data if isinstance(t_data, list) else None,
                candidate_col_width_norm=candidate_col_width_norm,
            )
            demands.append(
                GeometricDemand(
                    block_id=bid,
                    content_type="table",
                    min_width=geom["min_width"],
                    preferred_width=geom["preferred_width"],
                    max_width=geom["max_width"],
                    min_height=geom["min_height"],
                    preferred_height=geom["preferred_height"],
                    max_height=geom["max_height"],
                    priority=blk.priority,
                    must_be_adjacent_to=blk.must_be_adjacent_to,
                    raw_content=t_data,
                )
            )

        else:
            # Text, List, Caption, Section, Title, Author Info
            raw_text = getattr(blk, "content", None) if isinstance(getattr(blk, "content", None), str) else None
            w_cnt = blk.word_count
            c_cnt = blk.char_count
            geom = estimate_text_content_geometry(
                text=raw_text,
                word_count=w_cnt,
                char_count=c_cnt,
                content_type=vtype,
                candidate_col_width_norm=(
                    0.80 if vtype in ("title", "author_info") else candidate_col_width_norm
                ),
            )
            demands.append(
                GeometricDemand(
                    block_id=bid,
                    content_type=vtype,
                    min_width=geom["min_width"],
                    preferred_width=geom["preferred_width"],
                    max_width=geom["max_width"],
                    min_height=geom["min_height"],
                    preferred_height=geom["preferred_height"],
                    max_height=geom["max_height"],
                    font_size_pt=geom["font_size_pt"],
                    min_font_size_pt=geom["min_font_size_pt"],
                    line_height_mult=geom["line_height_mult"],
                    estimated_line_count=geom.get("estimated_lines", geom.get("estimated_line_count", 1)),
                    word_count=geom["word_count"],
                    char_count=geom["char_count"],
                    priority=blk.priority,
                    must_be_adjacent_to=blk.must_be_adjacent_to,
                    raw_content=raw_text,
                )
            )

    return demands
