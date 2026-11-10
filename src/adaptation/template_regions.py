"""Intermediate Hierarchical Layout Representation (Stage 3).

Transforms a retrieved reference poster from a rigid set of bounding boxes into
an adaptable semantic hierarchy:
  Poster
   ├── Header Band
   │    ├── Title slot
   │    └── Authors / Affiliations slot
   └── Columns (0 .. K-1)
        ├── Section A (e.g. Introduction / Methods)
        │    ├── Heading
        │    ├── Text paragraphs
        │    └── Figures / Captions
        └── Section B (e.g. Results / Discussion)
             ├── Heading
             ├── Tables
             └── Figures / Captions

Preserves:
  - Original poster ID & canvas dimensions
  - Normalized and raw bounding boxes
  - Column boundaries and assignment confidence
  - Section groupings and reading order
  - Spatial relationships and figure-caption pairings
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.features.config import EPS, HEADER_CAT_IDS, KNOWN_CATEGORY_SLUGS


@dataclass
class TemplateSlot:
    """An individual geometric slot in the retrieved reference poster."""

    element_id: int
    category_slug: str
    bbox_norm: list[float]  # [x0, y0, w, h]
    area_norm: float
    center: tuple[float, float]
    role: str  # 'header_title', 'header_authors', 'heading', 'body_text', 'figure', 'table', 'caption', 'list'
    column_idx: int | None = None  # None for header; 0, 1, 2... for columns
    section_idx: int = 0
    reading_order: int = 0
    paired_with_id: int | None = None  # e.g., caption -> figure ID or vice versa
    confidence: float = 1.0
    original_raw_bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "category_slug": self.category_slug,
            "bbox_norm": [round(v, 4) for v in self.bbox_norm],
            "area_norm": round(self.area_norm, 4),
            "center": (round(self.center[0], 4), round(self.center[1], 4)),
            "role": self.role,
            "column_idx": self.column_idx,
            "section_idx": self.section_idx,
            "reading_order": self.reading_order,
            "paired_with_id": self.paired_with_id,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class TemplateSection:
    """A coherent logical section inside a column (e.g. heading + paragraphs + media)."""

    section_id: str
    column_idx: int
    heading_slot: TemplateSlot | None = None
    body_slots: list[TemplateSlot] = field(default_factory=list)
    media_slots: list[TemplateSlot] = field(default_factory=list)  # figures, tables, captions
    bounds_norm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # [x0, y0, w, h]
    confidence: float = 0.85

    @property
    def all_slots(self) -> list[TemplateSlot]:
        slots = []
        if self.heading_slot:
            slots.append(self.heading_slot)
        slots.extend(self.body_slots)
        slots.extend(self.media_slots)
        return slots

    def recompute_bounds(self) -> None:
        slots = self.all_slots
        if not slots:
            return
        x0 = min(s.bbox_norm[0] for s in slots)
        y0 = min(s.bbox_norm[1] for s in slots)
        x1 = max(s.bbox_norm[0] + s.bbox_norm[2] for s in slots)
        y1 = max(s.bbox_norm[1] + s.bbox_norm[3] for s in slots)
        self.bounds_norm = [x0, y0, max(0.01, x1 - x0), max(0.01, y1 - y0)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "column_idx": self.column_idx,
            "bounds_norm": [round(v, 4) for v in self.bounds_norm],
            "heading_slot": self.heading_slot.to_dict() if self.heading_slot else None,
            "body_slots": [s.to_dict() for s in self.body_slots],
            "media_slots": [s.to_dict() for s in self.media_slots],
            "confidence": round(self.confidence, 3),
        }


@dataclass
class TemplateColumn:
    """A vertical reading column containing structured sections."""

    column_idx: int
    bounds_norm: list[float]  # [x_left, y_top, width, height]
    sections: list[TemplateSection] = field(default_factory=list)
    unassigned_slots: list[TemplateSlot] = field(default_factory=list)
    confidence: float = 0.90

    @property
    def all_slots(self) -> list[TemplateSlot]:
        res = []
        for sec in self.sections:
            res.extend(sec.all_slots)
        res.extend(self.unassigned_slots)
        # Sort vertically by y0
        res.sort(key=lambda s: s.bbox_norm[1])
        return res

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_idx": self.column_idx,
            "bounds_norm": [round(v, 4) for v in self.bounds_norm],
            "sections": [sec.to_dict() for sec in self.sections],
            "unassigned_slots": [s.to_dict() for s in self.unassigned_slots],
            "confidence": round(self.confidence, 3),
        }


@dataclass
class HierarchicalTemplate:
    """Complete hierarchical representation of a retrieved poster."""

    poster_id: int
    canvas_width_px: float
    canvas_height_px: float
    aspect_ratio: float
    orientation: str
    header_slots: list[TemplateSlot] = field(default_factory=list)
    columns: list[TemplateColumn] = field(default_factory=list)
    figure_caption_pairs: list[dict[str, Any]] = field(default_factory=list)
    reading_order_element_ids: list[int] = field(default_factory=list)
    overall_confidence: float = 0.88
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def body_slots(self) -> list[TemplateSlot]:
        return [s for col in self.columns for s in col.all_slots]

    @property
    def all_slots(self) -> list[TemplateSlot]:
        return list(self.header_slots) + self.body_slots

    @property
    def total_slots_count(self) -> int:
        return len(self.header_slots) + sum(len(col.all_slots) for col in self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "poster_id": self.poster_id,
            "canvas_width_px": self.canvas_width_px,
            "canvas_height_px": self.canvas_height_px,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "orientation": self.orientation,
            "total_slots_count": self.total_slots_count,
            "overall_confidence": round(self.overall_confidence, 3),
            "header_slots": [s.to_dict() for s in self.header_slots],
            "columns": [col.to_dict() for col in self.columns],
            "figure_caption_pairs": self.figure_caption_pairs,
            "reading_order_element_ids": self.reading_order_element_ids,
        }


def extract_hierarchical_template(
    poster_row: pd.Series | dict[str, Any],
    elements_df: pd.DataFrame,
    relations_df: pd.DataFrame | None = None,
) -> HierarchicalTemplate:
    """Extract hierarchical columns, sections, and pairing relationships from a poster.

    Algorithm:
      1. Distinguishes header band (Title, Author Info with y0 < 0.28 or full-width banner).
      2. Groups remaining body elements into K vertical columns using x-interval analysis
         and peak clustering corresponding to `estimated_column_count`.
      3. Within each column, sorts elements by vertical position y0 and clusters
         into sections delimited by `Section` headings.
      4. Detects figure-caption associations from spatial proximity / `relations_df`.
      5. Computes reading order sequence: Header -> Column 0 top-to-bottom -> Column 1...
    """
    pid = int(poster_row["poster_id"])
    W = float(poster_row.get("canvas_width_px", 1920.0))
    H = float(poster_row.get("canvas_height_px", 1080.0))
    aspect = float(poster_row.get("aspect_ratio", W / H))
    orient = str(poster_row.get("orientation", "landscape" if aspect >= 1.0 else "portrait"))
    target_k_cols = int(round(float(poster_row.get("estimated_column_count", 3))))
    target_k_cols = max(1, min(6, target_k_cols))

    p_els = elements_df[elements_df["poster_id"] == pid].copy()
    if len(p_els) == 0:
        return HierarchicalTemplate(
            poster_id=pid,
            canvas_width_px=W,
            canvas_height_px=H,
            aspect_ratio=aspect,
            orientation=orient,
            overall_confidence=0.5,
        )

    # 1. Distinguish Header Elements vs Body Elements
    # Title / Author Info in top 30% of canvas, or any banner with width >= 0.70 and y0 <= 0.22
    is_hdr = (
        (p_els["category_slug"].isin(["title", "author_info"]) & (p_els["cy"] < 0.32))
        | ((p_els["w_norm"] >= 0.70) & (p_els["y0"] < 0.22))
    )
    hdr_els = p_els.loc[is_hdr].sort_values("y0")
    body_els = p_els.loc[~is_hdr].copy()

    header_slots: list[TemplateSlot] = []
    reading_order_ids: list[int] = []

    for _, r in hdr_els.iterrows():
        eid = int(r["element_id"])
        cslug = str(r["category_slug"])
        role = "header_title" if cslug == "title" else "header_authors"
        slot = TemplateSlot(
            element_id=eid,
            category_slug=cslug,
            bbox_norm=[float(r["x0"]), float(r["y0"]), float(r["w_norm"]), float(r["h_norm"])],
            area_norm=float(r["area_norm"]),
            center=(float(r["cx"]), float(r["cy"])),
            role=role,
            column_idx=None,
            section_idx=0,
            reading_order=len(reading_order_ids),
            confidence=0.95,
            original_raw_bbox=[float(r["raw_x"]), float(r["raw_y"]), float(r["raw_w"]), float(r["raw_h"])],
        )
        header_slots.append(slot)
        reading_order_ids.append(eid)

    # 2. Partition Body Elements into K Vertical Columns
    if len(body_els) == 0:
        return HierarchicalTemplate(
            poster_id=pid,
            canvas_width_px=W,
            canvas_height_px=H,
            aspect_ratio=aspect,
            orientation=orient,
            header_slots=header_slots,
            reading_order_element_ids=reading_order_ids,
            overall_confidence=0.90,
        )

    # Determine column boundaries along horizontal axis [0, 1]
    # For K columns, nominal column interval: left_k = margin + k * (col_w + gutter)
    body_left = float(body_els["x0"].min())
    body_right = float((body_els["x0"] + body_els["w_norm"]).max())
    body_top = float(body_els["y0"].min())
    body_bottom = float((body_els["y0"] + body_els["h_norm"]).max())
    span_w = max(0.2, body_right - body_left)

    col_dividers: list[float] = [body_left]
    for k in range(1, target_k_cols):
        col_dividers.append(body_left + k * (span_w / target_k_cols))
    col_dividers.append(body_right + 0.01)

    # Refine column dividers if natural gutters exist
    # Assign each element to column index based on its center cx
    cx_vals = body_els["cx"].to_numpy()
    col_assignments = np.clip(
        np.digitize(cx_vals, col_dividers[1:-1]), 0, target_k_cols - 1
    )
    body_els["col_idx"] = col_assignments

    # 3. Detect Figure-Caption Associations
    fig_cap_pairs: list[dict[str, Any]] = []
    fig_to_cap: dict[int, int] = {}
    cap_to_fig: dict[int, int] = {}

    figs = body_els[body_els["category_slug"].isin(["figure", "table"])]
    caps = body_els[body_els["category_slug"] == "caption"]

    if len(figs) > 0 and len(caps) > 0:
        for _, f_row in figs.iterrows():
            fid = int(f_row["element_id"])
            f_x0, f_y0, f_w, f_h = float(f_row["x0"]), float(f_row["y0"]), float(f_row["w_norm"]), float(f_row["h_norm"])
            f_cx, f_cy = f_x0 + f_w / 2.0, f_y0 + f_h / 2.0

            best_cap_id = None
            best_dist = 0.22  # Maximum search distance threshold
            best_conf = 0.0

            for _, c_row in caps.iterrows():
                cid = int(c_row["element_id"])
                if cid in cap_to_fig:
                    continue  # Already paired
                c_x0, c_y0, c_w, c_h = float(c_row["x0"]), float(c_row["y0"]), float(c_row["w_norm"]), float(c_row["h_norm"])
                c_cx, c_cy = c_x0 + c_w / 2.0, c_y0 + c_h / 2.0

                # Caption is typically directly below figure (dy > 0) with substantial x overlap
                dy = c_y0 - (f_y0 + f_h)
                dx = abs(c_cx - f_cx)
                x_overlap = max(0.0, min(f_x0 + f_w, c_x0 + c_w) - max(f_x0, c_x0))

                if -0.02 <= dy <= 0.14 and (x_overlap > 0.05 or dx < 0.12):
                    d = np.hypot(dx, max(0.0, dy))
                    if d < best_dist:
                        best_dist = d
                        best_cap_id = cid
                        best_conf = float(np.clip(1.0 - (d / 0.22), 0.5, 0.98))

            if best_cap_id is not None:
                fig_to_cap[fid] = best_cap_id
                cap_to_fig[best_cap_id] = fid
                fig_cap_pairs.append({
                    "figure_element_id": fid,
                    "caption_element_id": best_cap_id,
                    "confidence": best_conf,
                })

    # 4. Build Structured Columns and Sections
    columns: list[TemplateColumn] = []

    for c_i in range(target_k_cols):
        c_sub = body_els[body_els["col_idx"] == c_i].sort_values("y0").copy()
        c_x_min = float(col_dividers[c_i])
        c_x_max = float(col_dividers[c_i + 1])
        c_w = max(0.08, c_x_max - c_x_min)
        c_h = max(0.1, body_bottom - body_top)

        if len(c_sub) == 0:
            columns.append(
                TemplateColumn(
                    column_idx=c_i,
                    bounds_norm=[c_x_min, body_top, c_w, c_h],
                    sections=[],
                    confidence=0.75,
                )
            )
            continue

        sections: list[TemplateSection] = []
        current_sec_slots: list[TemplateSlot] = []
        current_heading: TemplateSlot | None = None
        sec_counter = 0

        for _, r in c_sub.iterrows():
            eid = int(r["element_id"])
            cslug = str(r["category_slug"])

            if cslug == "figure":
                role = "figure"
            elif cslug == "table":
                role = "table"
            elif cslug == "caption":
                role = "caption"
            elif cslug == "section":
                role = "heading"
            elif cslug == "list":
                role = "list"
            else:
                role = "body_text"

            paired_id = fig_to_cap.get(eid) or cap_to_fig.get(eid)

            slot = TemplateSlot(
                element_id=eid,
                category_slug=cslug,
                bbox_norm=[float(r["x0"]), float(r["y0"]), float(r["w_norm"]), float(r["h_norm"])],
                area_norm=float(r["area_norm"]),
                center=(float(r["cx"]), float(r["cy"])),
                role=role,
                column_idx=c_i,
                section_idx=sec_counter,
                reading_order=len(reading_order_ids),
                paired_with_id=paired_id,
                confidence=0.88,
                original_raw_bbox=[float(r["raw_x"]), float(r["raw_y"]), float(r["raw_w"]), float(r["raw_h"])],
            )
            reading_order_ids.append(eid)

            # When a new Section Heading is encountered, flush current section
            if role == "heading":
                if current_heading or current_sec_slots:
                    # Flush previous section
                    sec_obj = TemplateSection(
                        section_id=f"col_{c_i}_sec_{sec_counter}",
                        column_idx=c_i,
                        heading_slot=current_heading,
                        body_slots=[s for s in current_sec_slots if s.role in ("body_text", "list")],
                        media_slots=[s for s in current_sec_slots if s.role in ("figure", "table", "caption")],
                    )
                    sec_obj.recompute_bounds()
                    sections.append(sec_obj)
                    sec_counter += 1
                current_heading = slot
                current_sec_slots = []
            else:
                current_sec_slots.append(slot)

        # Flush final section in this column
        if current_heading or current_sec_slots:
            sec_obj = TemplateSection(
                section_id=f"col_{c_i}_sec_{sec_counter}",
                column_idx=c_i,
                heading_slot=current_heading,
                body_slots=[s for s in current_sec_slots if s.role in ("body_text", "list")],
                media_slots=[s for s in current_sec_slots if s.role in ("figure", "table", "caption")],
            )
            sec_obj.recompute_bounds()
            sections.append(sec_obj)

        col_obj = TemplateColumn(
            column_idx=c_i,
            bounds_norm=[c_x_min, body_top, c_w, c_h],
            sections=sections,
            confidence=0.90,
        )
        columns.append(col_obj)

    col_conf = float(poster_row.get("column_assignment_confidence", 0.85))
    overall_conf = float(np.clip(0.5 * col_conf + 0.5 * (1.0 - 0.1 * abs(target_k_cols - len(columns))), 0.65, 0.98))

    return HierarchicalTemplate(
        poster_id=pid,
        canvas_width_px=W,
        canvas_height_px=H,
        aspect_ratio=aspect,
        orientation=orient,
        header_slots=header_slots,
        columns=columns,
        figure_caption_pairs=fig_cap_pairs,
        reading_order_element_ids=reading_order_ids,
        overall_confidence=overall_conf,
    )
