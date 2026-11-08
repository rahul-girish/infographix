"""Semantic Slot Assignment and Dynamic Region Splitting / Merging (Stage 3).

Assigns incoming user content blocks to slots of the retrieved hierarchical template
while handling capacity differences:
  1. Semantic slot assignment matching:
     - Header Title & Authors -> Header slots
     - Section headings + associated paragraphs -> kept together sequentially
     - Figures + associated captions -> kept together as coupled units
     - Tables -> allocated to high-capacity slots
  2. Dynamic Region Splitting & Merging:
     - If user provides more figures/tables than template slots, splits available
       body/media slots vertically or horizontally.
     - If user provides fewer blocks, merges adjacent empty slots or contracts
       whitespace gracefully.
     - Preserves the overall reading order and column balance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.adaptation.content_geometry import GeometricDemand
from src.adaptation.template_regions import HierarchicalTemplate, TemplateSlot


@dataclass
class AssignedSlot:
    """A user content demand mapped to a target layout region."""

    block_id: str
    content_type: str
    demand: GeometricDemand
    target_bbox_norm: list[float]  # [x0, y0, w, h] normalized in [0, 1]
    column_idx: int | None
    group_id: str
    reading_order: int
    paired_with_block_id: str | None = None
    is_split_slot: bool = False
    source_element_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "content_type": self.content_type,
            "target_bbox_norm": [round(v, 4) for v in self.target_bbox_norm],
            "column_idx": self.column_idx,
            "group_id": self.group_id,
            "reading_order": self.reading_order,
            "paired_with_block_id": self.paired_with_block_id,
            "is_split_slot": self.is_split_slot,
            "source_element_id": self.source_element_id,
        }


def assign_content_to_template(
    demands: list[GeometricDemand],
    template: HierarchicalTemplate,
) -> list[AssignedSlot]:
    """Map user demands to template slots with dynamic region splitting and merging."""
    assigned: list[AssignedSlot] = []

    # 1. Separate Header Demands vs Body Demands
    header_demands: list[GeometricDemand] = []
    body_demands: list[GeometricDemand] = []

    for d in demands:
        if d.content_type in ("title", "author_info", "authors", "logo"):
            header_demands.append(d)
        else:
            body_demands.append(d)

    # --- Step 1: Assign Header Slots ---
    # Header templates usually have 1 Title and 1 Author Info slot
    hdr_title_slots = [s for s in template.header_slots if s.role == "header_title"]
    hdr_author_slots = [s for s in template.header_slots if s.role == "header_authors"]

    default_title_box = [0.08, 0.04, 0.84, 0.08]
    default_auth_box = [0.08, 0.13, 0.84, 0.04]

    title_box = hdr_title_slots[0].bbox_norm if hdr_title_slots else default_title_box
    auth_box = hdr_author_slots[0].bbox_norm if hdr_author_slots else default_auth_box

    reading_idx = 0
    for d in header_demands:
        if d.content_type == "title":
            assigned.append(
                AssignedSlot(
                    block_id=d.block_id,
                    content_type="title",
                    demand=d,
                    target_bbox_norm=list(title_box),
                    column_idx=None,
                    group_id="header",
                    reading_order=reading_idx,
                    source_element_id=hdr_title_slots[0].element_id if hdr_title_slots else None,
                )
            )
            reading_idx += 1
        elif d.content_type in ("author_info", "authors"):
            assigned.append(
                AssignedSlot(
                    block_id=d.block_id,
                    content_type="author_info",
                    demand=d,
                    target_bbox_norm=list(auth_box),
                    column_idx=None,
                    group_id="header",
                    reading_order=reading_idx,
                    source_element_id=hdr_author_slots[0].element_id if hdr_author_slots else None,
                )
            )
            reading_idx += 1
        elif d.content_type == "logo":
            # Place logo at top right or top left
            logo_box = [auth_box[0] + auth_box[2] - 0.08, title_box[1], 0.07, 0.07]
            assigned.append(
                AssignedSlot(
                    block_id=d.block_id,
                    content_type="logo",
                    demand=d,
                    target_bbox_norm=logo_box,
                    column_idx=None,
                    group_id="header",
                    reading_order=reading_idx,
                )
            )
            reading_idx += 1

    # --- Step 2: Organize Body Demands into Logical Content Units ---
    # Group section headings with their paragraphs, and figures with their captions
    # Units: list of lists of demands that should stay together
    content_units: list[list[GeometricDemand]] = []
    curr_unit: list[GeometricDemand] = []

    # Map captions to their target figures if must_be_adjacent_to or naming matches
    fig_demands = {d.block_id: d for d in body_demands if d.content_type in ("figure", "table")}
    paired_captions: set[str] = set()

    # Pre-associate captions with figures
    fig_caption_map: dict[str, GeometricDemand] = {}
    for d in body_demands:
        if d.content_type == "caption":
            # Check explicit adjacency
            if d.must_be_adjacent_to and d.must_be_adjacent_to in fig_demands:
                fig_caption_map[d.must_be_adjacent_to] = d
                paired_captions.add(d.block_id)
            else:
                # Pair with preceding figure if unassigned
                prev_figs = [b for b in body_demands[:body_demands.index(d)] if b.content_type in ("figure", "table") and b.block_id not in fig_caption_map]
                if prev_figs:
                    target_fig = prev_figs[-1]
                    fig_caption_map[target_fig.block_id] = d
                    paired_captions.add(d.block_id)

    # Build sequential content units
    for d in body_demands:
        if d.block_id in paired_captions:
            continue  # Captions are packed directly with their parent figure/table unit

        if d.content_type == "section":
            if curr_unit:
                content_units.append(curr_unit)
                curr_unit = []
            curr_unit.append(d)
        elif d.content_type in ("figure", "table"):
            # Check if current unit has a heading; if so, keep unit, or treat figure as its own sub-unit
            if curr_unit and any(u.content_type == "section" for u in curr_unit) and len(curr_unit) == 1:
                curr_unit.append(d)
            else:
                if curr_unit:
                    content_units.append(curr_unit)
                    curr_unit = []
                curr_unit.append(d)
            if d.block_id in fig_caption_map:
                curr_unit.append(fig_caption_map[d.block_id])
                content_units.append(curr_unit)
                curr_unit = []
        else:
            curr_unit.append(d)

    if curr_unit:
        content_units.append(curr_unit)

    # --- Step 3: Column Allocation & Dynamic Region Splitting ---
    num_cols = max(1, len(template.columns))
    col_bounds = [col.bounds_norm for col in template.columns]

    # Distribute content units across columns in reading order
    # Balance total estimated height per column
    col_heights = [0.0] * num_cols
    col_units: list[list[list[GeometricDemand]]] = [[] for _ in range(num_cols)]

    for unit in content_units:
        unit_h = sum(d.preferred_height for d in unit)
        # Select the column that maintains balanced height while respecting reading order flow
        # Prioritize filling column c until it reaches nominal column capacity, then moving to c+1
        col_cap = col_bounds[0][3] if col_bounds else 0.70
        assigned_c = 0
        for c in range(num_cols):
            if col_heights[c] + unit_h * 0.5 <= col_cap or c == num_cols - 1:
                assigned_c = c
                break
        col_units[assigned_c].append(unit)
        col_heights[assigned_c] += unit_h

    # --- Step 4: Slot Positioning and Splitting within Each Column ---
    for c_i, units_in_col in enumerate(col_units):
        c_bounds = col_bounds[c_i]
        c_x0 = c_bounds[0]
        c_w = c_bounds[2]
        c_y_start = max(c_bounds[1], default_auth_box[1] + default_auth_box[3] + 0.03)
        curr_y = c_y_start

        # Check existing template slots in this column for initial sizing reference
        col_template_slots = template.columns[c_i].all_slots if c_i < len(template.columns) else []

        slot_idx_in_col = 0
        for u_idx, unit in enumerate(units_in_col):
            sec_group_id = f"col_{c_i}_sec_{u_idx}"

            # Process demands in this unit sequentially
            d_idx = 0
            while d_idx < len(unit):
                d = unit[d_idx]
                target_w = c_w

                # Check if this demand is a Figure with an attached Caption
                has_caption = (
                    d_idx + 1 < len(unit) and unit[d_idx + 1].content_type == "caption"
                )

                # Reference template slot if available
                ref_slot = col_template_slots[slot_idx_in_col] if slot_idx_in_col < len(col_template_slots) else None
                slot_idx_in_col += 1

                # Height calculation based on content demand
                target_h = d.preferred_height
                if d.content_type == "section":
                    target_h = min(0.045, d.preferred_height)
                elif d.content_type in ("figure", "table"):
                    # Scale height to column width
                    if d.aspect_ratio and d.lock_aspect_ratio:
                        target_h = float(np.clip(target_w * (template.aspect_ratio / d.aspect_ratio), 0.06, 0.45))
                    else:
                        target_h = float(np.clip(d.preferred_height, 0.08, 0.40))

                slot_bbox = [c_x0, curr_y, target_w, target_h]

                assigned.append(
                    AssignedSlot(
                        block_id=d.block_id,
                        content_type=d.content_type,
                        demand=d,
                        target_bbox_norm=slot_bbox,
                        column_idx=c_i,
                        group_id=sec_group_id,
                        reading_order=reading_idx,
                        is_split_slot=slot_idx_in_col > len(col_template_slots),
                        source_element_id=ref_slot.element_id if ref_slot else None,
                    )
                )
                reading_idx += 1
                curr_y += target_h + 0.012  # Standard intra-section gap

                # If followed by caption, place caption immediately below figure
                if has_caption:
                    cap_demand = unit[d_idx + 1]
                    cap_h = min(0.04, cap_demand.preferred_height)
                    cap_bbox = [c_x0, curr_y, target_w, cap_h]
                    assigned.append(
                        AssignedSlot(
                            block_id=cap_demand.block_id,
                            content_type="caption",
                            demand=cap_demand,
                            target_bbox_norm=cap_bbox,
                            column_idx=c_i,
                            group_id=sec_group_id,
                            reading_order=reading_idx,
                            paired_with_block_id=d.block_id,
                            is_split_slot=True,
                        )
                    )
                    reading_idx += 1
                    curr_y += cap_h + 0.016  # Extra gap after figure+caption
                    d_idx += 2
                else:
                    d_idx += 1

            curr_y += 0.015  # Gap between logical sections

    return assigned
