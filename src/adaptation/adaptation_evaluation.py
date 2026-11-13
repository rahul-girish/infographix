"""Comprehensive Evaluation Metrics for Layout Adaptation (Stage 3).

Measures layout adaptation quality across 9 objective dimensions:
  1. Placement success (fraction of requested content elements successfully positioned)
  2. Overflow rate (fraction of elements extending beyond canvas bounds)
  3. Overlap area (total unintended pairwise intersection area)
  4. Minimum-size violations (fraction of elements shrunk below readable minimum)
  5. Aspect-ratio distortion (mean error from target image aspect ratio)
  6. Relationship preservation (figure-caption proximity & heading-section coupling)
  7. Whitespace distribution (effective usage of canvas without excessive crowding)
  8. Template deviation (mean L1 shift from original template structure)
  9. Content preservation (100% check that zero user blocks were dropped)

Supports benchmarking across three baseline paradigms:
  - (1) Unmodified Retrieved Template
  - (2) Simple Greedy Sequential Placement
  - (3) Constraint-Based HiGHS Optimization
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.adaptation.content_geometry import GeometricDemand
from src.adaptation.layout_constraints import OptimizationResult
from src.adaptation.slot_assignment import AssignedSlot
from src.adaptation.template_regions import HierarchicalTemplate


def compute_pairwise_overlap_area(bboxes: list[list[float]]) -> float:
    """Compute total pairwise intersection area between bounding boxes [x0, y0, w, h]."""
    total_overlap = 0.0
    N = len(bboxes)
    for i in range(N):
        x1_i = bboxes[i][0] + bboxes[i][2]
        y1_i = bboxes[i][1] + bboxes[i][3]
        for j in range(i + 1, N):
            x1_j = bboxes[j][0] + bboxes[j][2]
            y1_j = bboxes[j][1] + bboxes[j][3]

            inter_w = max(0.0, min(x1_i, x1_j) - max(bboxes[i][0], bboxes[j][0]))
            inter_h = max(0.0, min(y1_i, y1_j) - max(bboxes[i][1], bboxes[j][1]))
            inter_area = inter_w * inter_h
            if inter_area > 1e-5:
                total_overlap += inter_area
    return float(total_overlap)


def evaluate_adapted_layout(
    demands: list[GeometricDemand],
    adapted_slots: list[AssignedSlot],
    canvas_aspect: float = 1.414,
    y_max_bound: float = 0.98,
) -> dict[str, Any]:
    """Calculate the 9 comprehensive adaptation quality metrics for a placed layout."""
    n_demanded = len(demands)
    n_placed = len(adapted_slots)

    # 1. Placement success & Content preservation
    placement_success = float(n_placed / max(1, n_demanded))
    demanded_ids = {d.block_id for d in demands}
    placed_ids = {s.block_id for s in adapted_slots}
    missing_ids = list(demanded_ids - placed_ids)
    content_preservation = float(len(demanded_ids & placed_ids) / max(1, len(demanded_ids)))

    # 2. Overflow rate & Canvas Bounds
    overflow_count = 0
    total_overflow_depth = 0.0
    for s in adapted_slots:
        b = s.target_bbox_norm
        y1 = b[1] + b[3]
        x1 = b[0] + b[2]
        if y1 > y_max_bound or x1 > 0.99 or b[0] < 0.01 or b[1] < 0.01:
            overflow_count += 1
            if y1 > y_max_bound:
                total_overflow_depth += (y1 - y_max_bound)

    overflow_rate = float(overflow_count / max(1, n_placed))

    # 3. Overlap area
    bboxes = [s.target_bbox_norm for s in adapted_slots]
    overlap_area = compute_pairwise_overlap_area(bboxes)

    # 4. Minimum-size violations
    min_size_violations = 0
    for s in adapted_slots:
        b = s.target_bbox_norm
        d = s.demand
        if b[3] < d.min_height * 0.95 or b[2] < d.min_width * 0.90:
            min_size_violations += 1
    min_size_violation_rate = float(min_size_violations / max(1, n_placed))

    # 5. Aspect-ratio distortion (for images/figures with locked aspect ratio)
    aspect_distortions: list[float] = []
    for s in adapted_slots:
        d = s.demand
        if d.content_type in ("figure", "image") and d.lock_aspect_ratio and d.aspect_ratio:
            b = s.target_bbox_norm
            # Actual normalized aspect ratio: (w * canvas_aspect) / h
            actual_asp = float((b[2] * canvas_aspect) / max(1e-4, b[3]))
            target_asp = float(d.aspect_ratio)
            err = abs(actual_asp - target_asp) / target_asp
            aspect_distortions.append(err)
    mean_aspect_distortion = float(np.mean(aspect_distortions)) if aspect_distortions else 0.0

    # 6. Relationship preservation (Figure-Caption coupling distance)
    slot_lookup = {s.block_id: s for s in adapted_slots}
    coupling_errors: list[float] = []
    for s in adapted_slots:
        if s.paired_with_block_id and s.paired_with_block_id in slot_lookup:
            fig_s = slot_lookup[s.paired_with_block_id]
            fig_b = fig_s.target_bbox_norm
            cap_b = s.target_bbox_norm
            # Caption should sit immediately below figure (dy >= 0, small gap <= 0.02)
            gap = cap_b[1] - (fig_b[1] + fig_b[3])
            err = max(0.0, abs(gap - 0.008))
            coupling_errors.append(err)
    rel_preservation_score = float(np.exp(-25.0 * np.mean(coupling_errors))) if coupling_errors else 1.0

    # 7. Whitespace distribution
    total_content_area = sum(b[2] * b[3] for b in bboxes)
    # Ideal occupied canvas area is between 0.45 and 0.75
    whitespace_utilization = float(np.clip(total_content_area / 0.65, 0.0, 1.0))

    # 8. Template deviation (L1 shift from allocated target box)
    shifts = [
        sum(abs(s.target_bbox_norm[i] - s.demand.preferred_height if i == 3 else 0.0) for i in range(4))
        for s in adapted_slots
    ]
    template_deviation = float(np.mean(shifts)) if shifts else 0.0

    # Composite Quality Score (0 to 100)
    score = (
        30.0 * content_preservation
        + 25.0 * (1.0 - min(1.0, overflow_rate + total_overflow_depth * 2.0))
        + 15.0 * (1.0 - min(1.0, overlap_area * 50.0))
        + 10.0 * (1.0 - min_size_violation_rate)
        + 10.0 * (1.0 - min(1.0, mean_aspect_distortion))
        + 10.0 * rel_preservation_score
    )

    return {
        "composite_quality_score": round(float(np.clip(score, 0.0, 100.0)), 1),
        "placement_success_rate": round(placement_success, 4),
        "content_preservation": round(content_preservation, 4),
        "overflow_rate": round(overflow_rate, 4),
        "total_overflow_depth": round(total_overflow_depth, 4),
        "overlap_area": round(overlap_area, 5),
        "min_size_violation_rate": round(min_size_violation_rate, 4),
        "mean_aspect_distortion": round(mean_aspect_distortion, 4),
        "relationship_preservation_score": round(rel_preservation_score, 4),
        "whitespace_utilization": round(whitespace_utilization, 4),
        "template_deviation": round(template_deviation, 4),
        "unplaced_count": len(missing_ids),
        "unplaced_block_ids": missing_ids,
    }


def run_greedy_baseline_placement(
    demands: list[GeometricDemand],
    template: HierarchicalTemplate,
) -> list[AssignedSlot]:
    """Greedy sequential baseline: fills columns top-to-bottom without constraint optimization."""
    assigned: list[AssignedSlot] = []
    num_cols = max(1, len(template.columns))
    col_bounds = [col.bounds_norm for col in template.columns]

    curr_col = 0
    curr_y = 0.20
    reading_idx = 0

    for d in demands:
        if d.content_type in ("title", "author_info", "logo"):
            assigned.append(
                AssignedSlot(
                    block_id=d.block_id,
                    content_type=d.content_type,
                    demand=d,
                    target_bbox_norm=[0.08, 0.04 if d.content_type == "title" else 0.12, 0.84, d.preferred_height],
                    column_idx=None,
                    group_id="header",
                    reading_order=reading_idx,
                )
            )
            reading_idx += 1
            continue

        c_w = col_bounds[curr_col][2]
        c_x0 = col_bounds[curr_col][0]
        h = d.preferred_height

        # If element overflows column bottom, greedily advance to next column
        if curr_y + h > 0.95 and curr_col < num_cols - 1:
            curr_col += 1
            curr_y = 0.20
            c_w = col_bounds[curr_col][2]
            c_x0 = col_bounds[curr_col][0]

        assigned.append(
            AssignedSlot(
                block_id=d.block_id,
                content_type=d.content_type,
                demand=d,
                target_bbox_norm=[c_x0, curr_y, c_w, h],
                column_idx=curr_col,
                group_id=f"col_{curr_col}",
                reading_order=reading_idx,
            )
        )
        reading_idx += 1
        curr_y += h + 0.015

    return assigned
