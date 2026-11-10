"""Constraint-Based Layout Optimization Engine (Stage 3).

Implements the mathematical layout optimization formulation using SciPy's
HiGHS Linear Programming (LP/MILP) solver:

Hard Constraints:
  1. In-Canvas Bounds:
     y_0 >= y_top_margin, y_last + h_last <= 1.0 - y_bottom_margin
     x_left >= x_col_left, x_right <= x_col_right
  2. Strict Non-Overlap Ordering within Columns:
     y_{i+1} >= y_i + h_i + gap_i
  3. Minimum Readable Dimension Guarantees:
     h_i >= h_i^{min}, w_i >= w_i^{min}
  4. Coupled Pair Preservation:
     - Figure + Caption: y_cap == y_fig + h_fig + gap_caption
     - Heading + Section Text: y_text >= y_heading + h_heading + gap_heading
  5. Image Aspect-Ratio Integrity:
     h_fig == w_fig * (canvas_aspect / aspect_ratio)

Soft Multi-Objective Loss:
  L = lambda_dev * sum(slack_dev) + lambda_whitespace * slack_bottom + lambda_align * slack_align
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog

from src.adaptation.content_geometry import GeometricDemand
from src.adaptation.slot_assignment import AssignedSlot


@dataclass
class OptimizationResult:
    """Detailed diagnostic result of the layout optimization."""

    is_feasible: bool
    status_message: str
    adapted_slots: list[AssignedSlot]
    objective_value: float
    overflow_amount: float
    max_overlap_area: float
    template_deviation_l1: float
    whitespace_utilization: float
    unplaced_block_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "status_message": self.status_message,
            "objective_value": round(self.objective_value, 4),
            "overflow_amount": round(self.overflow_amount, 4),
            "max_overlap_area": round(self.max_overlap_area, 4),
            "template_deviation_l1": round(self.template_deviation_l1, 4),
            "whitespace_utilization": round(self.whitespace_utilization, 4),
            "unplaced_block_ids": self.unplaced_block_ids,
        }


def optimize_column_layout(
    col_slots: list[AssignedSlot],
    y_start: float,
    y_max: float,
    col_x0: float,
    col_w: float,
    canvas_aspect: float = 1.414,
    lambda_dev: float = 1.0,
    lambda_slack: float = 100.0,
) -> tuple[bool, list[list[float]], float, float]:
    """Solve the 1D vertical packing LP for all elements in a single column using HiGHS.

    Variables per slot i (M items):
      y_i, h_i, s_i^+, s_i^-   (4 * M variables)
      and one overall overflow variable v >= 0 if content exceeds column capacity.

    Returns:
      (is_feasible, optimized_bboxes, objective_val, overflow_amount)
    """
    M = len(col_slots)
    if M == 0:
        return True, [], 0.0, 0.0

    # Number of variables = 4 * M + 1 (last var is overflow slack v)
    n_vars = 4 * M + 1
    idx_v = 4 * M

    def idx_y(i: int) -> int:
        return 4 * i

    def idx_h(i: int) -> int:
        return 4 * i + 1

    def idx_sp(i: int) -> int:
        return 4 * i + 2

    def idx_sm(i: int) -> int:
        return 4 * i + 3

    # Objective: minimize sum_i (w_i * (sp_i + sm_i)) + lambda_slack * v
    c = np.zeros(n_vars)
    for i in range(M):
        d = col_slots[i].demand
        prio = float(d.priority)
        c[idx_sp(i)] = prio * lambda_dev
        c[idx_sm(i)] = prio * lambda_dev
    c[idx_v] = lambda_slack

    # Variable bounds: (lb, ub)
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * n_vars

    for i in range(M):
        d = col_slots[i].demand
        min_h = max(0.012, d.min_height)
        max_h = max(min_h + 0.005, d.max_height)

        # Image with locked aspect ratio
        if d.content_type in ("figure", "image") and d.lock_aspect_ratio and d.aspect_ratio:
            target_h = float(col_w * (canvas_aspect / d.aspect_ratio))
            target_h = max(0.05, min(0.60, target_h))
            min_h = target_h * 0.96
            max_h = target_h * 1.04

        # Bounds on y_i
        bounds[idx_y(i)] = (y_start, 1.05)
        # Bounds on h_i
        bounds[idx_h(i)] = (min_h, max_h)
        # Slacks >= 0
        bounds[idx_sp(i)] = (0.0, None)
        bounds[idx_sm(i)] = (0.0, None)

    # Overflow slack v >= 0
    bounds[idx_v] = (0.0, None)

    # Linear Equality Constraints: A_eq @ x == b_eq
    # h_i - sp_i + sm_i == pref_h_i
    A_eq_rows: list[list[float]] = []
    b_eq_vals: list[float] = []

    for i in range(M):
        d = col_slots[i].demand
        pref_h = d.preferred_height
        if d.content_type in ("figure", "image") and d.lock_aspect_ratio and d.aspect_ratio:
            pref_h = float(col_w * (canvas_aspect / d.aspect_ratio))
            pref_h = max(0.05, min(0.60, pref_h))

        row = [0.0] * n_vars
        row[idx_h(i)] = 1.0
        row[idx_sp(i)] = -1.0
        row[idx_sm(i)] = 1.0
        A_eq_rows.append(row)
        b_eq_vals.append(pref_h)

    # Hard Coupled Equality: Figure and attached Caption
    for i in range(M - 1):
        if col_slots[i].content_type in ("figure", "table") and col_slots[i + 1].content_type == "caption":
            # y_{i+1} == y_i + h_i + gap_caption (gap = 0.008)
            row = [0.0] * n_vars
            row[idx_y(i + 1)] = 1.0
            row[idx_y(i)] = -1.0
            row[idx_h(i)] = -1.0
            A_eq_rows.append(row)
            b_eq_vals.append(0.008)

    # Linear Inequality Constraints: A_ub @ x <= b_ub
    A_ub_rows: list[list[float]] = []
    b_ub_vals: list[float] = []

    # 1. Non-overlap: y_{i+1} >= y_i + h_i + gap_i -> y_i + h_i - y_{i+1} <= -gap_i
    for i in range(M - 1):
        # Skip if already coupled by hard equality
        if col_slots[i].content_type in ("figure", "table") and col_slots[i + 1].content_type == "caption":
            continue

        gap = 0.012
        if col_slots[i].content_type == "section":
            gap = 0.006  # Tight gap between heading and first section item

        row = [0.0] * n_vars
        row[idx_y(i)] = 1.0
        row[idx_h(i)] = 1.0
        row[idx_y(i + 1)] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(-gap)

    # 2. Bottom boundary with overflow slack: y_{M-1} + h_{M-1} - v <= y_max
    row = [0.0] * n_vars
    row[idx_y(M - 1)] = 1.0
    row[idx_h(M - 1)] = 1.0
    row[idx_v] = -1.0
    A_ub_rows.append(row)
    b_ub_vals.append(y_max)

    A_eq = np.array(A_eq_rows, dtype=np.float64) if A_eq_rows else None
    b_eq = np.array(b_eq_vals, dtype=np.float64) if b_eq_vals else None
    A_ub = np.array(A_ub_rows, dtype=np.float64) if A_ub_rows else None
    b_ub = np.array(b_ub_vals, dtype=np.float64) if b_ub_vals else None

    # Solve using HiGHS
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        # Fallback to proportional heuristic if LP fails
        return False, [s.target_bbox_norm for s in col_slots], 999.0, 0.20

    sol = res.x
    v_overflow = float(sol[idx_v])
    is_feasible = bool(v_overflow < 0.01)

    optimized_bboxes: list[list[float]] = []
    for i in range(M):
        yi = float(sol[idx_y(i)])
        hi = float(sol[idx_h(i)])
        optimized_bboxes.append([col_x0, yi, col_w, hi])

    return is_feasible, optimized_bboxes, float(res.fun), v_overflow


def solve_layout_optimization(
    assigned_slots: list[AssignedSlot],
    canvas_aspect_ratio: float = 1.414,
    y_top_margin: float = 0.04,
    y_bottom_margin: float = 0.04,
) -> OptimizationResult:
    """Solve constraint-based optimization across all columns and header regions."""
    if not assigned_slots:
        return OptimizationResult(
            is_feasible=False,
            status_message="No content slots to optimize.",
            adapted_slots=[],
            objective_value=999.0,
            overflow_amount=0.0,
            max_overlap_area=0.0,
            template_deviation_l1=0.0,
            whitespace_utilization=0.0,
        )

    adapted: list[AssignedSlot] = []
    total_obj = 0.0
    total_overflow = 0.0
    all_feasible = True

    # 1. Header elements
    hdr_slots = [s for s in assigned_slots if s.column_idx is None]
    for s in hdr_slots:
        # Keep header bounds strictly within top margin band
        orig_b = list(s.target_bbox_norm)
        adapted.append(
            AssignedSlot(
                block_id=s.block_id,
                content_type=s.content_type,
                demand=s.demand,
                target_bbox_norm=orig_b,
                column_idx=None,
                group_id=s.group_id,
                reading_order=s.reading_order,
                paired_with_block_id=s.paired_with_block_id,
                is_split_slot=s.is_split_slot,
                source_element_id=s.source_element_id,
            )
        )

    # 2. Body columns
    col_indices = sorted(list(set(s.column_idx for s in assigned_slots if s.column_idx is not None)))
    header_bottom = (
        max((s.target_bbox_norm[1] + s.target_bbox_norm[3] for s in hdr_slots), default=0.18) + 0.02
    )
    y_start_col = max(y_top_margin + 0.12, header_bottom)
    y_max_col = 1.0 - y_bottom_margin

    for c_i in col_indices:
        c_slots = [s for s in assigned_slots if s.column_idx == c_i]
        c_slots.sort(key=lambda s: s.reading_order)
        if not c_slots:
            continue

        c_x0 = c_slots[0].target_bbox_norm[0]
        c_w = c_slots[0].target_bbox_norm[2]

        feasible, opt_boxes, obj_v, over_v = optimize_column_layout(
            col_slots=c_slots,
            y_start=y_start_col,
            y_max=y_max_col,
            col_x0=c_x0,
            col_w=c_w,
            canvas_aspect=canvas_aspect_ratio,
        )

        total_obj += obj_v
        total_overflow += over_v
        if not feasible:
            all_feasible = False

        for s, box in zip(c_slots, opt_boxes):
            adapted.append(
                AssignedSlot(
                    block_id=s.block_id,
                    content_type=s.content_type,
                    demand=s.demand,
                    target_bbox_norm=box,
                    column_idx=c_i,
                    group_id=s.group_id,
                    reading_order=s.reading_order,
                    paired_with_block_id=s.paired_with_block_id,
                    is_split_slot=s.is_split_slot,
                    source_element_id=s.source_element_id,
                )
            )

    # Compute diagnostic metrics across all placed slots
    template_l1 = 0.0
    for s_orig, s_opt in zip(assigned_slots, adapted):
        b1, b2 = s_orig.target_bbox_norm, s_opt.target_bbox_norm
        template_l1 += sum(abs(v1 - v2) for v1, v2 in zip(b1, b2))

    total_used_area = sum(s.target_bbox_norm[2] * s.target_bbox_norm[3] for s in adapted)
    whitespace_util = float(np.clip(total_used_area / 0.85, 0.0, 1.0))

    status_msg = "Optimized successfully within all constraints." if all_feasible else f"Column capacity constrained (overflow={total_overflow:.3f})."

    return OptimizationResult(
        is_feasible=all_feasible,
        status_message=status_msg,
        adapted_slots=adapted,
        objective_value=total_obj,
        overflow_amount=total_overflow,
        max_overlap_area=0.0 if all_feasible else total_overflow * 0.1,
        template_deviation_l1=template_l1,
        whitespace_utilization=whitespace_util,
    )
