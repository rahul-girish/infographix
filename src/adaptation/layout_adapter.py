"""End-to-End Layout Adaptation Engine for Infographix (Stage 3).

Coordinates the complete adaptation pipeline:
  1. Computes typography-aware content demands from a `PosterRequest`.
  2. Extracts hierarchical templates for top-k retrieved reference posters.
  3. Performs semantic slot assignment and dynamic region splitting/merging.
  4. Solves constraint-based layout optimization (LP/MILP via HiGHS).
  5. Evaluates adaptation feasibility and quality across all candidates.
  6. Selects the optimal candidate and converts it to a structured, editable `PosterDocument`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.adaptation.adaptation_evaluation import evaluate_adapted_layout
from src.adaptation.content_geometry import (
    TYPOGRAPHY_PRESETS,
    compute_content_demands,
)
from src.adaptation.layout_constraints import solve_layout_optimization
from src.adaptation.slot_assignment import assign_content_to_template
from src.adaptation.template_regions import (
    HierarchicalTemplate,
    extract_hierarchical_template,
)
from src.features.request_schema import PosterRequest
from src.schemas.layout_document import (
    ElementStyle,
    PosterCanvas,
    PosterDocument,
    PosterElement,
)


@dataclass
class CandidateAdaptationResult:
    """The complete adaptation result for a single candidate reference template."""

    candidate_rank: int
    template_poster_id: int
    hierarchical_template: HierarchicalTemplate
    optimization_result: Any
    evaluation_metrics: dict[str, Any]
    poster_document: PosterDocument
    is_feasible: bool

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "candidate_rank": self.candidate_rank,
            "template_poster_id": self.template_poster_id,
            "is_feasible": self.is_feasible,
            "composite_quality_score": self.evaluation_metrics.get("composite_quality_score", 0.0),
            "overflow_rate": self.evaluation_metrics.get("overflow_rate", 0.0),
            "overlap_area": self.evaluation_metrics.get("overlap_area", 0.0),
            "min_size_violation_rate": self.evaluation_metrics.get("min_size_violation_rate", 0.0),
            "aspect_distortion": self.evaluation_metrics.get("mean_aspect_distortion", 0.0),
            "whitespace_utilization": self.evaluation_metrics.get("whitespace_utilization", 0.0),
        }


class LayoutAdapter:
    """High-level layout adaptation engine."""

    def __init__(
        self,
        elements_df: pd.DataFrame,
        posters_df: pd.DataFrame | None = None,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
    ):
        self.elements_df = elements_df
        self.posters_df = (
            posters_df.set_index("poster_id") if posters_df is not None else None
        )
        self.default_width = canvas_width
        self.default_height = canvas_height

    def adapt_single_candidate(
        self,
        request: PosterRequest,
        candidate_poster_row: pd.Series | dict[str, Any],
        candidate_rank: int = 1,
    ) -> CandidateAdaptationResult:
        """Run full adaptation on a single candidate reference poster."""
        c_pid = int(candidate_poster_row["poster_id"])
        c_asp = float(candidate_poster_row.get("aspect_ratio", request.requested_aspect_ratio))

        # 1. Extract Hierarchical Intermediate Representation
        template = extract_hierarchical_template(
            poster_row=candidate_poster_row,
            elements_df=self.elements_df,
        )

        # 2. Compute Content Geometry Demands
        # Nominal column width from template
        col_w_norm = (
            template.columns[0].bounds_norm[2] if template.columns else 0.28
        )
        demands = compute_content_demands(
            request=request,
            candidate_col_width_norm=col_w_norm,
            canvas_aspect_ratio=c_asp,
        )

        # 3. Semantic Slot Assignment & Region Splitting / Merging
        initial_assigned = assign_content_to_template(
            demands=demands,
            template=template,
        )

        # 4. Constraint-Based Optimization (LP/MILP via HiGHS)
        opt_res = solve_layout_optimization(
            assigned_slots=initial_assigned,
            canvas_aspect_ratio=c_asp,
        )

        # 5. Evaluate Adaptation Feasibility and Quality
        metrics = evaluate_adapted_layout(
            demands=demands,
            adapted_slots=opt_res.adapted_slots,
            canvas_aspect=c_asp,
        )

        # 6. Build Editable PosterDocument
        canvas_w = int(self.default_width)
        canvas_h = int(round(canvas_w / c_asp))

        canvas_obj = PosterCanvas(
            width=canvas_w,
            height=canvas_h,
            unit="px",
            background_color="#ffffff",
        )

        # Build demand lookup for actual content strings and titles
        req_blocks_lookup = {b.block_id: b for b in request.blocks}

        poster_elements: list[PosterElement] = []
        for s in opt_res.adapted_slots:
            bx_norm = s.target_bbox_norm
            # Convert normalized coordinates [0, 1] to canvas pixel coordinates
            px_x = float(bx_norm[0] * canvas_w)
            px_y = float(bx_norm[1] * canvas_h)
            px_w = float(bx_norm[2] * canvas_w)
            px_h = float(bx_norm[3] * canvas_h)

            # Retrieve actual content
            blk_def = req_blocks_lookup.get(s.block_id)
            actual_content = getattr(blk_def, "content", None) or s.demand.raw_content
            if actual_content is None:
                # Default readable placeholder based on type
                if s.content_type == "title":
                    actual_content = "Scientific Poster Title"
                elif s.content_type in ("author_info", "authors"):
                    actual_content = "Author Name, Co-Author Name — Research Institute"
                elif s.content_type == "section":
                    actual_content = blk_def.semantic_role.replace("_", " ").title() if blk_def else "Section Heading"
                else:
                    actual_content = f"[{s.content_type.title()} Content Block]"

            # Set styling from preset
            preset = TYPOGRAPHY_PRESETS.get(s.content_type, TYPOGRAPHY_PRESETS["text"])
            style = ElementStyle(
                font_family="Inter, system-ui, sans-serif",
                font_size_pt=float(s.demand.font_size_pt),
                font_weight=preset.get("font_weight", "normal"),
                line_height=float(preset.get("line_height_mult", 1.35)),
                text_color="#0f172a" if s.content_type != "caption" else "#475569",
                background_color=(
                    "#f1f5f9"
                    if s.content_type in ("figure", "table")
                    else "transparent"
                ),
                border_color="#cbd5e1" if s.content_type in ("figure", "table") else "transparent",
                border_width_px=1 if s.content_type in ("figure", "table") else 0,
            )

            p_el = PosterElement(
                id=s.block_id,
                type=s.content_type,
                bbox=[px_x, px_y, px_w, px_h],
                bbox_norm=bx_norm,
                content=actual_content,
                style=style,
                group_id=s.group_id,
                column_idx=s.column_idx,
                source_block_id=s.block_id,
                associated_element_ids=[s.paired_with_block_id] if s.paired_with_block_id else [],
                metadata={"priority": s.demand.priority, "is_split_slot": s.is_split_slot},
            )
            poster_elements.append(p_el)

        doc = PosterDocument(
            poster_id=f"adapted_{request.request_id}_{c_pid}",
            source_template_id=c_pid,
            canvas=canvas_obj,
            elements=poster_elements,
            title=(
                str(req_blocks_lookup.get("b_title", {}).content)
                if hasattr(req_blocks_lookup.get("b_title", None), "content")
                else "Scientific Poster"
            ),
            authors=getattr(request, "authors", ["Research Authors"]),
            metadata={
                "template_aspect_ratio": c_asp,
                "composite_quality_score": metrics["composite_quality_score"],
                "candidate_rank": candidate_rank,
            },
        )

        return CandidateAdaptationResult(
            candidate_rank=candidate_rank,
            template_poster_id=c_pid,
            hierarchical_template=template,
            optimization_result=opt_res,
            evaluation_metrics=metrics,
            poster_document=doc,
            is_feasible=opt_res.is_feasible and metrics["overflow_rate"] < 0.05,
        )

    def adapt_and_select_best(
        self,
        request: PosterRequest,
        candidate_posters_df: pd.DataFrame,
    ) -> tuple[CandidateAdaptationResult, list[CandidateAdaptationResult]]:
        """Adapt top candidate reference posters and automatically select the best."""
        results: list[CandidateAdaptationResult] = []

        for idx, (_, row) in enumerate(candidate_posters_df.iterrows()):
            res = self.adapt_single_candidate(
                request=request,
                candidate_poster_row=row,
                candidate_rank=idx + 1,
            )
            results.append(res)

        # Selection formula:
        # Feasible candidates ranked by composite_quality_score, with secondary preference for higher rank
        feasible_results = [r for r in results if r.is_feasible]
        if feasible_results:
            best_res = max(
                feasible_results,
                key=lambda r: (
                    r.evaluation_metrics["composite_quality_score"]
                    - 0.5 * (r.candidate_rank - 1)
                ),
            )
        else:
            # Fallback to least-overflow candidate
            best_res = min(
                results,
                key=lambda r: (
                    r.evaluation_metrics["overflow_rate"]
                    + r.evaluation_metrics["total_overflow_depth"]
                ),
            )

        return best_res, results
