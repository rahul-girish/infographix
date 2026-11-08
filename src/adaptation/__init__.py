"""Adaptation package for Infographix (Stage 3).

Provides:
  - Template hierarchy extraction (columns, sections, paired slots)
  - Dynamic content geometry estimation
  - Semantic slot assignment and region splitting/merging
  - SciPy HiGHS-based constraint optimization
  - Objective layout quality evaluation
  - End-to-end multi-candidate layout adaptation
"""

from src.adaptation.adaptation_evaluation import (
    compute_pairwise_overlap_area,
    evaluate_adapted_layout,
    run_greedy_baseline_placement,
)
from src.adaptation.content_geometry import (
    TYPOGRAPHY_PRESETS,
    GeometricDemand,
    compute_content_demands,
    estimate_image_content_geometry,
    estimate_table_content_geometry,
    estimate_text_content_geometry,
)
from src.adaptation.layout_adapter import (
    CandidateAdaptationResult,
    LayoutAdapter,
)
from src.adaptation.layout_constraints import (
    OptimizationResult,
    optimize_column_layout,
    solve_layout_optimization,
)
from src.adaptation.slot_assignment import (
    AssignedSlot,
    assign_content_to_template,
)
from src.adaptation.template_regions import (
    HierarchicalTemplate,
    TemplateColumn,
    TemplateSection,
    TemplateSlot,
    extract_hierarchical_template,
)

__all__ = [
    "HierarchicalTemplate",
    "TemplateColumn",
    "TemplateSection",
    "TemplateSlot",
    "extract_hierarchical_template",
    "GeometricDemand",
    "TYPOGRAPHY_PRESETS",
    "compute_content_demands",
    "estimate_image_content_geometry",
    "estimate_table_content_geometry",
    "estimate_text_content_geometry",
    "AssignedSlot",
    "assign_content_to_template",
    "OptimizationResult",
    "optimize_column_layout",
    "solve_layout_optimization",
    "compute_pairwise_overlap_area",
    "evaluate_adapted_layout",
    "run_greedy_baseline_placement",
    "CandidateAdaptationResult",
    "LayoutAdapter",
]
