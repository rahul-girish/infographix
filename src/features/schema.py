"""Feature schema metadata generation for v2 feature tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

NON_FEATURE_POSTER_COLS: set[str] = {
    "poster_id",
    "split",
    "source_split",
    "canvas_width_px",
    "canvas_height_px",
    "orientation",
}


def generate_feature_schema_metadata(
    posters_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    schema_version: str = "2.0.0",
) -> dict[str, Any]:
    """Generate comprehensive versioned feature schema documentation."""
    poster_numeric_cols = [
        c for c in posters_df.columns if c not in NON_FEATURE_POSTER_COLS
    ]

    feature_groups = {
        "canvas_geometry": [
            "aspect_ratio",
            "log_aspect_ratio",
            "is_landscape",
            "is_portrait",
            "total_element_count",
            "known_element_count",
        ],
        "category_inventory": [
            c
            for c in poster_numeric_cols
            if c.startswith(("count_", "log1p_count_", "has_", "fraction_"))
            or c
            in {
                "visual_element_count",
                "textual_element_count",
                "heading_element_count",
                "media_to_text_count_ratio",
            }
        ],
        "size_and_capacity": [
            c
            for c in poster_numeric_cols
            if any(
                k in c
                for k in (
                    "_p10",
                    "_p25",
                    "_p50",
                    "_p75",
                    "_p90",
                    "_iqr",
                    "area_sum_",
                    "visual_area_sum",
                    "textual_area_sum",
                    "header_area_sum",
                    "visual_to_body_area_ratio",
                )
            )
        ],
        "coverage_and_margins": [
            "sum_box_area_fraction",
            "union_box_coverage",
            "overlap_excess",
            "unannotated_whitespace_fraction",
            "overlap_pair_count",
            "overlap_pair_fraction",
            "max_pair_iou",
            "mean_intersection_over_smaller_box",
            "content_left_margin",
            "content_right_margin",
            "content_top_margin",
            "content_bottom_margin",
            "content_extent_width",
            "content_extent_height",
            "content_extent_area",
            "header_band_bottom_y",
        ],
        "spatial_occupancy_and_profiles": [
            c
            for c in poster_numeric_cols
            if c.startswith(("occ_grid_", "horiz_occ_", "vert_occ_", "text_occ_", "visual_occ_"))
        ],
        "structure_columns_and_alignment": [
            "estimated_column_count",
            "column_assignment_confidence",
            "estimated_gutter_fraction",
            "estimated_row_band_count",
            "full_width_region_count",
            "multi_column_span_count",
            "frac_aligned_left",
            "frac_aligned_right",
            "frac_aligned_top",
            "frac_aligned_bottom",
            "frac_aligned_cx",
            "frac_aligned_cy",
            "left_right_area_balance",
            "top_bottom_area_balance",
            "horiz_profile_symmetry",
            "vert_profile_symmetry",
            "center_of_mass_x",
            "center_of_mass_y",
            "mean_nn_distance",
            "std_nn_distance",
            "fig_caption_paired_fraction",
        ],
    }

    return {
        "schema_version": schema_version,
        "dataset": "SciPostLayout v2",
        "tables": {
            "posters": {
                "row_count": len(posters_df),
                "total_columns": len(posters_df.columns),
                "numeric_feature_columns": poster_numeric_cols,
                "feature_groups": feature_groups,
            },
            "elements": {
                "row_count": len(elements_df),
                "columns": list(elements_df.columns),
            },
            "relations": {
                "row_count": len(relations_df),
                "columns": list(relations_df.columns),
            },
        },
    }
