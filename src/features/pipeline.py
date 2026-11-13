"""Orchestrator for the modular v2 multi-table feature engineering pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.config import (
    DEFAULT_SPLIT_RATIOS,
    PipelineConfig,
    SPLITS,
    load_pipeline_config,
)
from src.features.exporters import (
    export_legacy_v1_compatible_parquets,
    export_v2_tables_and_metadata,
)
from src.features.posters import build_posters_table
from src.features.relations import build_relations_and_spatial_summaries
from src.features.schema import generate_feature_schema_metadata
from src.features.validation import validate_and_extract_elements


def build_pipeline_report(
    posters_df: pd.DataFrame,
    quality_summary: dict[str, Any],
    elapsed_seconds: float,
    pipeline_version: str = "2.0.0",
) -> dict[str, Any]:
    """Assemble the summary execution and data-quality report dictionary."""
    est_col_dist = (
        posters_df["estimated_column_count"].value_counts().sort_index().to_dict()
        if "estimated_column_count" in posters_df.columns
        else {}
    )
    return {
        "pipeline_version": pipeline_version,
        "execution_time_seconds": elapsed_seconds,
        "quality_summary": quality_summary,
        "orientation_distribution": (
            posters_df.groupby(["split", "orientation"]).size().unstack(fill_value=0).to_dict()
        ),
        "estimated_columns_distribution": est_col_dist,
    }


def run_feature_pipeline_v2(
    raw_dir: str | Path,
    out_dir: str | Path,
    export_legacy: bool = True,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = 42,
    stratify_by_orientation: bool = True,
    splits: tuple[str, ...] = SPLITS,
    boundary_tolerance_px: float = 1.0,
    align_tol: float = 0.015,
    knn_k: int = 3,
    grid_res: int = 32,
    overwrite_legacy: bool = True,
    pipeline_version: str = "2.0.0",
) -> dict[str, Any]:
    """Execute the full v2 multi-table feature engineering pipeline and save artifacts."""
    t0 = time.time()
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("[1/4] Pooling source splits, assigning 70/15/15 splits, and validating elements...")
    images_by_split, elements_df, rejected_df, quality_summary = validate_and_extract_elements(
        raw_path,
        splits=splits,
        boundary_tolerance_px=boundary_tolerance_px,
        split_ratios=split_ratios,
        random_state=random_state,
        stratify_by_orientation=stratify_by_orientation,
    )

    print("[2/4] Building sparse pairwise spatial relations graph...")
    relations_df, poster_rel_stats = build_relations_and_spatial_summaries(
        elements_df,
        align_tol=align_tol,
        knn_k=knn_k,
    )

    print("[3/4] Building poster-level & global composition features...")
    posters_df = build_posters_table(
        images_by_split,
        elements_df,
        poster_rel_stats,
        grid_res=grid_res,
    )

    print("[4/4] Saving versioned v2 feature tables and schema metadata...")
    schema_meta = generate_feature_schema_metadata(
        posters_df,
        elements_df,
        relations_df,
        schema_version=pipeline_version,
    )
    elapsed = round(time.time() - t0, 2)
    report = build_pipeline_report(
        posters_df,
        quality_summary,
        elapsed_seconds=elapsed,
        pipeline_version=pipeline_version,
    )
    export_v2_tables_and_metadata(
        out_path,
        posters_df=posters_df,
        elements_df=elements_df,
        relations_df=relations_df,
        rejected_df=rejected_df,
        schema_meta=schema_meta,
        report=report,
    )

    if export_legacy:
        export_legacy_v1_compatible_parquets(
            posters_df,
            elements_df,
            out_path.parent,
            overwrite=overwrite_legacy,
        )

    print(f"Done in {elapsed}s! Outputs written to {out_path}")
    return {
        "posters": posters_df,
        "elements": elements_df,
        "relations": relations_df,
        "rejected": rejected_df,
        "schema": schema_meta,
        "report": report,
    }


def run_pipeline_from_config(
    config: PipelineConfig | str | Path | None = None,
) -> dict[str, Any]:
    """Run the v2 feature pipeline from a PipelineConfig instance or YAML config path."""
    cfg = config if isinstance(config, PipelineConfig) else load_pipeline_config(config)
    return run_feature_pipeline_v2(
        raw_dir=cfg.raw_dir,
        out_dir=cfg.out_dir,
        export_legacy=cfg.export_legacy,
        split_ratios=cfg.split_ratios,
        random_state=cfg.random_state,
        stratify_by_orientation=cfg.stratify_by_orientation,
        splits=cfg.splits,
        boundary_tolerance_px=cfg.boundary_tolerance_px,
        align_tol=cfg.align_tol,
        knn_k=cfg.knn_k,
        grid_res=cfg.grid_res,
        overwrite_legacy=cfg.overwrite_legacy,
        pipeline_version=cfg.pipeline_version,
    )
