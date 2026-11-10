"""Step 4: Parquet/JSON exports and legacy v1 feature table compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.config import SCIPOST_CATEGORIES, SPLITS
from src.features.helpers import category_slug


def export_v2_tables_and_metadata(
    out_dir: Path,
    posters_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    schema_meta: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """Persist versioned v2 Parquet tables, schema metadata, and pipeline quality report."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    posters_df.to_parquet(out_path / "posters.parquet", index=False)
    elements_df.to_parquet(out_path / "elements.parquet", index=False)
    relations_df.to_parquet(out_path / "relations.parquet", index=False)
    if len(rejected_df) > 0:
        rejected_df.to_parquet(out_path / "rejected_annotations.parquet", index=False)

    (out_path / "feature_schema.json").write_text(
        json.dumps(schema_meta, indent=2), encoding="utf-8"
    )
    (out_path / "feature_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def export_legacy_v1_compatible_parquets(
    posters_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    legacy_out_dir: Path,
    overwrite: bool = True,
) -> None:
    """Ensure legacy v1 parquet files exist and stay aligned with current splits."""
    legacy_out_dir = Path(legacy_out_dir)
    legacy_out_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and (legacy_out_dir / "train_features.parquet").exists():
        return

    for split in SPLITS:
        p_sub = posters_df[posters_df["split"] == split].copy()
        e_sub = elements_df[elements_df["split"] == split].copy()

        df = pd.DataFrame(
            {
                "poster_id": p_sub["poster_id"].to_numpy(),
                "split": split,
                "aspect_ratio": p_sub["aspect_ratio"].to_numpy(),
                "pixel_width": p_sub["canvas_width_px"].to_numpy(),
                "pixel_height": p_sub["canvas_height_px"].to_numpy(),
                "element_count": p_sub["total_element_count"].to_numpy(),
                "element_density": p_sub["total_element_count"].astype(float).to_numpy(),
                "sum_bbox_area_ratio": p_sub["sum_box_area_fraction"].to_numpy(),
            }
        )

        geom = (
            e_sub.groupby("poster_id")
            .agg(
                median_element_area=("area_norm", "median"),
                median_element_width=("w_norm", "median"),
                median_element_height=("h_norm", "median"),
                median_center_x=("cx", "median"),
                median_center_y=("cy", "median"),
                spread_center_x=("cx", lambda s: s.std(ddof=0)),
                spread_center_y=("cy", lambda s: s.std(ddof=0)),
            )
            .reindex(df["poster_id"])
            .fillna(0.0)
            .reset_index(drop=True)
        )
        df = pd.concat([df, geom], axis=1)

        counts = df["element_count"].to_numpy()
        for cid, cname in SCIPOST_CATEGORIES.items():
            slug = category_slug(cname)
            c_sub = e_sub[e_sub["category_id"] == cid]
            c_cnt = (
                c_sub.groupby("poster_id")
                .size()
                .reindex(df["poster_id"], fill_value=0)
                .to_numpy()
            )
            df[f"count_{slug}"] = c_cnt
            df[f"proportion_{slug}"] = np.divide(
                c_cnt, counts, out=np.zeros(len(counts), dtype=float), where=counts > 0
            )
            c_geom = (
                c_sub.groupby("poster_id")
                .agg(
                    area=("area_norm", "median"),
                    width=("w_norm", "median"),
                    height=("h_norm", "median"),
                    center_x=("cx", "median"),
                    center_y=("cy", "median"),
                    spread_x=("cx", lambda s: s.std(ddof=0)),
                    spread_y=("cy", lambda s: s.std(ddof=0)),
                )
                .reindex(df["poster_id"])
                .fillna(0.0)
                .reset_index(drop=True)
            )
            for stat in c_geom.columns:
                df[f"{stat}_{slug}"] = c_geom[stat].to_numpy()

        df.to_parquet(legacy_out_dir / f"{split}_features.parquet", index=False)
