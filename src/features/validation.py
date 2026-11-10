"""Step 1: Pool COCO splits, assign deterministic splits, validate bboxes, and build elements table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.config import DEFAULT_SPLIT_RATIOS, EPS, SPLITS, UNKNOWN_CAT_ID
from src.features.helpers import assign_pooled_splits, category_slug


def _load_pooled_coco_splits(
    raw_dir: Path,
    splits: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Load COCO JSON files across source splits and pool images and annotations."""
    raw_image_frames: list[pd.DataFrame] = []
    raw_ann_frames: list[pd.DataFrame] = []
    cat_lookup: dict[int, str] = {}

    for source_split in splits:
        json_path = raw_dir / f"{source_split}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        cat_lookup.update({c["id"]: c["name"] for c in coco["categories"]})
        images_df = pd.DataFrame(coco["images"]).copy()
        ann_df = pd.DataFrame(coco["annotations"]).copy()

        assert images_df["id"].is_unique, f"Duplicate image IDs in {source_split}"
        assert (images_df["width"] > 0).all() and (images_df["height"] > 0).all()

        images_df["source_split"] = source_split
        ann_df["source_split"] = source_split
        raw_image_frames.append(images_df)
        raw_ann_frames.append(ann_df)

    pooled_images_df = pd.concat(raw_image_frames, ignore_index=True)
    assert pooled_images_df["id"].is_unique, "Duplicate image IDs across pooled splits"
    pooled_ann_df = pd.concat(raw_ann_frames, ignore_index=True)
    return pooled_images_df, pooled_ann_df, cat_lookup


def _clip_and_validate_annotations(
    ann_df: pd.DataFrame,
    pooled_images_df: pd.DataFrame,
    cat_lookup: dict[int, str],
    boundary_tolerance_px: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Unpack COCO bboxes, clip minor boundary overshoots, and separate valid vs rejected rows."""
    bbox_cols = pd.DataFrame(
        ann_df["bbox"].tolist(),
        columns=["raw_x", "raw_y", "raw_w", "raw_h"],
        index=ann_df.index,
    ).apply(pd.to_numeric, errors="coerce")

    enriched = pd.concat([ann_df, bbox_cols], axis=1)
    img_meta = pooled_images_df.set_index("id")
    enriched["canvas_width_px"] = enriched["image_id"].map(img_meta["width"]).astype(float)
    enriched["canvas_height_px"] = enriched["image_id"].map(img_meta["height"]).astype(float)
    enriched["split"] = enriched["image_id"].map(img_meta["split"]).fillna(enriched["source_split"])

    x_clip = np.where(
        (enriched["raw_x"] < 0) & (enriched["raw_x"] >= -boundary_tolerance_px),
        0.0,
        enriched["raw_x"],
    )
    y_clip = np.where(
        (enriched["raw_y"] < 0) & (enriched["raw_y"] >= -boundary_tolerance_px),
        0.0,
        enriched["raw_y"],
    )
    x2_raw = x_clip + enriched["raw_w"]
    y2_raw = y_clip + enriched["raw_h"]
    w_clip = np.where(
        (x2_raw > enriched["canvas_width_px"])
        & (x2_raw <= enriched["canvas_width_px"] + boundary_tolerance_px),
        enriched["canvas_width_px"] - x_clip,
        enriched["raw_w"],
    )
    h_clip = np.where(
        (y2_raw > enriched["canvas_height_px"])
        & (y2_raw <= enriched["canvas_height_px"] + boundary_tolerance_px),
        enriched["canvas_height_px"] - y_clip,
        enriched["raw_h"],
    )

    valid_img_ids = set(pooled_images_df["id"])
    is_finite = np.asarray(
        np.isfinite(enriched[["raw_x", "raw_y", "raw_w", "raw_h"]]).all(axis=1),
        dtype=bool,
    )
    has_valid_img = np.asarray(enriched["image_id"].isin(valid_img_ids), dtype=bool)
    has_valid_cat = np.asarray(enriched["category_id"].isin(cat_lookup.keys()), dtype=bool)
    non_neg_origin = np.asarray((x_clip >= 0) & (y_clip >= 0), dtype=bool)
    pos_dims = np.asarray((w_clip > 0) & (h_clip > 0), dtype=bool)
    in_bounds = np.asarray(
        (x_clip + w_clip <= enriched["canvas_width_px"].to_numpy() + 1e-5)
        & (y_clip + h_clip <= enriched["canvas_height_px"].to_numpy() + 1e-5),
        dtype=bool,
    )

    valid_mask = (
        is_finite
        & has_valid_img
        & has_valid_cat
        & non_neg_origin
        & pos_dims
        & in_bounds
    )
    inv_mask = ~valid_mask

    if inv_mask.any():
        rej = enriched.loc[inv_mask].copy()
        reasons = np.full(len(rej), "unknown_error", dtype=object)
        reasons = np.where(~in_bounds[inv_mask], "exceeds_canvas_bounds", reasons)
        reasons = np.where(~non_neg_origin[inv_mask], "negative_origin", reasons)
        reasons = np.where(~pos_dims[inv_mask], "non_positive_dimensions", reasons)
        reasons = np.where(~has_valid_cat[inv_mask], "invalid_category_id", reasons)
        reasons = np.where(~has_valid_img[inv_mask], "orphan_image_id", reasons)
        reasons = np.where(~is_finite[inv_mask], "non_finite_coordinates", reasons)
        rej["rejection_reason"] = reasons
        rejected_df = (
            rej[
                [
                    "id",
                    "image_id",
                    "split",
                    "source_split",
                    "category_id",
                    "raw_x",
                    "raw_y",
                    "raw_w",
                    "raw_h",
                    "canvas_width_px",
                    "canvas_height_px",
                    "rejection_reason",
                ]
            ]
            .rename(columns={"id": "element_id", "image_id": "poster_id"})
            .reset_index(drop=True)
        )
    else:
        rejected_df = pd.DataFrame()

    val = enriched.loc[valid_mask].copy()
    val["raw_x"] = x_clip[valid_mask]
    val["raw_y"] = y_clip[valid_mask]
    val["raw_w"] = w_clip[valid_mask]
    val["raw_h"] = h_clip[valid_mask]
    return enriched, val, rejected_df


def _build_valid_elements_df(val: pd.DataFrame, cat_lookup: dict[int, str]) -> pd.DataFrame:
    """Compute normalized geometry, within-poster ranks, and positional bins for valid elements."""
    width_px = val["canvas_width_px"].to_numpy()
    height_px = val["canvas_height_px"].to_numpy()
    x = val["raw_x"].to_numpy()
    y = val["raw_y"].to_numpy()
    w = val["raw_w"].to_numpy()
    h = val["raw_h"].to_numpy()

    x0 = np.clip(x / width_px, 0.0, 1.0)
    y0 = np.clip(y / height_px, 0.0, 1.0)
    w_norm = np.clip(w / width_px, EPS, 1.0)
    h_norm = np.clip(h / height_px, EPS, 1.0)
    x1 = np.clip(x0 + w_norm, 0.0, 1.0)
    y1 = np.clip(y0 + h_norm, 0.0, 1.0)
    cx = np.clip(x0 + w_norm / 2.0, 0.0, 1.0)
    cy = np.clip(y0 + h_norm / 2.0, 0.0, 1.0)
    area_norm = w_norm * h_norm
    bbox_aspect = w_norm / (h_norm + EPS)

    cat_names = val["category_id"].map(cat_lookup)
    cat_slugs = cat_names.map(category_slug)

    elements_df = pd.DataFrame(
        {
            "poster_id": val["image_id"].to_numpy(dtype=np.int64),
            "element_id": val["id"].to_numpy(dtype=np.int64),
            "split": val["split"].to_numpy(),
            "source_split": val["source_split"].to_numpy(),
            "category_id": val["category_id"].to_numpy(dtype=np.int64),
            "category_name": cat_names.to_numpy(),
            "category_slug": cat_slugs.to_numpy(),
            "is_unknown": (val["category_id"].to_numpy() == UNKNOWN_CAT_ID),
            "canvas_width_px": width_px,
            "canvas_height_px": height_px,
            "raw_x": x,
            "raw_y": y,
            "raw_w": w,
            "raw_h": h,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "w_norm": w_norm,
            "h_norm": h_norm,
            "cx": cx,
            "cy": cy,
            "area_norm": area_norm,
            "bbox_aspect_ratio": bbox_aspect,
            "log_w_norm": np.log(w_norm + EPS),
            "log_h_norm": np.log(h_norm + EPS),
            "log_area_norm": np.log(area_norm + EPS),
            "log_bbox_aspect_ratio": np.log(bbox_aspect + EPS),
            "left_margin": x0,
            "right_margin": np.maximum(0.0, 1.0 - x1),
            "top_margin": y0,
            "bottom_margin": np.maximum(0.0, 1.0 - y1),
        }
    )

    elements_df["min_horizontal_margin"] = np.minimum(
        elements_df["left_margin"], elements_df["right_margin"]
    )
    elements_df["min_vertical_margin"] = np.minimum(
        elements_df["top_margin"], elements_df["bottom_margin"]
    )
    elements_df["center_dist_to_edge"] = np.minimum.reduce(
        [
            elements_df["cx"],
            1.0 - elements_df["cx"],
            elements_df["cy"],
            1.0 - elements_df["cy"],
        ]
    )

    poster_med_area = elements_df.groupby("poster_id")["area_norm"].transform("median")
    elements_df["ratio_to_poster_median_area"] = elements_df["area_norm"] / (poster_med_area + EPS)
    elements_df["poster_area_rank_pct"] = elements_df.groupby("poster_id")["area_norm"].rank(
        pct=True
    )
    elements_df["category_area_rank_pct"] = elements_df.groupby(["poster_id", "category_slug"])[
        "area_norm"
    ].rank(pct=True)
    desc_rank = elements_df.groupby("poster_id")["area_norm"].rank(ascending=False, method="min")
    elements_df["is_top3_largest"] = desc_rank <= 3

    elements_df["horiz_zone"] = np.select(
        [elements_df["cx"] < 1.0 / 3.0, elements_df["cx"] <= 2.0 / 3.0],
        ["left", "center"],
        default="right",
    )
    elements_df["vert_zone"] = np.select(
        [elements_df["cy"] < 0.25, elements_df["cy"] < 0.50, elements_df["cy"] < 0.75],
        ["top", "upper_mid", "lower_mid"],
        default="bottom",
    )
    gx = np.clip((elements_df["cx"].to_numpy() * 4).astype(int), 0, 3)
    gy = np.clip((elements_df["cy"].to_numpy() * 4).astype(int), 0, 3)
    elements_df["grid_4x4_cell"] = gy * 4 + gx
    return elements_df


def _attach_train_category_ratios(elements_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit category reference medians strictly on the train split and compute ratio columns."""
    train_els = elements_df[elements_df["split"] == "train"]
    train_cat_medians = (
        train_els.groupby("category_slug")[["w_norm", "h_norm", "area_norm"]]
        .median()
        .to_dict(orient="index")
    )
    global_train_medians = {
        "w_norm": float(train_els["w_norm"].median()) if len(train_els) > 0 else 0.0,
        "h_norm": float(train_els["h_norm"].median()) if len(train_els) > 0 else 0.0,
        "area_norm": float(train_els["area_norm"].median()) if len(train_els) > 0 else 0.0,
    }

    ref_w = elements_df["category_slug"].map(
        lambda s: train_cat_medians.get(s, global_train_medians)["w_norm"]
    )
    ref_h = elements_df["category_slug"].map(
        lambda s: train_cat_medians.get(s, global_train_medians)["h_norm"]
    )
    ref_area = elements_df["category_slug"].map(
        lambda s: train_cat_medians.get(s, global_train_medians)["area_norm"]
    )

    elements_df["ratio_w_to_train_cat_median"] = elements_df["w_norm"] / (ref_w + EPS)
    elements_df["ratio_h_to_train_cat_median"] = elements_df["h_norm"] / (ref_h + EPS)
    elements_df["ratio_area_to_train_cat_median"] = elements_df["area_norm"] / (ref_area + EPS)
    return train_cat_medians


def validate_and_extract_elements(
    raw_dir: Path,
    splits: tuple[str, ...] = SPLITS,
    boundary_tolerance_px: float = 1.0,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = 42,
    stratify_by_orientation: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Step 1: Pool source JSONs, re-split 70/15/15, validate annotations, and build elements table."""
    raw_dir = Path(raw_dir)
    pooled_images_df, raw_ann_df, cat_lookup = _load_pooled_coco_splits(raw_dir, splits)

    pooled_images_df = assign_pooled_splits(
        pooled_images_df,
        split_ratios=split_ratios,
        random_state=random_state,
        stratify_by_orientation=stratify_by_orientation,
    )
    images_by_split: dict[str, pd.DataFrame] = {
        split: pooled_images_df[pooled_images_df["split"] == split].reset_index(drop=True)
        for split in SPLITS
    }

    enriched_ann_df, valid_ann_df, rejected_df = _clip_and_validate_annotations(
        raw_ann_df,
        pooled_images_df,
        cat_lookup,
        boundary_tolerance_px,
    )
    elements_df = _build_valid_elements_df(valid_ann_df, cat_lookup)
    train_cat_medians = _attach_train_category_ratios(elements_df)

    quality_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_imgs = images_by_split[split]
        split_valid = elements_df[elements_df["split"] == split]
        split_anns_count = int((enriched_ann_df["split"] == split).sum())
        split_rej_count = (
            int((rejected_df["split"] == split).sum()) if len(rejected_df) > 0 else 0
        )
        quality_rows.append(
            {
                "split": split,
                "total_posters": len(split_imgs),
                "posters_retained": len(split_imgs),
                "posters_with_valid_elements": int(split_valid["poster_id"].nunique()),
                "total_annotations": split_anns_count,
                "valid_annotations": len(split_valid),
                "rejected_annotations": split_rej_count,
            }
        )

    source_to_new_counts = (
        pooled_images_df.groupby(["source_split", "split"]).size().unstack(fill_value=0).to_dict()
    )

    quality_summary: dict[str, Any] = {
        "split_policy": {
            "mode": "pooled_resplit",
            "source_splits": list(splits),
            "split_ratios": list(split_ratios),
            "random_state": random_state,
            "stratify_by_orientation": stratify_by_orientation,
        },
        "splits": quality_rows,
        "source_split_to_new_split_counts": source_to_new_counts,
        "train_category_reference_medians": train_cat_medians,
        "rejected_by_reason": (
            rejected_df["rejection_reason"].value_counts().to_dict()
            if len(rejected_df) > 0
            else {}
        ),
    }

    return images_by_split, elements_df, rejected_df, quality_summary
