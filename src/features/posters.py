"""Step 3: Build canonical poster-level & global composition features table (Groups A & D)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.features.config import (
    BODY_STRUCT_CAT_IDS,
    EPS,
    KNOWN_CATEGORY_SLUGS,
    TEXTUAL_CAT_IDS,
    VISUAL_CAT_IDS,
)
from src.features.helpers import classify_aspect_orientation


def _build_base_poster_row(
    pid: int,
    split: str,
    source_split: str,
    width_px: float,
    height_px: float,
    total_count: int,
    known_count: int,
    unk_count: int,
) -> dict[str, Any]:
    """Create shared poster metadata and element-count fields."""
    aspect = width_px / height_px
    orientation = classify_aspect_orientation(aspect)
    return {
        "poster_id": pid,
        "split": split,
        "source_split": source_split,
        "canvas_width_px": width_px,
        "canvas_height_px": height_px,
        "aspect_ratio": aspect,
        "log_aspect_ratio": float(np.log(aspect + EPS)),
        "orientation": orientation,
        "is_landscape": float(orientation == "landscape"),
        "is_portrait": float(orientation == "portrait"),
        "total_element_count": total_count,
        "known_element_count": known_count,
        "unknown_element_count": unk_count,
        "unknown_element_fraction": float(unk_count / total_count) if total_count > 0 else 0.0,
    }


def _add_category_inventory_features(
    row: dict[str, Any],
    cat_slugs_arr: np.ndarray,
    geom_5_stack: np.ndarray,
    area_all: np.ndarray,
    known_count: int,
) -> None:
    """Populate Group A2 & A4 per-category counts, presence flags, and geometry quantiles."""
    denom_known = max(1, known_count)
    gcol_names = ("w_norm", "h_norm", "area_norm", "cx", "cy")

    for slug in KNOWN_CATEGORY_SLUGS:
        c_mask = cat_slugs_arr == slug
        c = int(np.sum(c_mask))
        row[f"count_{slug}"] = c
        row[f"log1p_count_{slug}"] = float(np.log1p(c))
        row[f"has_{slug}"] = float(c > 0)
        row[f"fraction_{slug}"] = float(c / denom_known)

        if c > 0:
            row[f"area_sum_{slug}"] = float(np.sum(area_all[c_mask]))
            if c == 1:
                single_vals = geom_5_stack[:, c_mask][:, 0]
                for g_i, gcol in enumerate(gcol_names):
                    v = float(single_vals[g_i])
                    row[f"{gcol}_p25_{slug}"] = v
                    row[f"{gcol}_p50_{slug}"] = v
                    row[f"{gcol}_p75_{slug}"] = v
            else:
                q_mat = np.percentile(geom_5_stack[:, c_mask], [25, 50, 75], axis=1)
                for g_i, gcol in enumerate(gcol_names):
                    row[f"{gcol}_p25_{slug}"] = float(q_mat[0, g_i])
                    row[f"{gcol}_p50_{slug}"] = float(q_mat[1, g_i])
                    row[f"{gcol}_p75_{slug}"] = float(q_mat[2, g_i])
        else:
            row[f"area_sum_{slug}"] = 0.0
            for gcol in gcol_names:
                row[f"{gcol}_p25_{slug}"] = np.nan
                row[f"{gcol}_p50_{slug}"] = np.nan
                row[f"{gcol}_p75_{slug}"] = np.nan

    row["visual_element_count"] = row["count_figure"] + row["count_table"]
    row["textual_element_count"] = (
        row["count_text"] + row["count_list"] + row["count_caption"]
    )
    row["heading_element_count"] = row["count_title"] + row["count_section"]
    row["media_to_text_count_ratio"] = float(
        row["visual_element_count"] / (row["textual_element_count"] + 1.0)
    )


def _add_raster_and_column_features(
    row: dict[str, Any],
    x0_all: np.ndarray,
    y0_all: np.ndarray,
    x1_all: np.ndarray,
    y1_all: np.ndarray,
    w_all: np.ndarray,
    cat_ids: np.ndarray,
    total_box_area_sum: float,
    grid_res: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Populate Group A6 & Group D raster occupancy, profiles, and column/row structure."""
    canvas_mask = np.zeros((grid_res, grid_res), dtype=np.float32)
    body_horiz_profile = np.zeros(grid_res, dtype=np.float32)
    body_vert_profile = np.zeros(grid_res, dtype=np.float32)
    text_mask = np.zeros((grid_res, grid_res), dtype=np.float32)
    vis_mask = np.zeros((grid_res, grid_res), dtype=np.float32)

    for bx0, by0, bx1, by1, cid in zip(x0_all, y0_all, x1_all, y1_all, cat_ids):
        ix0 = max(0, min(grid_res - 1, int(np.floor(bx0 * grid_res))))
        ix1 = max(ix0 + 1, min(grid_res, int(np.ceil(bx1 * grid_res))))
        iy0 = max(0, min(grid_res - 1, int(np.floor(by0 * grid_res))))
        iy1 = max(iy0 + 1, min(grid_res, int(np.ceil(by1 * grid_res))))

        canvas_mask[iy0:iy1, ix0:ix1] = 1.0
        if cid in TEXTUAL_CAT_IDS:
            text_mask[iy0:iy1, ix0:ix1] = 1.0
        elif cid in VISUAL_CAT_IDS:
            vis_mask[iy0:iy1, ix0:ix1] = 1.0

        if cid in BODY_STRUCT_CAT_IDS and (bx1 - bx0) < 0.80:
            body_horiz_profile[ix0:ix1] += float(by1 - by0)
            body_vert_profile[iy0:iy1] += float(bx1 - bx0)

    union_cov = float(np.mean(canvas_mask))
    row["union_box_coverage"] = union_cov
    row["overlap_excess"] = max(0.0, total_box_area_sum - union_cov)
    row["unannotated_whitespace_fraction"] = max(0.0, 1.0 - union_cov)

    grid_4x4 = canvas_mask.reshape(4, grid_res // 4, 4, grid_res // 4).mean(axis=(1, 3))
    for r_idx in range(4):
        for c_idx in range(4):
            row[f"occ_grid_r{r_idx}_c{c_idx}"] = float(grid_4x4[r_idx, c_idx])

    half = grid_res // 2
    row["text_occ_top_half"] = float(text_mask[:half, :].mean())
    row["text_occ_bottom_half"] = float(text_mask[half:, :].mean())
    row["visual_occ_left_half"] = float(vis_mask[:, :half].mean())
    row["visual_occ_right_half"] = float(vis_mask[:, half:].mean())
    row["visual_occ_center"] = float(
        vis_mask[grid_res // 4 : 3 * grid_res // 4, grid_res // 4 : 3 * grid_res // 4].mean()
    )

    h_prof_16 = canvas_mask.mean(axis=0).reshape(16, grid_res // 16).mean(axis=1)
    v_prof_16 = canvas_mask.mean(axis=1).reshape(16, grid_res // 16).mean(axis=1)
    for b_idx in range(16):
        row[f"horiz_occ_{b_idx:02d}"] = float(h_prof_16[b_idx])
        row[f"vert_occ_{b_idx:02d}"] = float(v_prof_16[b_idx])

    if body_horiz_profile.max() > 0:
        prof_norm = body_horiz_profile / body_horiz_profile.max()
        smoothed = np.convolve(prof_norm, np.array([0.2, 0.6, 0.2]), mode="same")
        peaks, _ = find_peaks(smoothed, height=0.25, distance=4, prominence=0.08)
        est_cols = max(1, min(6, len(peaks)))
        if len(peaks) >= 2:
            valleys = [
                float(np.min(smoothed[peaks[p_idx] : peaks[p_idx + 1] + 1]))
                for p_idx in range(len(peaks) - 1)
            ]
            mean_peak = float(np.mean(smoothed[peaks]))
            mean_valley = float(np.mean(valleys))
            col_conf = float(np.clip((mean_peak - mean_valley) / (mean_peak + EPS), 0.0, 1.0))
            gutter_width_est = float(np.mean(smoothed < 0.25))
        else:
            col_conf = 0.5
            gutter_width_est = float(1.0 - union_cov)
    else:
        est_cols = 1
        col_conf = 0.0
        gutter_width_est = 1.0

    if body_vert_profile.max() > 0:
        v_norm = body_vert_profile / body_vert_profile.max()
        v_smooth = np.convolve(v_norm, np.array([0.2, 0.6, 0.2]), mode="same")
        v_peaks, _ = find_peaks(v_smooth, height=0.25, distance=4, prominence=0.08)
        est_rows = max(1, min(6, len(v_peaks)))
    else:
        est_rows = 1

    col_unit_width = 1.0 / max(1, est_cols)
    row["estimated_column_count"] = int(est_cols)
    row["column_assignment_confidence"] = col_conf
    row["estimated_gutter_fraction"] = gutter_width_est
    row["estimated_row_band_count"] = int(est_rows)
    row["full_width_region_count"] = int(np.sum(w_all >= 0.70))
    row["multi_column_span_count"] = int(
        np.sum((w_all >= 1.5 * col_unit_width) & (w_all < 0.70))
    )
    return h_prof_16, v_prof_16


def _add_balance_and_symmetry_features(
    row: dict[str, Any],
    h_prof_16: np.ndarray,
    v_prof_16: np.ndarray,
    cx_all: np.ndarray,
    cy_all: np.ndarray,
    area_all: np.ndarray,
    total_box_area_sum: float,
) -> None:
    """Populate Group D5 spatial balance, cosine profile symmetry, and center of mass."""
    left_occ = float(h_prof_16[:8].mean())
    right_occ = float(h_prof_16[8:].mean())
    top_occ = float(v_prof_16[:8].mean())
    bot_occ = float(v_prof_16[8:].mean())

    row["left_right_area_balance"] = float(
        (left_occ - right_occ) / (left_occ + right_occ + EPS)
    )
    row["top_bottom_area_balance"] = float(
        (top_occ - bot_occ) / (top_occ + bot_occ + EPS)
    )

    h_left, h_right_rev = h_prof_16[:8], h_prof_16[8:][::-1]
    h_denom = float(np.linalg.norm(h_left) * np.linalg.norm(h_right_rev))
    row["horiz_profile_symmetry"] = (
        float(np.dot(h_left, h_right_rev) / h_denom) if h_denom > EPS else 0.0
    )

    v_top, v_bot_rev = v_prof_16[:8], v_prof_16[8:][::-1]
    v_denom = float(np.linalg.norm(v_top) * np.linalg.norm(v_bot_rev))
    row["vert_profile_symmetry"] = (
        float(np.dot(v_top, v_bot_rev) / v_denom) if v_denom > EPS else 0.0
    )

    row["center_of_mass_x"] = float(np.sum(cx_all * area_all) / (total_box_area_sum + EPS))
    row["center_of_mass_y"] = float(np.sum(cy_all * area_all) / (total_box_area_sum + EPS))


def build_posters_table(
    images_by_split: dict[str, pd.DataFrame],
    elements_df: pd.DataFrame,
    poster_rel_stats: dict[int, dict[str, float]],
    grid_res: int = 32,
) -> pd.DataFrame:
    """Step 3: Build canonical poster & global composition table (`posters.parquet`, Groups A & D)."""
    all_images = pd.concat(list(images_by_split.values()), ignore_index=True)
    el_groups = {int(pid): grp for pid, grp in elements_df.groupby("poster_id", sort=False)}
    poster_rows: list[dict[str, Any]] = []

    for _, img in all_images.iterrows():
        pid = int(img["id"])
        split = str(img["split"])
        source_split = str(img["source_split"]) if "source_split" in img else split
        width_px = float(img["width"])
        height_px = float(img["height"])

        grp = el_groups.get(pid)
        if grp is None or len(grp) == 0:
            poster_rows.append(
                _build_base_poster_row(
                    pid, split, source_split, width_px, height_px, 0, 0, 0
                )
            )
            continue

        total_count = len(grp)
        unk_count = int(np.sum(grp["is_unknown"].to_numpy()))
        known_count = total_count - unk_count
        row = _build_base_poster_row(
            pid,
            split,
            source_split,
            width_px,
            height_px,
            total_count,
            known_count,
            unk_count,
        )

        cat_slugs_arr = grp["category_slug"].to_numpy()
        cat_ids = grp["category_id"].to_numpy()
        x0_all = grp["x0"].to_numpy()
        y0_all = grp["y0"].to_numpy()
        x1_all = grp["x1"].to_numpy()
        y1_all = grp["y1"].to_numpy()
        w_all = grp["w_norm"].to_numpy()
        h_all = grp["h_norm"].to_numpy()
        cx_all = grp["cx"].to_numpy()
        cy_all = grp["cy"].to_numpy()
        area_all = grp["area_norm"].to_numpy()
        log_area_all = grp["log_area_norm"].to_numpy()

        geom_5_stack = np.vstack([w_all, h_all, area_all, cx_all, cy_all])
        _add_category_inventory_features(row, cat_slugs_arr, geom_5_stack, area_all, known_count)

        overall_4_stack = np.vstack([w_all, h_all, area_all, log_area_all])
        ov_q = np.percentile(overall_4_stack, [10, 25, 50, 75, 90], axis=1)
        for col_i, col_name in enumerate(("w_norm", "h_norm", "area_norm", "log_area_norm")):
            row[f"{col_name}_p10"] = float(ov_q[0, col_i])
            row[f"{col_name}_p25"] = float(ov_q[1, col_i])
            row[f"{col_name}_p50"] = float(ov_q[2, col_i])
            row[f"{col_name}_p75"] = float(ov_q[3, col_i])
            row[f"{col_name}_p90"] = float(ov_q[4, col_i])
            row[f"{col_name}_iqr"] = float(ov_q[3, col_i] - ov_q[1, col_i])

        row["visual_area_sum"] = row["area_sum_figure"] + row["area_sum_table"]
        row["textual_area_sum"] = (
            row["area_sum_text"] + row["area_sum_list"] + row["area_sum_caption"]
        )
        row["header_area_sum"] = row["area_sum_title"] + row["area_sum_author_info"]
        total_box_area_sum = float(np.sum(area_all))
        row["sum_box_area_fraction"] = total_box_area_sum
        row["visual_to_body_area_ratio"] = float(
            row["visual_area_sum"]
            / (row["visual_area_sum"] + row["textual_area_sum"] + EPS)
        )

        c_left = float(np.min(x0_all))
        c_right = float(max(0.0, 1.0 - np.max(x1_all)))
        c_top = float(np.min(y0_all))
        c_bottom = float(max(0.0, 1.0 - np.max(y1_all)))
        ext_w = float(max(0.0, 1.0 - c_left - c_right))
        ext_h = float(max(0.0, 1.0 - c_top - c_bottom))

        row["content_left_margin"] = c_left
        row["content_right_margin"] = c_right
        row["content_top_margin"] = c_top
        row["content_bottom_margin"] = c_bottom
        row["content_extent_width"] = ext_w
        row["content_extent_height"] = ext_h
        row["content_extent_area"] = ext_w * ext_h

        hdr_mask = ((cat_ids == 487601213) | (cat_ids == 674852286)) & (cy_all < 0.35)
        row["header_band_bottom_y"] = (
            float(np.max(y1_all[hdr_mask])) if np.any(hdr_mask) else float(c_top)
        )

        h_prof_16, v_prof_16 = _add_raster_and_column_features(
            row, x0_all, y0_all, x1_all, y1_all, w_all, cat_ids, total_box_area_sum, grid_res
        )
        _add_balance_and_symmetry_features(
            row, h_prof_16, v_prof_16, cx_all, cy_all, area_all, total_box_area_sum
        )

        row.update(poster_rel_stats.get(pid, {}))
        poster_rows.append(row)

    return pd.DataFrame(poster_rows)
