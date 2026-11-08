"""Group E: Visual Appearance and Design Style Feature Extractor.

Extracts pixel-level design descriptors from poster PNGs:
  - Background color & luminance (border sampling)
  - Dominant color palette
  - Luminance, saturation, and Hasler-Süsstrunk colorfulness
  - Edge density, local contrast, and Shannon entropy (visual complexity)
  - Region-specific contrast (Text vs Figure visual complexity)

Kept strictly separate from the structural geometry pipeline so layout retrieval
and generation models never depend on features unavailable from user text/asset inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def extract_single_poster_visual_features(
    img_path: Path,
    elements_sub: pd.DataFrame | None = None,
    resize_dim: tuple[int, int] = (256, 256),
) -> dict[str, Any]:
    """Extract low-cost visual appearance and complexity features from a poster image."""
    with Image.open(img_path) as im:
        im_rgb = im.convert("RGB").resize(resize_dim, Image.Resampling.BILINEAR)
        arr = np.asarray(im_rgb, dtype=np.float32) / 255.0

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Perceived luminance (ITU-R BT.709)
    lum = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # HSV Saturation proxy: (max - min) / (max + 1e-6)
    c_max = np.maximum(np.maximum(R, G), B)
    c_min = np.minimum(np.minimum(R, G), B)
    sat = (c_max - c_min) / (c_max + 1e-6)

    # Hasler & Süsstrunk (2003) colorfulness metric
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, std_yb = float(np.std(rg)), float(np.std(yb))
    mean_rg, mean_yb = float(np.mean(rg)), float(np.mean(yb))
    colorfulness = float(
        np.hypot(std_rg, std_yb) + 0.3 * np.hypot(mean_rg, mean_yb)
    )

    # Border sampling (outer 4% border) for background color estimation
    bw = max(2, int(resize_dim[0] * 0.04))
    border_pixels = np.concatenate(
        [
            arr[:bw, :, :].reshape(-1, 3),
            arr[-bw:, :, :].reshape(-1, 3),
            arr[:, :bw, :].reshape(-1, 3),
            arr[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_rgb = np.median(border_pixels, axis=0)
    bg_lum = float(0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2])

    # Edge density via finite differences (gradient magnitude)
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    grad_mag = np.hypot(gx, gy)
    edge_density = float(np.mean(grad_mag > 0.08))

    # Local RMS contrast in 8x8 blocks
    H_r, W_r = lum.shape
    blocks = lum.reshape(8, H_r // 8, 8, W_r // 8)
    local_contrast = float(np.mean(np.std(blocks, axis=(1, 3))))

    # Shannon luminance entropy (32 bins)
    hist, _ = np.histogram(lum, bins=32, range=(0.0, 1.0), density=True)
    hist_p = hist / (np.sum(hist) + 1e-9)
    hist_p = hist_p[hist_p > 0]
    visual_entropy = float(-np.sum(hist_p * np.log2(hist_p)))

    # Region-specific visual contrast (Text vs Figure crops)
    text_local_contrast = np.nan
    figure_colorfulness = np.nan
    if elements_sub is not None and len(elements_sub) > 0:
        t_boxes = elements_sub[elements_sub["category_slug"] == "text"]
        f_boxes = elements_sub[elements_sub["category_slug"] == "figure"]

        if len(t_boxes) > 0:
            t_contrasts = []
            for _, trow in t_boxes.head(5).iterrows():
                x0 = int(np.clip(trow["x0"] * W_r, 0, W_r - 2))
                x1 = int(np.clip(trow["x1"] * W_r, x0 + 1, W_r))
                y0 = int(np.clip(trow["y0"] * H_r, 0, H_r - 2))
                y1 = int(np.clip(trow["y1"] * H_r, y0 + 1, H_r))
                crop_lum = lum[y0:y1, x0:x1]
                if crop_lum.size > 4:
                    t_contrasts.append(float(np.std(crop_lum)))
            if t_contrasts:
                text_local_contrast = float(np.mean(t_contrasts))

        if len(f_boxes) > 0:
            f_cols = []
            for _, frow in f_boxes.head(5).iterrows():
                x0 = int(np.clip(frow["x0"] * W_r, 0, W_r - 2))
                x1 = int(np.clip(frow["x1"] * W_r, x0 + 1, W_r))
                y0 = int(np.clip(frow["y0"] * H_r, 0, H_r - 2))
                y1 = int(np.clip(frow["y1"] * H_r, y0 + 1, H_r))
                crop_rg = rg[y0:y1, x0:x1]
                crop_yb = yb[y0:y1, x0:x1]
                if crop_rg.size > 4:
                    f_cols.append(float(np.hypot(np.std(crop_rg), np.std(crop_yb))))
            if f_cols:
                figure_colorfulness = float(np.mean(f_cols))

    return {
        "mean_luminance": float(np.mean(lum)),
        "std_luminance": float(np.std(lum)),
        "mean_saturation": float(np.mean(sat)),
        "std_saturation": float(np.std(sat)),
        "colorfulness_index": colorfulness,
        "bg_r": float(bg_rgb[0]),
        "bg_g": float(bg_rgb[1]),
        "bg_b": float(bg_rgb[2]),
        "bg_luminance": bg_lum,
        "is_dark_background": float(bg_lum < 0.35),
        "edge_density": edge_density,
        "local_contrast": local_contrast,
        "visual_entropy": visual_entropy,
        "text_region_contrast": text_local_contrast,
        "figure_region_colorfulness": figure_colorfulness,
    }


def extract_visual_features_for_sample(
    png_root_dir: str | Path,
    posters_df: pd.DataFrame,
    elements_df: pd.DataFrame | None = None,
    max_per_split: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Extract Group E visual appearance features for a stratified sample across splits."""
    import json

    png_root = Path(png_root_dir)
    rng = np.random.default_rng(random_state)
    el_groups = (
        {int(pid): grp for pid, grp in elements_df.groupby("poster_id")}
        if elements_df is not None
        else {}
    )

    records: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        sub = posters_df[posters_df["split"] == split]
        if len(sub) == 0:
            continue

        # Load id -> file_name mapping from split JSON
        json_path = png_root / f"{split}.json"
        id_to_fname: dict[int, str] = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                coco_meta = json.load(f)
            id_to_fname = {int(im["id"]): str(im["file_name"]) for im in coco_meta["images"]}

        n_pick = min(max_per_split, len(sub))
        sampled_ids = rng.choice(sub["poster_id"].to_numpy(), size=n_pick, replace=False)

        for pid in sampled_ids:
            pid_int = int(pid)
            fname = id_to_fname.get(pid_int, f"{pid_int}.png")
            img_path = png_root / split / fname
            if not img_path.exists():
                continue
            try:
                feats = extract_single_poster_visual_features(
                    img_path, elements_sub=el_groups.get(pid_int)
                )
                feats["poster_id"] = pid_int
                feats["file_name"] = fname
                feats["split"] = split
                records.append(feats)
            except Exception:
                continue

    return pd.DataFrame(records)
