"""Shared utility functions for category normalization, orientation, and split assignment."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.features.config import (
    DEFAULT_SPLIT_RATIOS,
    LANDSCAPE_MIN_ASPECT,
    PORTRAIT_MAX_ASPECT,
    SPLITS,
)


def category_slug(name: str) -> str:
    """Create stable snake_case identifier from a category name."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return slug or "unnamed"


def classify_aspect_orientation(aspect: float) -> str:
    """Classify a single poster aspect ratio into portrait, landscape, or square."""
    if aspect < PORTRAIT_MAX_ASPECT:
        return "portrait"
    if aspect > LANDSCAPE_MIN_ASPECT:
        return "landscape"
    return "square"


def compute_orientation(widths: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Classify vectorized poster widths and heights into portrait, landscape, or square."""
    aspect = widths / heights
    return np.select(
        [aspect < PORTRAIT_MAX_ASPECT, aspect > LANDSCAPE_MIN_ASPECT],
        ["portrait", "landscape"],
        default="square",
    )


# Backward-compatible alias
_compute_orientation = compute_orientation


def assign_pooled_splits(
    pooled_images_df: pd.DataFrame,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    random_state: int = 42,
    stratify_by_orientation: bool = True,
) -> pd.DataFrame:
    """Deterministically assign pooled posters into train/dev/test splits."""
    if len(split_ratios) != 3 or any(r <= 0 for r in split_ratios):
        raise ValueError(f"split_ratios must contain 3 positive floats, got {split_ratios}")
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(
            f"split_ratios must sum to 1.0, got {split_ratios} (sum={sum(split_ratios)})"
        )

    df = pooled_images_df.sort_values("id", kind="mergesort").reset_index(drop=True).copy()
    r_train, r_dev, _ = split_ratios
    rng = np.random.default_rng(random_state)
    assigned_splits = np.empty(len(df), dtype=object)

    def _assign_group(indices: np.ndarray) -> None:
        perm = indices[rng.permutation(len(indices))]
        n = len(perm)
        n_train = int(round(n * r_train))
        n_dev = int(round(n * r_dev))
        if n_train + n_dev > n:
            n_dev = max(0, n - n_train)
        assigned_splits[perm[:n_train]] = SPLITS[0]
        assigned_splits[perm[n_train : n_train + n_dev]] = SPLITS[1]
        assigned_splits[perm[n_train + n_dev :]] = SPLITS[2]

    if stratify_by_orientation:
        orientations = compute_orientation(
            df["width"].to_numpy(dtype=float),
            df["height"].to_numpy(dtype=float),
        )
        for orient in sorted(np.unique(orientations)):
            _assign_group(np.where(orientations == orient)[0])
    else:
        _assign_group(np.arange(len(df)))

    df["split"] = assigned_splits
    return df
