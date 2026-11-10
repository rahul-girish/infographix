"""Visualizer for Neural Poster Layout Comparisons.

Renders side-by-side comparisons:
  Panel 1: Ground-Truth Layout
  Panel 2: Retrieved Conditioning Template
  Panel 3: Neural Model Generated Layout
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from src.neural_layout.dataset import KNOWN_CATEGORIES, CATEGORY_TO_IDX

CATEGORY_COLORS = {
    "title": "#1f77b4",        # Blue
    "author_info": "#aec7e8",  # Light blue
    "section": "#2ca02c",      # Green
    "text": "#7f7f7f",         # Slate/Gray
    "figure": "#ff7f0e",       # Orange
    "table": "#9467bd",        # Purple
    "caption": "#bcbd22",      # Olive/Amber
    "list": "#8c564b",         # Brown
    "logo": "#e377c2",         # Pink
}


def render_tri_layout_comparison(
    target_boxes: np.ndarray,
    target_categories: list[str],
    target_orders: list[int],
    template_boxes: np.ndarray,
    template_categories: list[str],
    pred_boxes: np.ndarray,
    pred_categories: list[str],
    pred_orders: list[int],
    poster_id: int | str,
    template_id: int | str,
    aspect_ratio: float = 0.71,
    save_path: Path | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """Render 3-panel side-by-side visual comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor("#f8f9fa")

    panels = [
        ("1. Ground-Truth Layout", target_boxes, target_categories, target_orders, axes[0], f"Poster #{poster_id}"),
        ("2. Retrieved Template (Conditioning)", template_boxes, template_categories, None, axes[1], f"Template #{template_id}"),
        ("3. Neural Model Prediction (Generated)", pred_boxes, pred_categories, pred_orders, axes[2], f"Generated Composition {title_suffix}"),
    ]

    for title, boxes, cats, orders, ax, subtitle in panels:
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)  # Invert Y so 0 is top
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=10)

        # Poster boundary
        outer_rect = patches.Rectangle(
            (0, 0), 1.0, 1.0,
            linewidth=2, edgecolor="#2b2b2b", facecolor="none", linestyle="-"
        )
        ax.add_patch(outer_rect)

        # Draw boxes
        for i, box in enumerate(boxes):
            x0, y0, w, h = box[:4]
            # Clip display to prevent canvas distortion
            x0_c = float(np.clip(x0, 0.0, 1.0))
            y0_c = float(np.clip(y0, 0.0, 1.0))
            w_c = float(np.clip(w, 0.01, 1.0 - x0_c))
            h_c = float(np.clip(h, 0.01, 1.0 - y0_c))

            cat = cats[i] if i < len(cats) else "text"
            color = CATEGORY_COLORS.get(cat, "#7f7f7f")

            rect = patches.Rectangle(
                (x0_c, y0_c), w_c, h_c,
                linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.35,
            )
            ax.add_patch(rect)

            border = patches.Rectangle(
                (x0_c, y0_c), w_c, h_c,
                linewidth=1.5, edgecolor=color, facecolor="none",
            )
            ax.add_patch(border)

            label = cat
            if orders is not None and i < len(orders):
                label = f"{orders[i]}:{cat[:4]}"
            else:
                label = cat[:5]

            # Label text
            if w_c > 0.05 and h_c > 0.02:
                ax.text(
                    x0_c + w_c / 2.0, y0_c + h_c / 2.0,
                    label,
                    ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#1a1a1a",
                    bbox=dict(boxstyle="square,pad=0.1", fc="#ffffff", ec="none", alpha=0.6)
                )

        ax.grid(True, linestyle="--", alpha=0.3, color="#cccccc")
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")

    # Legend at bottom
    legend_elements = [
        patches.Patch(facecolor=color, edgecolor="#333", alpha=0.6, label=cat)
        for cat, color in CATEGORY_COLORS.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(CATEGORY_COLORS),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig


def render_training_curves(
    history: dict[str, list[float]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Render 3-panel figure showing training loss progression, alignment, and constraints."""
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#f8f9fa")

    # Panel 1: Loss Curve
    axes[0].set_facecolor("#ffffff")
    axes[0].plot(epochs, history.get("train_loss", []), label="Train Loss", marker="o", color="#1f77b4", linewidth=2)
    if "val_loss" in history and any(history["val_loss"]):
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Multi-Task Loss Progression", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Panel 2: Alignment and Overlap
    axes[1].set_facecolor("#ffffff")
    if "val_iou" in history:
        axes[1].plot(epochs, history["val_iou"], label="Val IoU vs GT", marker="^", color="#2ca02c", linewidth=2)
    if "val_giou" in history:
        axes[1].plot(epochs, history["val_giou"], label="Val GIoU vs GT", marker="v", color="#9467bd", linewidth=2)
    axes[1].set_title("Geometric Alignment Metrics", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Panel 3: Constraints and Structural Properties
    axes[2].set_facecolor("#ffffff")
    if "val_boundary_violations" in history:
        axes[2].plot(epochs, history["val_boundary_violations"], label="Boundary Violations", marker="x", color="#d62728", linewidth=2)
    if "val_unintended_overlaps" in history:
        axes[2].plot(epochs, history["val_unintended_overlaps"], label="Unintended Overlap Rate", marker="d", color="#8c564b", linewidth=2)
    if "val_col_acc" in history:
        axes[2].plot(epochs, history["val_col_acc"], label="Column Accuracy", marker="+", color="#17becf", linewidth=2)
    axes[2].set_title("Constraints & Structural Accuracy", fontweight="bold", fontsize=12)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Rate / Accuracy")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig
