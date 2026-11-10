"""Evaluation and registration script for the Infographix LayoutRetriever model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models.layout_retrieval import (
    LayoutRetriever,
    evaluate_retrieval_comprehensive,
)
from src.registry.model_registry import ModelRegistry


def _resolve_features_dir() -> Path:
    for candidate in ("v2", "v1"):
        p = PROJECT_ROOT / "data" / "processed" / "features" / candidate
        if (p / "posters.parquet").exists():
            return p
    raise FileNotFoundError("Could not find posters.parquet under data/processed/features/{v2,v1}")


def register_layout_retriever(version: str = "v2.0.0") -> None:
    """Fit, evaluate, and register the constraint-aware LayoutRetriever."""
    features_dir = _resolve_features_dir()
    posters_df = pd.read_parquet(features_dir / "posters.parquet")
    elements_df = pd.read_parquet(features_dir / "elements.parquet")

    train_p = posters_df[posters_df["split"] == "train"].reset_index(drop=True)
    test_p = posters_df[posters_df["split"] == "test"].reset_index(drop=True)
    test_e = elements_df[elements_df["split"] == "test"].reset_index(drop=True)

    retriever = LayoutRetriever(mmr_lambda=0.78, strict_orientation=True).fit(train_p)
    metrics = evaluate_retrieval_comprehensive(
        retriever,
        test_p,
        test_e,
        top_k=5,
        diversify=True,
        active_groups=("canvas", "inventory", "capacity", "structure"),
        random_state=42,
    )

    reg = ModelRegistry()
    reg.register_model(
        model_name="poster_layout_retriever",
        version=version,
        model_object=retriever,
        preprocessor=None,
        feature_schema_columns=list(retriever.group_weights.keys()),
        hyperparameters={
            "group_weights": retriever.group_weights,
            "mmr_lambda": retriever.mmr_lambda,
            "strict_orientation": retriever.strict_orientation,
        },
        metrics=metrics,
        stage="production",
        description="Direct constraint-aware, group-normalized nearest-neighbour layout retriever with MMR diversity reranking.",
    )
    print(f"Registered poster_layout_retriever {version} from {features_dir}")


if __name__ == "__main__":
    register_layout_retriever()
