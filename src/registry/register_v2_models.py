"""Fit, evaluate, and register the Infographix LayoutRetriever in ClearML Model Registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from clearml import Task

from src.models.layout_retrieval import (
    LayoutRetriever,
    evaluate_retrieval_comprehensive,
)
from src.registry.clearml_registry import register_retriever_model


def _resolve_features_dir() -> Path:
    for candidate in ("v2", "v1"):
        p = PROJECT_ROOT / "data" / "processed" / "features" / candidate
        if (p / "posters.parquet").exists():
            return p
    raise FileNotFoundError("Could not find posters.parquet under data/processed/features/{v2,v1}")


def register_layout_retriever(
    semantic_version: str = "2.1.0",
    lifecycle_status: str = "candidate",
    dataset_identifier: str = "SciPostLayout v2 (70/15/15)",
) -> str:
    """Fit LayoutRetriever strictly on train split, evaluate on test, and register in ClearML."""
    features_dir = _resolve_features_dir()
    posters_df = pd.read_parquet(features_dir / "posters.parquet")
    elements_df = pd.read_parquet(features_dir / "elements.parquet")

    train_p = posters_df[posters_df["split"] == "train"].reset_index(drop=True)
    test_p = posters_df[posters_df["split"] == "test"].reset_index(drop=True)
    test_e = elements_df[elements_df["split"] == "test"].reset_index(drop=True)

    task = Task.init(
        project_name="Infographix",
        task_name=f"Register LayoutRetriever v{semantic_version}",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        tags=["layout_retriever", f"version:v{semantic_version}"],
        auto_connect_frameworks=False,
    )

    try:
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

        logger = task.get_logger()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                logger.report_scalar("Retrieval Evaluation", k, float(v), iteration=0)

        model_id = register_retriever_model(
            retriever_model=retriever,
            training_task_id=str(task.id),
            semantic_version=semantic_version,
            dataset_identifier=dataset_identifier,
            validation_metrics=metrics,
            lifecycle_status=lifecycle_status,
            comment="Train-only LayoutRetriever registered directly to ClearML Native Model Registry.",
        )
        print(f"Registered poster_layout_retriever-v{semantic_version} in ClearML (ID: {model_id})")
        return model_id
    finally:
        task.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register LayoutRetriever in ClearML Model Registry.")
    parser.add_argument("--version", default="2.1.0", help="Semantic version (e.g. 2.1.0)")
    parser.add_argument("--stage", default="candidate", choices=["candidate", "staging", "production"])
    parser.add_argument("--dataset", default="SciPostLayout v2 (70/15/15)")
    args = parser.parse_args()
    register_layout_retriever(
        semantic_version=args.version,
        lifecycle_status=args.stage,
        dataset_identifier=args.dataset,
    )
