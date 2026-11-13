"""Inspect and validate synthetic training pair construction (Stage A)."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models.layout_retrieval import LayoutRetriever
from src.neural_layout.dataset import build_synthetic_request_and_sample
from src.registry.model_registry import ModelRegistry


def main():
    features_dir = PROJECT_ROOT / "data" / "processed" / "features" / "v2"
    posters_df = pd.read_parquet(features_dir / "posters.parquet")
    elements_df = pd.read_parquet(features_dir / "elements.parquet")

    reg = ModelRegistry(registry_root=PROJECT_ROOT / "models" / "registry")
    retriever_model, _, _ = reg.load_production_model("poster_layout_retriever")
    train_posters = posters_df[posters_df["split"] == "train"].copy().reset_index(drop=True)
    retriever_model.fit(train_posters)

    pdf_dir = PROJECT_ROOT / "data" / "interim" / "scipostlayout" / "poster" / "pdf"

    # Select 3 distinct posters across different archetype column counts
    sample_indices = [0, 10, 25]
    print("=" * 80)
    print("STAGE A: VALIDATION OF SYNTHETIC TRAINING REQUESTS & ANTI-LEAKAGE PROOF")
    print("=" * 80)

    for idx in sample_indices:
        target_row = train_posters.iloc[idx]
        sample = build_synthetic_request_and_sample(
            target_poster_row=target_row,
            target_elements_df=elements_df,
            retriever_model=retriever_model,
            pdf_base_dir=pdf_dir,
        )

        s_dict = sample.to_dict()
        print(f"\n[Sample #{idx + 1}] Target Poster ID: {sample.target_poster_id} (Split: {sample.split})")
        print(f"  Canvas Aspect Ratio: {sample.canvas_aspect_ratio:.2f} | Columns: {sample.target_column_count}")
        print(f"  Conditioning Template ID: {sample.template_poster_id} (Target Excluded: {sample.template_poster_id != sample.target_poster_id})")
        print(f"  Total Requested Elements: {len(sample.content_items)} | Template Slots: {len(sample.template_slots)}")
        print("\n  Simulated User Content Demands (Zero Spatial Leakage into inputs):")
        for i, item in enumerate(sample.content_items[:8]):
            src_tag = f"[{item.text_source}]"
            asp_str = f"aspect={item.aspect_ratio:.2f}" if item.category_slug in ("figure", "logo") else ""
            print(f"    {i+1}. {item.category_slug:<12} {src_tag:<24} words={int(item.word_count):<4} chars={int(item.char_count):<5} {asp_str}")
            if item.sample_text:
                print(f"       Text: \"{item.sample_text[:70]}...\"")

        print("\n  Ground-Truth Supervision Targets (What the Neural Model Learns to Predict):")
        for i in range(min(5, len(sample.target_boxes))):
            box = sample.target_boxes[i]
            col = sample.target_columns[i]
            order = sample.target_reading_orders[i]
            cat = sample.content_items[i].category_slug
            print(f"    Target #{i+1} ({cat}): box=[x0={box[0]:.3f}, y0={box[1]:.3f}, w={box[2]:.3f}, h={box[3]:.3f}] | col={col} | order={order}")

        print("-" * 80)


if __name__ == "__main__":
    main()
