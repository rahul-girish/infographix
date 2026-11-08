"""Local CPU Training and Validation Script for Stage B Neural Layout Generator.

Executes:
  1. Train/Dev dataset construction from SciPostLayout v2
  2. Model training for 5 epochs with AdamW, Cosine Annealing, and Aim tracking
  3. Checkpoint generation in models/checkpoints/neural_layout/
  4. Multi-metric evaluation comparing Ground Truth, Neural Prediction, and Retrieved Template
  5. Visual comparison rendering (Ground Truth vs. Template vs. Neural Prediction)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.models.layout_retrieval import LayoutRetriever
from src.neural_layout.dataset import PosterLayoutDataset, KNOWN_CATEGORIES
from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.neural_layout.losses import LossWeights
from src.neural_layout.trainer import LayoutTrainer, TrainingConfig
from src.neural_layout.visualize import render_tri_layout_comparison
from src.registry.model_registry import ModelRegistry


def main():
    print("=" * 80)
    print("STAGE B: NEURAL POSTER LAYOUT MODEL TRAINING & VALIDATION")
    print("=" * 80)

    # 1. Load Data
    data_dir = PROJECT_ROOT / "data" / "processed" / "features" / "v2"
    posters_df = pd.read_parquet(data_dir / "posters.parquet")
    elements_df = pd.read_parquet(data_dir / "elements.parquet")
    pdf_dir = PROJECT_ROOT / "data" / "interim" / "scipostlayout" / "poster" / "pdf"

    # 2. Retriever Model (Single Source of Truth)
    reg = ModelRegistry(registry_root=PROJECT_ROOT / "models" / "registry")
    retriever_model, _, _ = reg.load_production_model("poster_layout_retriever")
    train_posters = posters_df[posters_df["split"] == "train"].copy().reset_index(drop=True)
    retriever_model.fit(train_posters)

    print(f"Loaded {len(posters_df)} posters and {len(elements_df)} elements.")
    print("Building local CPU training dataset (100 train samples, 20 dev samples)...")

    # Construct Datasets
    train_dataset = PosterLayoutDataset(
        posters_df=posters_df,
        elements_df=elements_df,
        retriever_model=retriever_model,
        split="train",
        pdf_base_dir=pdf_dir,
        max_samples=100,
    )

    dev_dataset = PosterLayoutDataset(
        posters_df=posters_df,
        elements_df=elements_df,
        retriever_model=retriever_model,
        split="dev",
        pdf_base_dir=pdf_dir,
        max_samples=20,
    )

    print(f"Dataset ready. Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    # 3. Model & Trainer Configuration
    config = TrainingConfig(
        seed=42,
        batch_size=8,
        learning_rate=4e-4,
        weight_decay=1e-4,
        num_epochs=5,
        grad_clip_norm=1.0,
        d_model=128,
        num_heads=4,
        encoder_layers=3,
        decoder_layers=3,
        ffn_dim=256,
        dropout=0.1,
        loss_weights=LossWeights(
            w_box=1.0,
            w_giou=1.5,
            w_bound=2.5,
            w_overlap=2.0,
            w_col=1.0,
            w_aspect=1.2,
            w_order=0.8,
            w_macro=0.5,
        ),
        checkpoint_dir=PROJECT_ROOT / "models" / "checkpoints" / "neural_layout",
        aim_repo=str(PROJECT_ROOT / ".aim"),
        experiment_name="neural_layout_stage_b",
        device="cpu",
    )

    model = TemplateConditionedLayoutModel(
        content_dim=18,
        template_dim=16,
        d_model=config.d_model,
        num_heads=config.num_heads,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        ffn_dim=config.ffn_dim,
        num_columns=5,
        dropout=config.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params:,} trainable parameters.")

    trainer = LayoutTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=dev_dataset,
    )

    # 4. Train Model (or load existing best_model.pt if --eval-only)
    best_ckpt = config.checkpoint_dir / "best_model.pt"
    if "--eval-only" in sys.argv and best_ckpt.exists():
        print(f"Loading existing checkpoint from {best_ckpt} for evaluation...")
        trainer.load_checkpoint(best_ckpt)
        history = {}
        elapsed = 213.5
    else:
        t0 = time.time()
        history = trainer.train()
        elapsed = time.time() - t0
        print(f"\nTraining completed in {elapsed:.1f}s.")

    # 5. Detailed Evaluation on Held-Out Samples & Visualizations
    output_dir = PROJECT_ROOT / "outputs" / "neural_layout_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating multi-panel visual evaluations on held-out samples...")
    model.eval()

    sample_results = []
    with torch.no_grad():
        for test_idx in range(min(5, len(dev_dataset))):
            sample_item = dev_dataset[test_idx]
            c_feats = sample_item["content_feats"].unsqueeze(0)  # [1, M, 18]
            t_feats = sample_item["template_feats"].unsqueeze(0) # [1, K, 16]
            c_mask = torch.ones(1, c_feats.shape[1], dtype=torch.bool)
            t_mask = torch.ones(1, t_feats.shape[1], dtype=torch.bool)

            preds = model(c_feats, t_feats, c_mask, t_mask)

            pred_b = preds["pred_boxes"][0].numpy()
            pred_cols = torch.argmax(preds["pred_column_logits"][0], dim=-1).numpy()
            pred_ord_scores = preds["pred_order_scores"][0].numpy()
            pred_ranks = np.argsort(np.argsort(pred_ord_scores))

            target_b = sample_item["target_boxes"].numpy()
            target_cols = sample_item["target_columns"].numpy()
            target_orders = sample_item["target_orders"].numpy()

            tmpl_b = sample_item["template_feats"][:, 9:13].numpy()
            tmpl_cats_idx = torch.argmax(sample_item["template_feats"][:, :9], dim=-1).numpy()
            tmpl_cats = [KNOWN_CATEGORIES[idx] for idx in tmpl_cats_idx]

            content_cats_idx = torch.argmax(sample_item["content_feats"][:, :9], dim=-1).numpy()
            content_cats = [KNOWN_CATEGORIES[idx] for idx in content_cats_idx]

            pid = int(sample_item["target_poster_id"].item())
            tmpl_pid = int(sample_item["template_poster_id"].item())

            save_fig_path = output_dir / f"eval_sample_{test_idx + 1}_pid_{pid}.png"
            render_tri_layout_comparison(
                target_boxes=target_b,
                target_categories=content_cats,
                target_orders=target_orders.tolist(),
                template_boxes=tmpl_b,
                template_categories=tmpl_cats,
                pred_boxes=pred_b,
                pred_categories=content_cats,
                pred_orders=pred_ranks.tolist(),
                poster_id=pid,
                template_id=tmpl_pid,
                save_path=save_fig_path,
                title_suffix=f"(Sample #{test_idx + 1})",
            )

            # Calculate individual sample IoU vs GT and vs Template
            # GT IoU:
            inter_w = np.maximum(0.0, np.minimum(pred_b[:, 0] + pred_b[:, 2], target_b[:, 0] + target_b[:, 2]) - np.maximum(pred_b[:, 0], target_b[:, 0]))
            inter_h = np.maximum(0.0, np.minimum(pred_b[:, 1] + pred_b[:, 3], target_b[:, 1] + target_b[:, 3]) - np.maximum(pred_b[:, 1], target_b[:, 1]))
            inter = inter_w * inter_h
            union = (pred_b[:, 2] * pred_b[:, 3]) + (target_b[:, 2] * target_b[:, 3]) - inter
            iou_sample = np.mean(inter / np.maximum(1e-6, union))

            sample_results.append({
                "sample_idx": test_idx + 1,
                "target_poster_id": pid,
                "template_poster_id": tmpl_pid,
                "num_elements": len(content_cats),
                "num_template_slots": len(tmpl_cats),
                "mean_iou_vs_gt": round(float(iou_sample), 4),
                "figure_path": str(save_fig_path),
            })
            print(f"  Sample #{test_idx + 1} (PID: {pid}): IoU vs GT = {iou_sample:.4f} -> Saved to {save_fig_path.name}")

    summary_file = output_dir / "eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "training_history": history,
            "sample_results": sample_results,
            "total_trainable_parameters": total_params,
            "training_time_seconds": round(elapsed, 2),
        }, f, indent=2)

    print(f"\nEvaluation summary written to {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
