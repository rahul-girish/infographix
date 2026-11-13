"""Colab Integration Smoke Test for Stage B Neural Layout Generator with ClearML.

Verifies that executing the notebook's ClearML setup, dataset loading, model initialization,
2-epoch training, metric tracking, and visualization upload reports cleanly to the shared
'Infographix' ClearML project with Colab-style tags and configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from clearml import Task

from src.neural_layout.dataset import PosterLayoutDataset, KNOWN_CATEGORIES
from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.neural_layout.losses import LossWeights
from src.neural_layout.trainer import LayoutTrainer, TrainingConfig, set_seed
from src.neural_layout.visualize import render_tri_layout_comparison, render_training_curves
from src.registry.model_registry import ModelRegistry


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    execution_env = "colab_gpu" if torch.cuda.is_available() else "colab_smoke_test"
    tags = ["neural_layout", "stage_b", "colab", "smoke_test", device]

    # 1. Initialize ClearML Task at Entry Point
    task = Task.init(
        project_name="Infographix",
        task_name="Neural Layout Training - Colab Smoke Test",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        tags=tags,
        auto_connect_frameworks=False,
    )
    clearml_logger = task.get_logger()
    print(f"ClearML Colab Smoke Test Task initialized: {task.id}")
    print(f"Results page: {task.get_output_log_web_page()}")

    # 2. Load Data & Fit Retriever
    features_dir = PROJECT_ROOT / "data" / "processed" / "features" / "v2"
    posters_df = pd.read_parquet(features_dir / "posters.parquet")
    elements_df = pd.read_parquet(features_dir / "elements.parquet")
    pdf_dir = PROJECT_ROOT / "data" / "interim" / "scipostlayout" / "poster" / "pdf"

    reg = ModelRegistry(registry_root=PROJECT_ROOT / "models" / "registry")
    retriever_model, _, _ = reg.load_production_model("poster_layout_retriever")
    train_posters = posters_df[posters_df["split"] == "train"].copy().reset_index(drop=True)
    retriever_model.fit(train_posters)

    train_dataset = PosterLayoutDataset(
        posters_df=posters_df,
        elements_df=elements_df,
        retriever_model=retriever_model,
        split="train",
        pdf_base_dir=pdf_dir,
        max_samples=8,
    )
    dev_dataset = PosterLayoutDataset(
        posters_df=posters_df,
        elements_df=elements_df,
        retriever_model=retriever_model,
        split="dev",
        pdf_base_dir=pdf_dir,
        max_samples=4,
    )

    # 3. Configure and Train
    config = TrainingConfig(
        seed=42,
        batch_size=4,
        learning_rate=3e-4,
        weight_decay=1e-4,
        num_epochs=2,
        grad_clip_norm=1.0,
        d_model=128,
        num_heads=4,
        encoder_layers=3,
        decoder_layers=3,
        ffn_dim=256,
        dropout=0.1,
        loss_weights=LossWeights(),
        checkpoint_dir=PROJECT_ROOT / "models" / "checkpoints" / "colab_smoke_test",
        project_name="Infographix",
        task_name="Neural Layout Training - Colab Smoke Test",
        tags=tags,
        execution_env=execution_env,
        device=device,
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

    trainer = LayoutTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=dev_dataset,
        clearml_task=task,
    )
    history = trainer.train()

    # 4. Upload Visualizations
    fig_curves = render_training_curves(history)
    clearml_logger.report_matplotlib_figure(
        title="Training Curves",
        series="Multi-Task Loss and Alignment",
        figure=fig_curves,
        iteration=config.num_epochs,
    )
    plt.close(fig_curves)

    sample_item = dev_dataset[0]
    with torch.no_grad():
        c_feats = sample_item["content_feats"].unsqueeze(0).to(device)
        t_feats = sample_item["template_feats"].unsqueeze(0).to(device)
        c_mask = torch.ones(1, c_feats.shape[1], dtype=torch.bool, device=device)
        t_mask = torch.ones(1, t_feats.shape[1], dtype=torch.bool, device=device)
        preds = model(c_feats, t_feats, c_mask, t_mask)

    pred_b = preds["pred_boxes"][0].cpu().numpy()
    pred_ord_scores = preds["pred_order_scores"][0].cpu().numpy()
    pred_ranks = np.argsort(np.argsort(pred_ord_scores))
    target_b = sample_item["target_boxes"].numpy()
    target_orders = sample_item["target_orders"].numpy()
    tmpl_b = sample_item["template_feats"][:, 9:13].numpy()
    tmpl_cats = [KNOWN_CATEGORIES[idx] for idx in torch.argmax(sample_item["template_feats"][:, :9], dim=-1).numpy()]
    content_cats = [KNOWN_CATEGORIES[idx] for idx in torch.argmax(sample_item["content_feats"][:, :9], dim=-1).numpy()]
    pid = int(sample_item["target_poster_id"].item())
    tmpl_pid = int(sample_item["template_poster_id"].item())

    fig_sample = render_tri_layout_comparison(
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
        title_suffix="(Colab Smoke Test Sample)",
    )
    clearml_logger.report_matplotlib_figure(
        title="Validation Layout Visualizations",
        series=f"Sample_1_PID_{pid}",
        figure=fig_sample,
        iteration=config.num_epochs,
    )
    plt.close(fig_sample)

    # 5. Flush and Close
    clearml_logger.flush()
    task.flush()
    task.close()
    print(f"Colab smoke test completed. Task ID: {task.id}")


if __name__ == "__main__":
    main()
