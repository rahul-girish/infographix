"""Generate turnkey Google Colab GPU training notebook: notebooks/04_neural_layout_model.ipynb."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/04_neural_layout_model.ipynb")


def make_cell(cell_type: str, source: list[str]) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [s + "\n" for s in source],
        "outputs": [] if cell_type == "code" else None,
        "execution_count": None if cell_type == "code" else None,
    }


def build_notebook():
    cells = []

    # Title & Metadata
    cells.append(make_cell("markdown", [
        "# Infographix: Retrieval-Augmented Neural Poster Layout Generator",
        "### Stage B: Neural Architecture, Multi-Task Losses, and Training (Turnkey Colab Notebook)",
        "",
        "This notebook trains a **Template-Conditioned Set Transformer** to arrange user-provided content blocks into visually professional scientific posters.",
        "",
        "#### Key Architecture Highlights:",
        "1. **ContentEncoder**: Permutation-equivariant Set Transformer (Multi-Head Self-Attention) encoding variable content demands ($M$ items) with explicit missingness flags.",
        "2. **TemplateEncoder**: Set Transformer encoding retrieved conditioning template slots ($K$ items) and normalized coordinates.",
        "3. **LayoutDecoder**: Cross-Attention decoder where content items query template slots, followed by self-attention across content blocks to resolve placement and interactions.",
        "4. **Multi-Task Objective**: SmoothL1 + GIoU bounding-box regression, hard boundary violation penalty, pairwise unintended overlap penalty, column classification, intrinsic figure aspect ratio preservation, and continuous reading order ranking.",
        "5. **Dual Baseline Validation**: Predictions are evaluated against both **Ground-Truth layouts** and the **Retrieved Conditioning Templates** to demonstrate active structural adaptation.",
    ]))

    # Cell 1: Environment Setup & Colab / Local Detection
    cells.append(make_cell("code", [
        "# 1. Environment Setup & Device Verification",
        "import sys",
        "from pathlib import Path",
        "import torch",
        "",
        "# Colab support: Clone repo or locate root directory",
        "IN_COLAB = 'google.colab' in sys.modules",
        "if IN_COLAB:",
        "    print('Running on Google Colab GPU environment.')",
        "    !pip install -q aim pyarrow pandas scikit-learn matplotlib scipy",
        "    # If cloned in Colab, cd to repository root",
        "    if not Path('src').exists():",
        "        !git clone https://github.com/rahul/infographix.git",
        "        %cd infographix",
        "else:",
        "    print('Running in local environment.')",
        "",
        "PROJECT_ROOT = Path('.').resolve()",
        "if str(PROJECT_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(PROJECT_ROOT))",
        "",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "print(f'Active device: {device}')",
        "if torch.cuda.is_available():",
        "    print(f'GPU Device: {torch.cuda.get_device_name(0)}')",
        "    print(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')",
    ]))

    # Cell 2: Imports
    cells.append(make_cell("code", [
        "# 2. Module Imports",
        "import math",
        "import random",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from torch.utils.data import DataLoader",
        "",
        "from src.models.layout_retrieval import LayoutRetriever",
        "from src.neural_layout.dataset import PosterLayoutDataset, collate_layout_batch, KNOWN_CATEGORIES",
        "from src.neural_layout.layout_model import TemplateConditionedLayoutModel",
        "from src.neural_layout.losses import MultiTaskLayoutLoss, LossWeights",
        "from src.neural_layout.evaluator import LayoutEvaluator",
        "from src.neural_layout.trainer import LayoutTrainer, TrainingConfig, set_seed",
        "from src.neural_layout.visualize import render_tri_layout_comparison",
        "from src.registry.model_registry import ModelRegistry",
        "",
        "set_seed(42)",
        "print('All modules imported successfully.')",
    ]))

    # Cell 3: Data Loading & Split Preparation
    cells.append(make_cell("code", [
        "# 3. Data Loading & Retriever Fitting",
        "features_dir = PROJECT_ROOT / 'data' / 'processed' / 'features' / 'v2'",
        "posters_df = pd.read_parquet(features_dir / 'posters.parquet')",
        "elements_df = pd.read_parquet(features_dir / 'elements.parquet')",
        "pdf_dir = PROJECT_ROOT / 'data' / 'interim' / 'scipostlayout' / 'poster' / 'pdf'",
        "",
        "print(f'Total Posters: {len(posters_df):,}')",
        "print(f'Total Valid Elements: {len(elements_df):,}')",
        "",
        "# Load production LayoutRetriever as candidate template generator",
        "reg = ModelRegistry(registry_root=PROJECT_ROOT / 'models' / 'registry')",
        "retriever_model, _, _ = reg.load_production_model('poster_layout_retriever')",
        "train_posters = posters_df[posters_df['split'] == 'train'].copy().reset_index(drop=True)",
        "retriever_model.fit(train_posters)",
        "print('LayoutRetriever fitted on train split.')",
    ]))

    # Cell 4: Dataset Construction
    cells.append(make_cell("code", [
        "# 4. PyTorch Dataset Construction",
        "# In Colab GPU, max_samples can be set to None for full dataset training",
        "MAX_TRAIN_SAMPLES = None if torch.cuda.is_available() else 100",
        "MAX_DEV_SAMPLES = None if torch.cuda.is_available() else 25",
        "",
        "train_dataset = PosterLayoutDataset(",
        "    posters_df=posters_df,",
        "    elements_df=elements_df,",
        "    retriever_model=retriever_model,",
        "    split='train',",
        "    pdf_base_dir=pdf_dir,",
        "    max_samples=MAX_TRAIN_SAMPLES,",
        ")",
        "",
        "dev_dataset = PosterLayoutDataset(",
        "    posters_df=posters_df,",
        "    elements_df=elements_df,",
        "    retriever_model=retriever_model,",
        "    split='dev',",
        "    pdf_base_dir=pdf_dir,",
        "    max_samples=MAX_DEV_SAMPLES,",
        ")",
        "",
        "print(f'Train Dataset: {len(train_dataset):,} samples')",
        "print(f'Dev Dataset: {len(dev_dataset):,} samples')",
    ]))

    # Cell 5: Inspect Batch Tensor Shapes
    cells.append(make_cell("code", [
        "# 5. Verify Batch Collation and Padding Masks",
        "sample_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_layout_batch)",
        "batch = next(iter(sample_loader))",
        "",
        "print('Batch Tensors:')",
        "for k, v in batch.items():",
        "    if isinstance(v, torch.Tensor):",
        "        print(f'  {k:<20}: shape={list(v.shape)}, dtype={v.dtype}')",
    ]))

    # Cell 6: Model Architecture & Parameter Count
    cells.append(make_cell("code", [
        "# 6. Initialize Template-Conditioned Layout Model",
        "config = TrainingConfig(",
        "    seed=42,",
        "    batch_size=16 if torch.cuda.is_available() else 8,",
        "    learning_rate=3e-4,",
        "    weight_decay=1e-4,",
        "    num_epochs=10 if torch.cuda.is_available() else 5,",
        "    grad_clip_norm=1.0,",
        "    d_model=128,",
        "    num_heads=4,",
        "    encoder_layers=3,",
        "    decoder_layers=3,",
        "    ffn_dim=256,",
        "    dropout=0.1,",
        "    loss_weights=LossWeights(",
        "        w_box=1.0,",
        "        w_giou=1.5,",
        "        w_bound=2.5,   # Hard boundary constraint",
        "        w_overlap=2.0, # Hard collision prevention",
        "        w_col=1.0,",
        "        w_aspect=1.2,",
        "        w_order=0.8,",
        "        w_macro=0.5,",
        "    ),",
        "    checkpoint_dir=PROJECT_ROOT / 'models' / 'checkpoints' / 'neural_layout',",
        "    aim_repo=str(PROJECT_ROOT / '.aim'),",
        "    experiment_name='colab_neural_layout_training',",
        "    device=device,",
        ")",
        "",
        "model = TemplateConditionedLayoutModel(",
        "    content_dim=18,",
        "    template_dim=16,",
        "    d_model=config.d_model,",
        "    num_heads=config.num_heads,",
        "    encoder_layers=config.encoder_layers,",
        "    decoder_layers=config.decoder_layers,",
        "    ffn_dim=config.ffn_dim,",
        "    num_columns=5,",
        "    dropout=config.dropout,",
        ")",
        "",
        "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "print(f'Initialized TemplateConditionedLayoutModel with {total_params:,} parameters on {device}.')",
    ]))

    # Cell 7: Training Run with Aim Logging
    cells.append(make_cell("code", [
        "# 7. Execute Training Loop with Aim Experiment Tracking",
        "trainer = LayoutTrainer(",
        "    model=model,",
        "    config=config,",
        "    train_dataset=train_dataset,",
        "    val_dataset=dev_dataset,",
        ")",
        "",
        "history = trainer.train()",
    ]))

    # Cell 8: Plot Training and Validation Loss Curves
    cells.append(make_cell("code", [
        "# 8. Training and Validation Performance Curves",
        "epochs = history['epoch']",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
        "",
        "# Loss Curve",
        "axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='#1f77b4')",
        "axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', color='#ff7f0e')",
        "axes[0].set_title('Multi-Task Loss Progression', fontweight='bold')",
        "axes[0].set_xlabel('Epoch')",
        "axes[0].set_ylabel('Total Loss')",
        "axes[0].grid(True, linestyle='--', alpha=0.5)",
        "axes[0].legend()",
        "",
        "# IoU and GIoU",
        "axes[1].plot(epochs, history['val_iou'], label='Val IoU', marker='^', color='#2ca02c')",
        "axes[1].plot(epochs, history['val_giou'], label='Val GIoU', marker='v', color='#9467bd')",
        "axes[1].set_title('Validation Bounding-Box Alignment', fontweight='bold')",
        "axes[1].set_xlabel('Epoch')",
        "axes[1].set_ylabel('Metric Value')",
        "axes[1].grid(True, linestyle='--', alpha=0.5)",
        "axes[1].legend()",
        "",
        "# Constraints: Boundary Violations & Overlaps",
        "axes[2].plot(epochs, history['val_boundary_violations'], label='Boundary Violation Rate', marker='x', color='#d62728')",
        "axes[2].plot(epochs, history['val_unintended_overlaps'], label='Unintended Overlap Rate', marker='d', color='#8c564b')",
        "axes[2].set_title('Hard Geometric Constraint Enforcement', fontweight='bold')",
        "axes[2].set_xlabel('Epoch')",
        "axes[2].set_ylabel('Rate')",
        "axes[2].grid(True, linestyle='--', alpha=0.5)",
        "axes[2].legend()",
        "",
        "plt.tight_layout()",
        "plt.show()",
    ]))

    # Cell 9: Inference & Dual Baseline Visual Evaluation
    cells.append(make_cell("code", [
        "# 9. Dual Baseline Visual Comparisons: Ground Truth vs. Template vs. Neural Model",
        "best_ckpt = config.checkpoint_dir / 'best_model.pt'",
        "if best_ckpt.exists():",
        "    trainer.load_checkpoint(best_ckpt)",
        "    print(f'Loaded best checkpoint from {best_ckpt}')",
        "",
        "model.eval()",
        "with torch.no_grad():",
        "    for test_idx in range(min(3, len(dev_dataset))):",
        "        sample_item = dev_dataset[test_idx]",
        "        c_feats = sample_item['content_feats'].unsqueeze(0).to(device)",
        "        t_feats = sample_item['template_feats'].unsqueeze(0).to(device)",
        "        c_mask = torch.ones(1, c_feats.shape[1], dtype=torch.bool, device=device)",
        "        t_mask = torch.ones(1, t_feats.shape[1], dtype=torch.bool, device=device)",
        "",
        "        preds = model(c_feats, t_feats, c_mask, t_mask)",
        "",
        "        pred_b = preds['pred_boxes'][0].cpu().numpy()",
        "        pred_ord_scores = preds['pred_order_scores'][0].cpu().numpy()",
        "        pred_ranks = np.argsort(np.argsort(pred_ord_scores))",
        "",
        "        target_b = sample_item['target_boxes'].numpy()",
        "        target_orders = sample_item['target_orders'].numpy()",
        "",
        "        tmpl_b = sample_item['template_feats'][:, 9:13].numpy()",
        "        tmpl_cats_idx = torch.argmax(sample_item['template_feats'][:, :9], dim=-1).numpy()",
        "        tmpl_cats = [KNOWN_CATEGORIES[idx] for idx in tmpl_cats_idx]",
        "",
        "        content_cats_idx = torch.argmax(sample_item['content_feats'][:, :9], dim=-1).numpy()",
        "        content_cats = [KNOWN_CATEGORIES[idx] for idx in content_cats_idx]",
        "",
        "        pid = int(sample_item['target_poster_id'].item())",
        "        tmpl_pid = int(sample_item['template_poster_id'].item())",
        "",
        "        fig = render_tri_layout_comparison(",
        "            target_boxes=target_b,",
        "            target_categories=content_cats,",
        "            target_orders=target_orders.tolist(),",
        "            template_boxes=tmpl_b,",
        "            template_categories=tmpl_cats,",
        "            pred_boxes=pred_b,",
        "            pred_categories=content_cats,",
        "            pred_orders=pred_ranks.tolist(),",
        "            poster_id=pid,",
        "            template_id=tmpl_pid,",
        "            title_suffix=f'(Validation Sample #{test_idx + 1})',",
        "        )",
        "        plt.show()",
    ]))

    # Cell 10: Colab GPU Export & Instructions
    cells.append(make_cell("markdown", [
        "## Summary & Production Model Registration",
        "",
        "Once trained on Colab GPU, the model weights can be downloaded or synced:",
        "```python",
        "# Export checkpoint to Google Drive or local machine",
        "from google.colab import files",
        "files.download('models/checkpoints/neural_layout/best_model.pt')",
        "```",
        "",
        "The model is now ready for Stage C (Inference & Poster Document Rendering) and Stage D (Feedback Adaptation).",
    ]))

    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)

    print(f"Generated {NOTEBOOK_PATH} successfully.")


if __name__ == "__main__":
    build_notebook()
