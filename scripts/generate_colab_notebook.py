"""Generate turnkey Google Colab GPU training notebook: notebooks/04_neural_layout_model.ipynb with ClearML."""

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
        "### Stage B: Neural Architecture, Multi-Task Losses, and Training (ClearML Cloud Integration)",
        "",
        "This notebook trains a **Template-Conditioned Set Transformer** to arrange user-provided content blocks into visually professional scientific posters.",
        "Experiment tracking, multi-task losses, evaluation metrics, and layout visualizations are streamed live to **ClearML Cloud** under the `Infographix` project.",
        "",
        "#### Key Architecture Highlights:",
        "1. **ContentEncoder**: Permutation-equivariant Set Transformer (Multi-Head Self-Attention) encoding variable content demands ($M$ items) with explicit missingness flags.",
        "2. **TemplateEncoder**: Set Transformer encoding retrieved conditioning template slots ($K$ items) and normalized coordinates.",
        "3. **LayoutDecoder**: Cross-Attention decoder where content items query template slots, followed by self-attention across content blocks to resolve placement and interactions.",
        "4. **Multi-Task Objective**: SmoothL1 + GIoU bounding-box regression, hard boundary violation penalty, pairwise unintended overlap penalty, column classification, intrinsic figure aspect ratio preservation, and continuous reading order ranking.",
        "5. **ClearML Tracking**: Live streaming of all 8 loss components, IoU/GIoU, boundary violation rate, unintended overlap rate, column accuracy, reading order Kendall Tau, and 3-panel visual layout comparisons.",
    ]))

    # Cell 1: Environment Setup & ClearML Authentication
    cells.append(make_cell("code", [
        "# 1. Environment Setup & Secure ClearML Authentication",
        "import os",
        "import sys",
        "from pathlib import Path",
        "import torch",
        "",
        "IN_COLAB = 'google.colab' in sys.modules",
        "if IN_COLAB:",
        "    print('Running on Google Colab environment.')",
        "    !pip install -q clearml pyarrow pandas scikit-learn matplotlib scipy",
        "    if not Path('src').exists():",
        "        !git clone https://github.com/rahul/infographix.git",
        "        %cd infographix",
        "",
        "    # Authenticate through Google Colab Secrets (No hardcoded credentials)",
        "    try:",
        "        from google.colab import userdata",
        "        api_access_key = userdata.get('CLEARML_API_ACCESS_KEY')",
        "        api_secret_key = userdata.get('CLEARML_API_SECRET_KEY')",
        "        if api_access_key and api_secret_key:",
        "            os.environ['CLEARML_API_ACCESS_KEY'] = api_access_key",
        "            os.environ['CLEARML_API_SECRET_KEY'] = api_secret_key",
        "            os.environ['CLEARML_API_HOST'] = userdata.get('CLEARML_API_HOST') or 'https://api.clear.ml'",
        "            os.environ['CLEARML_WEB_HOST'] = userdata.get('CLEARML_WEB_HOST') or 'https://app.clear.ml'",
        "            os.environ['CLEARML_FILES_HOST'] = userdata.get('CLEARML_FILES_HOST') or 'https://files.clear.ml'",
        "            print('Successfully authenticated ClearML via Colab Secrets.')",
        "    except Exception as e:",
        "        print(f'Note: Colab Secrets check: {e}. ClearML will use local config or interactive prompt if needed.')",
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

    # Cell 2: ClearML Task Init at Entry Point
    cells.append(make_cell("code", [
        "# 2. Initialize ClearML Task at Entry Point",
        "from clearml import Task",
        "",
        "execution_env = 'colab_gpu' if (IN_COLAB and torch.cuda.is_available()) else ('colab_cpu' if IN_COLAB else 'local')",
        "tags = ['neural_layout', 'stage_b', 'gpu' if torch.cuda.is_available() else 'cpu', 'colab' if IN_COLAB else 'local']",
        "",
        "task = Task.init(",
        "    project_name='Infographix',",
        "    task_name=f'Neural Layout Training - {execution_env.upper()}',",
        "    task_type=Task.TaskTypes.training,",
        "    reuse_last_task_id=False,",
        "    tags=tags,",
        "    auto_connect_frameworks=False,",
        ")",
        "clearml_logger = task.get_logger()",
        "print(f'ClearML Task initialized: {task.id}')",
        "print(f'Results page: {task.get_output_log_web_page()}')",
    ]))

    # Cell 3: Imports
    cells.append(make_cell("code", [
        "# 3. Module Imports (Single Source of Truth)",
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
        "from src.neural_layout.visualize import render_tri_layout_comparison, render_training_curves",
        "from src.registry.model_registry import ModelRegistry",
        "",
        "set_seed(42)",
        "print('All modules imported successfully.')",
    ]))

    # Cell 4: Data Loading & Split Preparation
    cells.append(make_cell("code", [
        "# 4. Data Loading & Retriever Fitting",
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

    # Cell 5: Dataset Construction
    cells.append(make_cell("code", [
        "# 5. PyTorch Dataset Construction",
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

    # Cell 6: Model Architecture & Parameter Count
    cells.append(make_cell("code", [
        "# 6. Initialize Template-Conditioned Layout Model & Training Config",
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
        "    project_name='Infographix',",
        "    task_name=f'Neural Layout Training - {execution_env.upper()}',",
        "    tags=tags,",
        "    execution_env=execution_env,",
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

    # Cell 7: Training Run with ClearML Logging
    cells.append(make_cell("code", [
        "# 7. Execute Training Loop with ClearML Tracking",
        "trainer = LayoutTrainer(",
        "    model=model,",
        "    config=config,",
        "    train_dataset=train_dataset,",
        "    val_dataset=dev_dataset,",
        "    clearml_task=task,",
        ")",
        "",
        "history = trainer.train()",
    ]))

    # Cell 8: Plot Training and Validation Loss Curves & Upload to ClearML
    cells.append(make_cell("code", [
        "# 8. Training and Validation Performance Curves & ClearML Reporting",
        "fig_curves = render_training_curves(history)",
        "clearml_logger.report_matplotlib_figure(",
        "    title='Training Curves',",
        "    series='Multi-Task Loss and Alignment',",
        "    figure=fig_curves,",
        "    iteration=config.num_epochs,",
        ")",
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
        "        fig_sample = render_tri_layout_comparison(",
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
        "",
        "        clearml_logger.report_matplotlib_figure(",
        "            title='Validation Layout Visualizations',",
        "            series=f'Sample_{test_idx + 1}_PID_{pid}',",
        "            figure=fig_sample,",
        "            iteration=config.num_epochs,",
        "        )",
        "        plt.show()",
        "",
        "# Final ClearML sync",
        "clearml_logger.flush()",
        "task.flush()",
        "task.close()",
        "print('ClearML task successfully synced and completed.')",
    ]))

    # Cell 10: Colab GPU Export & Instructions
    cells.append(make_cell("markdown", [
        "## Summary & Checkpoint Management",
        "",
        "Once trained on Colab GPU, local checkpoints (`models/checkpoints/neural_layout/best_model.pt`) can be exported:",
        "```python",
        "from google.colab import files",
        "files.download('models/checkpoints/neural_layout/best_model.pt')",
        "```",
        "",
        "The ClearML Task page provides complete auditability of hyperparameters, scalars, and visual plots.",
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

    print(f"Generated {NOTEBOOK_PATH} with ClearML integration successfully.")


if __name__ == "__main__":
    build_notebook()
