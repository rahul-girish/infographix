# Infographix: Neural Layout Generation, ClearML Tracking & Native Model Registry

This document covers the Stage A/B Neural Layout Generation pipeline, ClearML Cloud experiment tracking integration, and ClearML Native Model Registry specification for downstream consumers (inference, evaluation, and deployment).

---

## 1. Architecture Overview

Infographix implements a **Retrieval-Augmented Neural Layout Generation** system trained on **SciPostLayout v2** (7,855 posters, 192,657 valid elements, 676,383 spatial relations):

1. **Candidate Template Retrieval ([`src/models/layout_retrieval.py`](../src/models/layout_retrieval.py))**:
   - Retrieves top-$K$ structurally compatible layouts (`poster_layout_retriever` v2.0.0) using MMR diversity while strictly excluding the target poster ID (`exclude_poster_id=target_pid`) during training.
2. **Anti-Leakage Dataset Construction ([`src/neural_layout/dataset.py`](../src/neural_layout/dataset.py))**:
   - **Content Features (18 dims, `CONTENT_FEAT_DIM = 18`)**: 9-dim category one-hot (`KNOWN_CATEGORIES`), `log1p(word_count)`, `log1p(char_count)`, explicit `has_text_length` flag, `aspect_ratio`, explicit `has_aspect_ratio` flag, normalized `table_rows` / `table_cols`, `priority`, and deterministically shuffled `shuffled_order_idx / M`.
   - **Zero Spatial Leakage**: Text word/character counts come exclusively from `/usr/bin/pdftotext -bbox-layout` spatial matching; unavailable text is explicitly tagged `has_text_length=False`, `text_source="unavailable"`. Figure aspect ratios derive exclusively from `/usr/bin/pdfimages -list` intrinsic image dimensions; missing assets are tagged `has_aspect_ratio=False`, `aspect_source="unavailable"`. Input ordering is decoupled from target reading order via deterministic permutation.
   - **Template Features (16 dims, `TEMPLATE_FEAT_DIM = 16`)**: 9-dim category one-hot + normalized `[x0, y0, w, h, cx, cy, area_norm]`.
3. **Neural Model ([`src/neural_layout/layout_model.py`](../src/neural_layout/layout_model.py))**:
   - **`ContentEncoder`** ([`content_encoder.py`](../src/neural_layout/content_encoder.py)): 3-layer Set Transformer (Multi-Head Self-Attention) with padding masks and global macro pooling (`PMA`).
   - **`TemplateEncoder`** ([`template_encoder.py`](../src/neural_layout/template_encoder.py)): 3-layer Set Transformer encoding variable retrieved template slots.
   - **`LayoutDecoder`** ([`layout_decoder.py`](../src/neural_layout/layout_decoder.py)): 3-layer cross-attention (`Q=content, K,V=template`) + content self-attention predicting:
     - `pred_boxes`: `[B, M, 4]` normalized `[x0, y0, w, h]` in `(0, 1)`
     - `pred_column_logits`: `[B, M, 5]`
     - `pred_order_scores`: `[B, M]` continuous ranking score
     - `pred_macro_column_logits`: `[B, 5]`
   - Default parameter count (`d_model=128`, `num_heads=4`, `encoder_layers=3`, `decoder_layers=3`, `ffn_dim=256`): **1,513,871** trainable parameters.
4. **Multi-Task Loss ([`src/neural_layout/losses.py`](../src/neural_layout/losses.py))**:
   - `SmoothL1` box regression (`w_box=1.0`), `GIoU` (`w_giou=1.5`), canvas boundary penalty (`w_bound=2.5`), selective pairwise overlap penalty ignoring figure-caption pairs (`w_overlap=2.0`), column cross-entropy (`w_col=1.0`), masked intrinsic aspect-ratio log error (`w_aspect=1.2`), pairwise reading-order ranking (`w_order=0.8`), and macro column cross-entropy (`w_macro=0.5`).

---

## 2. ClearML Experiment Tracking

Active training experiment tracking runs on **ClearML Cloud** under the shared `Infographix` project ([`src/neural_layout/trainer.py`](../src/neural_layout/trainer.py)):

- **Entry-Point Initialization**: Both [`scripts/train_neural_layout_local.py`](../scripts/train_neural_layout_local.py) and [`notebooks/04_neural_layout_model.ipynb`](../notebooks/04_neural_layout_model.ipynb) call `Task.init(project_name="Infographix", reuse_last_task_id=False, auto_connect_frameworks=False, ...)` prior to pipeline initialization. (`auto_connect_frameworks=False` disables automatic checkpoint hijacking so registration stays explicit.)
- **Captured Configuration (`Experiment Configuration`)**:
  - `model/*`, `training/*`, `loss_weights/*`, `dataset/*`, and `environment/*` (git commit hash, branch, dirty status).
- **Consistent Epoch Metrics**:
  - `Loss - Total` (`train` / `val`)
  - All 8 individual loss titles: `Loss - Box Regression`, `Loss - GIoU`, `Loss - Boundary`, `Loss - Overlap`, `Loss - Column Assignment`, `Loss - Aspect Ratio`, `Loss - Reading Order`, `Loss - Macro Column`
  - Evaluation titles: `Metrics - Alignment` (`val_iou_mean`, `val_giou_mean`), `Metrics - Geometric Constraints` (`val_canvas_violation_rate`, `val_unintended_overlap_rate`), `Metrics - Structure` (`val_column_accuracy`, `val_reading_order_kendall_tau`), `Metrics - Template Baseline` (`val_template_divergence_iou`, `val_template_baseline_gt_iou`), and `Hyperparameters` (`learning_rate`).
- **Visualizations**: Training loss/metric curves (`Training Curves`) and 3-panel sample comparisons (`Validation Layout Visualizations`: Ground Truth vs. Conditioning Template vs. Neural Prediction).
- **Verified Cloud Validation Tasks**:
  - Local CPU Validation: Task ID `8cd6cb755a084535a7bba3ae551e3e20`
  - Colab Smoke Test: Task ID `2a669cc3b9624247abf14def42d36590`

---

## 3. ClearML Native Model Registry

Model registration and retrieval are implemented in [`src/registry/clearml_registry.py`](../src/registry/clearml_registry.py) using ClearML's native `OutputModel`, `InputModel`, and `Model.query_models` APIs.

> **Deprecation Notice:** Local [`ModelRegistry.register_model`](../src/registry/model_registry.py) has been retired from active registration (emitting a `DeprecationWarning`). Read-only loading in `ModelRegistry.load_production_model` remains temporarily for legacy v2 scikit-learn models (`poster_layout_retriever`, `poster_layout_archetypes`) used during candidate template retrieval.

### 3.1 Naming, Tags & Lifecycle Convention

- **Model Family**: `poster_layout_generator`
- **Versioned Name**: `poster_layout_generator-vX.Y.Z` (e.g., `poster_layout_generator-v0.1.0`) — immutable; attempting to re-register an existing semantic version raises `ValueError`.
- **Tags**: `family:poster_layout_generator`, `version:vX.Y.Z`, `arch:TemplateConditionedLayoutModel`, `dataset:<identifier>`, `stage:<lifecycle_status>`.
- **Lifecycle Stages**: `candidate`, `staging`, `production`, `archived`. Transitions are explicit via `set_model_lifecycle_status(...)` and kept strictly separate from ClearML's native `Published` flag.

### 3.2 Versioned Model Bundle & Manifest

Each registered model is packaged as a ZIP bundle on ClearML's files server containing:
1. `model_weights.pt`: Inference-only `model_state_dict` (optimizer and scheduler states stripped).
2. `model_config.json`: Exact [`TemplateConditionedLayoutModel`](../src/neural_layout/layout_model.py) hyperparameters.
3. `feature_schema.json`: Input/output tensor schemas (`schema_version = "v1.0.0"`).
4. `preprocessing_config.json`: Anti-leakage defaults and normalization constants.
5. `environment.json`: Python, PyTorch, ClearML, NumPy versions, platform, and git commit.
6. `manifest.json`: Canonical identity card including `project_name`, `model_family`, `semantic_version`, `clearml_training_task_id`, `dataset_identifier`, `git_commit`, `checkpoint_sha256`, `bundle_sha256s` (SHA-256 for every bundle file), `created_at`, `lifecycle_status`, `parent_model_id`, `training_config`, and `metrics`.

### 3.3 Registered Development Candidate (`v0.1.0`)

Registered and verified via [`scripts/register_and_verify_clearml_model.py`](../scripts/register_and_verify_clearml_model.py):

- **Model Name**: `poster_layout_generator-v0.1.0`
- **ClearML Model ID**: `6a2c2d4897384aaf8aa59c7c66a86bfb`
- **Originating ClearML Task**: `8cd6cb755a084535a7bba3ae551e3e20` (provenance verified via `config_and_metric_match`)
- **Lifecycle Status**: `candidate` (`published=False`)
- **Checkpoint SHA-256**: `510cdbba983471f1e41ecabc3270a60bc8f234b10e33824c58833197203281a7`
- **Reproducibility Verification**: Retrieved remotely with local checkpoints hidden (`force_download=True`); predictions on held-out dev Poster ID `194056279` matched original checkpoint outputs with `max_abs_diff = 0.00e+00` across all 4 heads, and tampered-bundle checksum verification was confirmed.

### 3.4 Downstream Retrieval Example (Handover Usage)

Any authenticated teammate or downstream inference service can reconstruct a specific version without access to the original training directory:

```python
from src.registry.clearml_registry import (
    load_registered_model,
    query_registered_models,
    register_model,
    set_model_lifecycle_status,
)

# 1. Inspect available models in Infographix ClearML workspace
models = query_registered_models(semantic_version="0.1.0")
print(models)

# 2. Download bundle, verify SHA-256 checksums, and reconstruct PyTorch model
model, manifest = load_registered_model(
    semantic_version="0.1.0",   # Or pass exact model_id="6a2c2d4897384aaf8aa59c7c66a86bfb"
    device="cpu",
    return_manifest=True,
)
# model is ready in .eval() mode with model.registry_manifest & model.feature_schema attached
```
