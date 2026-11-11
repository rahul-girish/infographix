# Infographix 📊

> **AI-Powered Scientific Poster Generation System**  
> An automated ML platform for researchers, students, and academics to arrange user-provided content (text, figures, tables, captions, and author metadata) into structured, publication-grade scientific posters built on **SciPostLayout v2** (7,855 posters, 192,657 valid elements, 676,383 spatial relations).

---

## 🏗️ Architecture & Core Components

1. **COCO & Poster Document Schemas (`src/schemas/coco.py`, `src/schemas/layout_document.py`)**:
   - Canonical layout interchange format supporting HTML, SVG, and PDF rendering (`src/rendering/`).
2. **Multi-Table Feature Engineering & Candidate Retrieval (`src/features/`, `src/models/layout_retrieval.py`)**:
   - Extracts poster, element, and spatial relation tables (`data/processed/features/v2/`).
   - Macro-level layout archetype clustering (`poster_layout_archetypes` v2.0.0) and MMR-diversified nearest-neighbor template retrieval (`poster_layout_retriever` v2.0.0).
3. **Anti-Leakage Neural Training Dataset (`src/neural_layout/dataset.py`)**:
   - Extracts genuine PDF text via `/usr/bin/pdftotext -bbox-layout` and intrinsic image aspect ratios via `/usr/bin/pdfimages -list`.
   - Explicitly represents missing metadata (`has_text_length=False`, `has_aspect_ratio=False`) without bounding-box area estimation, deterministically decouples input order from reading order, and excludes `target_poster_id` during retrieval.
4. **Template-Conditioned Neural Layout Model (`src/neural_layout/`)**:
   - **`ContentEncoder`**: Set Transformer encoding variable content elements ($M \times 18$).
   - **`TemplateEncoder`**: Set Transformer encoding retrieved conditioning template slots ($K \times 16$).
   - **`LayoutDecoder`**: Cross-attention + inter-element self-attention predicting normalized bounding boxes, column assignments, continuous reading-order scores, and macro column counts.
   - **Multi-Task Objective (`src/neural_layout/losses.py`)**: SmoothL1 + GIoU regression, hard canvas-boundary and selective overlap penalties, column cross-entropy, masked intrinsic aspect-ratio preservation, reading-order ranking, and macro column planning.
5. **Experiment Tracking — ClearML Cloud (`src/neural_layout/trainer.py`)**:
   - Tracks all 8 loss components, validation alignment/constraint metrics, git provenance, hyperparameters, and 3-panel layout visualizations in the shared `Infographix` ClearML workspace.
   - Local CPU training (`scripts/train_neural_layout_local.py`) and turnkey Google Colab GPU notebook (`notebooks/04_neural_layout_model.ipynb`).
6. **Model Registry — ClearML Native Registry (`src/registry/clearml_registry.py`)**:
   - Versioned, immutable model bundles (`OutputModel`, `InputModel`, `Model.query_models`) containing weights-only PyTorch checkpoints, architecture configs, feature/preprocessing schemas, runtime metadata, and SHA-256 verified manifests.
   - Lifecycle tags (`candidate`, `staging`, `production`, `archived`) managed independently of ClearML `Published` flags.
   - See [`docs/NEURAL_LAYOUT_AND_CLEARML_REGISTRY.md`](docs/NEURAL_LAYOUT_AND_CLEARML_REGISTRY.md) for full technical specifications.

---

## 🚀 Quickstart: Training & ClearML Model Registry

### 1. ClearML Authentication
Configure ClearML Cloud locally once:
```bash
.venv/bin/clearml-init
```
On Google Colab, store `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` in **Colab Secrets**; [`notebooks/04_neural_layout_model.ipynb`](notebooks/04_neural_layout_model.ipynb) authenticates automatically without hardcoded credentials.

### 2. Run Training (Local CPU or Colab GPU)
```bash
# Short local CPU validation run reported to ClearML Cloud
.venv/bin/python -u scripts/train_neural_layout_local.py --epochs 3 --train-samples 16 --val-samples 8
```
Local checkpoints are written to `models/checkpoints/neural_layout/` (`best_model.pt` and `latest_checkpoint.pt`).

### 3. Register a Model Version in ClearML
```python
from src.registry.clearml_registry import register_model

model_id = register_model(
    checkpoint_path="models/checkpoints/neural_layout/best_model.pt",
    training_task_id="8cd6cb755a084535a7bba3ae551e3e20",
    semantic_version="0.1.0",
    dataset_identifier="SciPostLayout v2.0.0",
    lifecycle_status="candidate",
)
```
Or run the full end-to-end registration and reproducibility verification script:
```bash
.venv/bin/python -u scripts/register_and_verify_clearml_model.py
```

### 4. Query & Load a Registered Model (Inference / Downstream Services)
```python
from src.registry.clearml_registry import (
    load_registered_model,
    query_registered_models,
    set_model_lifecycle_status,
)

# List registered models in Infographix ClearML workspace
models = query_registered_models(semantic_version="0.1.0")
print(models)

# Download verified bundle from ClearML Cloud and reconstruct PyTorch model
model, manifest = load_registered_model(
    semantic_version="0.1.0",   # or exact ClearML model_id="6a2c2d4897384aaf8aa59c7c66a86bfb"
    device="cpu",
    return_manifest=True,
)
# Returns TemplateConditionedLayoutModel in .eval() mode with SHA-256 verified weights
```

### 5. Run Unit & Integration Tests
```bash
.venv/bin/python -m unittest \
  src/test_dataset_stage_a.py \
  src/test_neural_model_stage_b.py \
  src/test_clearml_registry.py \
  src/test_v2_pipeline.py \
  src/test_stage3_pipeline.py
.venv/bin/python src/test_model_registry.py
```

---

## 📚 Additional Documentation

- [`docs/NEURAL_LAYOUT_AND_CLEARML_REGISTRY.md`](docs/NEURAL_LAYOUT_AND_CLEARML_REGISTRY.md) — Stage A/B Neural Layout pipeline, ClearML Cloud tracking, and ClearML Native Model Registry handover guide.
- [`docs/FEATURE_ENGINEERING_AND_MODELING_V2.md`](docs/FEATURE_ENGINEERING_AND_MODELING_V2.md) — Multi-table feature engineering & v2 retrieval pipeline.
- [`docs/STAGE3_ADAPTATION_AND_GENERATION.md`](docs/STAGE3_ADAPTATION_AND_GENERATION.md) — Layout adaptation benchmarks & rendering formats.
- [`docs/BACKEND_ARCHITECTURE.md`](docs/BACKEND_ARCHITECTURE.md) & [`docs/BACKEND_API_SPECIFICATION.md`](docs/BACKEND_API_SPECIFICATION.md) — Streamlined FastAPI inference & MLOps telemetry service (`POST /api/v1/generate`, `/runs`, `/export/pdf`).
- [`docs/DATABASE_DESIGN.md`](docs/DATABASE_DESIGN.md) — Supabase PostgreSQL `generation_runs` schema for storing raw inputs, model outputs, and human edits.
- [`docs/FRONTEND_ARCHITECTURE.md`](docs/FRONTEND_ARCHITECTURE.md) & [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md) — Minimal React + Konva input/editor UI, PNG/PDF export, and Supabase deployment guide.

### Running the Web App (Model Testing & Data Collection)
```bash
# 1. Start FastAPI backend (loads LayoutRetriever + LayoutAdapter / Neural model)
PYTHONPATH=. .venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Start Vite + React frontend
cd frontend && npm run dev
```

---

## Setup and Environment

Operating System: Ubuntu 24.04 (On Windows use WSL2).

Run the following command in PowerShell as an administrator:
```bash
wsl --install -d Ubuntu-24.04
```

Install the following in your Ubuntu environment:

Install Python and Poppler PDF utilities (`pdftotext`, `pdfimages`):
```bash
sudo apt install python3.12 poppler-utils
```

Install `uv` (`uv` is the Python package manager used in this project):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a Python virtual environment and activate it.
**Always make sure your venv is activated when running code.**
```bash
uv venv
source .venv/bin/activate
```

Install dependencies:
```bash
uv pip install -r requirements.txt
```

Download and extract the SciPostLayout v2 dataset (requires ~66 GB total free space):
```bash
uv pip install huggingface_hub
mkdir -p data/raw/scipostlayout data/interim
hf download omron-sinicx/scipostlayout_v2 --repo-type dataset --local-dir data/raw/scipostlayout
unzip data/raw/scipostlayout/scipostlayout.zip -d data/interim/scipostlayout
```

Register the virtual environment as a Jupyter kernel:
```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=infographix-venv
```

Start JupyterLab in a separate terminal **outside the venv**:
```bash
jupyter lab
```
Open a notebook, set the kernel to `infographix-venv`, and execute cells.

**Before every commit, verify `requirements.txt` is up to date inside the activated `.venv`:**
```bash
uv pip freeze > requirements.txt
```

---

## DVC Setup Notes

Note that all DVC commands should be run from within the virtual environment.

Initialize DVC:
```bash
dvc init
```

Configure Google Drive Remote:
```bash
dvc remote add --default gdrive gdrive://folder_id
dvc remote list
```

Configure Google Drive OAuth client credentials:
```bash
dvc remote modify gdrive --local gdrive_client_id 'client_id'
dvc remote modify gdrive --local gdrive_client_secret 'client_secret'
```
