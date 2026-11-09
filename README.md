# Infographix 📊

> **AI-Powered Technical Poster and Infographic Generator**  
> An automated platform for students, researchers, and academics to transform research papers into customizable, publication-grade technical posters.

---

## 🏗️ Architecture & Core Components

1. **COCO Data Schema (`src/schemas/coco.py`)**:
   - Universal data interchange format connecting PDF document extraction, template libraries, and the frontend visual canvas editor.
2. **DBSCAN Layout Clustering (`src/clustering/dbscan_clustering.py`)**:
   - **Macro-level**: PCA (15 components) + DBSCAN discovers coherent layout archetypes and isolates irregular designs as noise (`-1`).
   - **Micro-level**: 2D anisotropic spatial DBSCAN groups scattered text fragments into columns and pairs figures with captions.
3. **Experiment Tracking**:
   - Tracked using **Aim** (with local JSON fallback via `src/tracking/tracker.py`).
4. **Model Registry (`src/registry/clearml_registry.py` & `src/registry/model_registry.py`)**:
   - **ClearML Model Registry**: Production artifact versioning, hyperparameter & metric tracking, and stage management (`production`, `staging`, `archived`) with remote cloud storage and web UI.
   - **Local File-based Fallback**: Zero-crash offline guarantee if ClearML credentials or network are unavailable.
   - Currently registered in production: `poster_layout_clusterer` (`v1.0.0` / `v1.1.0-clearml`).

---

## 🚀 Quickstart: Using the ClearML Production Model Registry

### 1. Configure ClearML (One-time setup)
If using the free ClearML Community Cloud (`https://app.clear.ml`):
```bash
# Interactive setup: paste your access key & secret from app.clear.ml/settings
clearml-init
```
Or set environment variables in your `.env` or shell:
```bash
export CLEARML_API_ACCESS_KEY="your_access_key"
export CLEARML_API_SECRET_KEY="your_secret_key"
```
*(If no credentials are provided, ClearML runs in offline mode automatically with zero errors).*

### 2. Load the Active Production Model in FastAPI / Python
```python
from src.registry.clearml_registry import ClearMLModelRegistry

registry = ClearMLModelRegistry(project_name="Infographix")

# Load active production model & preprocessor (fetches from ClearML or local cache)
model, preprocessor, meta = registry.load_production_model("poster_layout_clusterer")

print(f"Loaded: {meta.get('name', 'poster_layout_clusterer')}")
print(f"Source: {meta.get('source')}")

# Run inference on extracted paper features (92-column vector)
scaled_features = preprocessor.transform(raw_features)
cluster_id = model.predict(scaled_features)
# cluster_id = 0, 1... (assigned archetype) or -1 (noise/outlier)
```

### 3. Register / Publish a Model to ClearML
```bash
python src/registry/register_clearml_production_model.py
```

### 4. Inspect Registered Models
```python
from src.registry.clearml_registry import ClearMLModelRegistry

registry = ClearMLModelRegistry()
print(registry.list_models())
```

### 5. Run Live End-to-End Demo
```bash
python demo_live_pipeline.py
```

### 6. Run Unit Tests
```bash
python -c "
import src.test_coco_dbscan as t1
import src.test_model_registry as t2
import src.test_clearml_registry as t3
t1.test_coco_data_models_and_validation()
t1.test_convert_extracted_document_to_coco()
t1.test_canvas_to_coco_conversion()
t1.test_poster_layout_dbscan_clustering()
t1.test_spatial_block_dbscan_and_pairing()
t2.test_model_registry_lifecycle()
t3.test_clearml_registry_offline_lifecycle()
print('All tests passed successfully!')
"
```

---

## Setup and Environment

Operating System: Ubuntu 24.04 (On Windows use WSL2).

Run the following command in powershell as an administrator:
```bash
wsl --install -d Ubuntu-24.04
```

Install the following in your Ubuntu environment:

Install python
```bash
sudo apt install python3.12
```

Install `uv`. `uv` is a package manager for python
```bash
curl LsSf https://astral.sh/uv/0.13.0/install.sh | sh
```

Create a python virtual environment and activate it.
**Always make sure your venv is activated when you are running code.**
```bash
uv venv
source .venv/bin/activate
```

Install HuggingFace CLI
```bash
uv pip install huggingface_hub
```

Install DVC
```bash
uv pip install dvc dvc-gdrive
```

Create a new directory and download the dataset. You will need at least 33 GB of free space.
```bash
mkdir data/raw/scipostlayout
hf download omron-sinicx/scipostlayout_v2 --repo-type dataset --local-dir data/raw/scipostlayout
```

Unzip the dataset. You will need another 33 GB of space
```bash
mkdir data/interim
unzip data/raw/scipostlayout/scipostlayout.zip -d data/interim/scipostlayout
```

Install requirements.
```bash
uv pip install -r requirements.txt
```

Register the virtual environment as a jupyter kernel.
```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=infographix-venv  
```

Start jupyter lab in a separate terminal **outside the venv**.
```bash
jupyter lab
```
Go to the url shown in the terminal, or go to http://localhost:8888/lab.
Open a notebook, set the kernel to infographix-venv, and run something to test that it's working.

**Before every commit, make sure to update requirements.txt**
**Make sure you are in the activated virtual environment before running this:**
```bash
uv pip freeze | Out-File -Encoding utf8 requirements.txt
```

---

## Custom DVC / MyDVC Setup Notes
Dataset versions and file manifests are tracked under `.mydvc/manifests/`.
