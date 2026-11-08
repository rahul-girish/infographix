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
4. **Model Registry (`src/registry/model_registry.py`)**:
   - Version-controlled, immutable model storage managing lifecycle stages (`development`, `staging`, `production`, `archived`).
   - Currently registered in production: `poster_layout_clusterer` (`v1.0.0`).

---

## 🚀 Quickstart: Using the Production Model Registry

### Load the Active Production Model in FastAPI / Python
```python
from src.registry.model_registry import ModelRegistry

registry = ModelRegistry()

# 1. Load active production model, preprocessor, and metadata manifest
model, preprocessor, manifest = registry.load_production_model("poster_layout_clusterer")

print(f"Loaded: {manifest.model_name} ({manifest.version})")
print(f"Stage: {manifest.stage}")
print(f"Metrics: Train Silhouette={manifest.metrics['train_silhouette']:.4f}")

# 2. Run inference on extracted paper features
# raw_features: 2D array matching feature_schema.json (92 columns)
scaled_features = preprocessor.transform(raw_features)
cluster_id = model.predict(scaled_features)
# cluster_id = 0, 1... (assigned archetype) or -1 (noise/outlier)
```

### Inspect the Registry
```python
from src.registry.model_registry import ModelRegistry

registry = ModelRegistry()
overview = registry.list_models()
print(overview)
```

### Register / Re-train a Production Model
```bash
python src/registry/register_production_model.py
```

### Run Benchmarks & Model Comparisons
```bash
python src/clustering/comparison.py
```
Outputs are written to `reports/comparisons/clustering_comparison.csv` and `reports/comparisons/clustering_comparison_report.md`.

### Run Unit Tests
```bash
python -c "
import src.test_coco_dbscan as t1
import src.test_model_registry as t2
t1.test_coco_data_models_and_validation()
t1.test_convert_extracted_document_to_coco()
t1.test_canvas_to_coco_conversion()
t1.test_poster_layout_dbscan_clustering()
t1.test_spatial_block_dbscan_and_pairing()
t2.test_model_registry_lifecycle()
print('All tests passed!')
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
uv pip freeze > requirements.txt
```

---

## DVC Setup Notes

Note that all DVC commands can only be run from within the virtual environment.

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
