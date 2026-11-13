# Infographix

An end-to-end AI-powered platform for converting academic papers and technical documents into presentation-ready scientific posters and infographics.

---

## 🏗️ Architecture & Core Components

1. **COCO & Poster Document Schemas (`src/schemas/coco.py`, `src/schemas/layout_document.py`)**:
   - Canonical layout interchange format connecting PDF document extraction, template libraries, and the frontend visual canvas editor.
   - Full HTML, SVG, and PDF rendering support (`src/rendering/`).
2. **DBSCAN Clustering & Multi-Table Feature Engineering (`src/clustering/dbscan_clustering.py`, `src/features/`, `src/models/layout_retrieval.py`)**:
   - **Macro-level**: Discovers coherent layout archetypes (DBSCAN + PCA-15 / MMR-diversified template retrieval) without forcing outliers into arbitrary clusters.
   - **Micro-level**: 2D anisotropic spatial DBSCAN groups scattered text fragments into columns and pairs figures with captions.
   - Multi-table feature extraction for posters, elements, and spatial relations (`data/processed/features/`).
3. **Template-Conditioned Neural Layout Model (`src/neural_layout/`)**:
   - **`ContentEncoder`**: Set Transformer encoding variable content elements ($M \times 18$).
   - **`TemplateEncoder`**: Set Transformer encoding retrieved conditioning template slots ($K \times 14$).
   - **`LayoutDecoder`**: Autoregressive decoder with coordinate regression and overlap/boundary penalties.
4. **ClearML Model Registry (`src/registry/clearml_registry.py` & `src/registry/model_registry.py`)**:
   - **ClearML Native Model Registry**: Production artifact versioning, SHA-256 checksum verification, hyperparameter & metric tracking, and lifecycle stage management (`production`, `staging`, `candidate`, `archived`).
   - **Local Dual-Mirror Fallback**: Zero-crash offline guarantee if ClearML credentials or network are unavailable.
5. **Experiment Tracking & CI/CD**:
   - Experiment tracking powered by **ClearML** and **Aim**.
   - Automated GitHub Actions workflows for continuous integration (`model-ci.yml`), training & registration (`model-train-register.yml`), and production promotion to Render (`model-cd-promote.yml`).

---

## 🚀 Quickstart: Using the ClearML Production Model Registry

### 1. Configure ClearML (One-time setup)
If using ClearML Community Cloud (`https://app.clear.ml`):
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
```

### 3. Load Neural Layout Generator from ClearML
```python
from src.registry.clearml_registry import load_registered_model

# Load candidate or production neural layout model by ClearML ID or semantic version
model, manifest = load_registered_model(
    model_id="YOUR_CLEARML_MODEL_ID",
    device="cpu",
    return_manifest=True,
)
```

### 4. Run Unit & Integration Tests
```bash
python -m unittest src/test_clearml_registry.py -v
python src/test_model_registry.py
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

## 🛠️ Setup and Environment

Operating System: Ubuntu 24.04 (On Windows use WSL2 or native PowerShell).

### 1. Python Environment Setup
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and activate it
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Custom DVC / MyDVC Setup Notes
Dataset versions and file manifests are tracked under `.mydvc/manifests/` (e.g. `.mydvc/manifests/v0.manifest`, `v1.manifest`, `v2.manifest`), keeping Git repository size minimal while tracking full SHA-256 integrity for datasets in Google Drive.
