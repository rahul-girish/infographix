"""Unit tests for ClearML Model Registry integration."""

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.dummy import DummyClassifier

from src.registry.clearml_registry import ClearMLModelRegistry, is_clearml_configured


def test_clearml_registry_offline_lifecycle():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # 1. Initialize ClearML registry in forced offline mode
        registry = ClearMLModelRegistry(
            project_name="Infographix-Tests",
            offline=True,
            local_fallback_root=temp_dir,
        )

        dummy = DummyClassifier(strategy="constant", constant=0)
        dummy.fit(np.zeros((10, 4)), np.zeros(10))

        # 2. Register model
        res = registry.register_model(
            model_name="test_clearml_clusterer",
            version="v1.0.0",
            model_object=dummy,
            preprocessor=None,
            feature_schema_columns=["f1", "f2", "f3", "f4"],
            hyperparameters={"eps": 4.8},
            metrics={"train_silhouette": 0.24},
            stage="production",
            description="ClearML Test Clusterer",
            tags=["test", "unit-test"],
        )

        assert res["model_name"] == "test_clearml_clusterer"
        assert res["version"] == "v1.0.0"
        assert res["stage"] == "production"
        assert Path(res["local_path"]).exists()

        # 3. Load production model
        model, prep, meta = registry.load_production_model("test_clearml_clusterer")
        assert model is not None
        pred = model.predict(np.zeros((2, 4)))
        assert list(pred) == [0, 0]

        # 4. List models
        models = registry.list_models()
        assert "test_clearml_clusterer" in models

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_clearml_registry_offline_lifecycle()
    print("ALL CLEARML REGISTRY UNIT TESTS PASSED!")
