"""Unit tests for Model Registry module."""

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.dummy import DummyClassifier

from src.registry.model_registry import ModelManifest, ModelRegistry


def test_model_registry_lifecycle():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        registry = ModelRegistry(registry_root=temp_dir)

        # 1. Register model
        dummy = DummyClassifier(strategy="constant", constant=1)
        dummy.fit(np.zeros((10, 2)), np.ones(10))

        manifest = registry.register_model(
            model_name="test_clusterer",
            version="1.0.0",
            model_object=dummy,
            preprocessor=None,
            feature_schema_columns=["feat_a", "feat_b"],
            hyperparameters={"param_1": 10},
            metrics={"silhouette": 0.35},
            stage="staging",
            description="Test clusterer",
        )

        assert manifest.version == "v1.0.0"
        assert manifest.stage == "staging"

        # 2. Immutability check: cannot overwrite existing version
        try:
            registry.register_model(
                model_name="test_clusterer",
                version="1.0.0",
                model_object=dummy,
            )
            assert False, "Should have raised FileExistsError"
        except FileExistsError:
            pass

        # 3. Load by version
        model, prep, loaded_manifest = registry.load_model("test_clusterer", version_or_stage="v1.0.0")
        assert model is not None
        assert loaded_manifest.version == "v1.0.0"
        pred = model.predict(np.zeros((2, 2)))
        assert list(pred) == [1, 1]

        # 4. Promote to production
        promoted = registry.promote_model("test_clusterer", version="v1.0.0", target_stage="production")
        assert promoted.stage == "production"

        # 5. Load production model
        prod_model, _, prod_manifest = registry.load_production_model("test_clusterer")
        assert prod_manifest.version == "v1.0.0"

        # 6. List models
        overview = registry.list_models()
        assert "test_clusterer" in overview
        assert overview["test_clusterer"]["active_stages"]["production"] == "v1.0.0"
        assert "v1.0.0" in overview["test_clusterer"]["versions"]

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_model_registry_lifecycle()
    print("ALL MODEL REGISTRY TESTS PASSED!")
