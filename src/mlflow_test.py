
import json
from pathlib import Path

import mlflow

# Store experiment data locally in the project.
TRACKING_DIR = Path("mlruns").resolve()
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("infographix-baseline")

with mlflow.start_run(run_name="framework-smoke-test"):
    # Synthetic example values to test tracking only.
    mlflow.log_param("test_type", "synthetic_smoke_test")
    mlflow.log_param("model_name", "placeholder")
    mlflow.log_metric("example_score", 0.85)

    report = {
        "purpose": "Verify MLflow artifact logging",
        "is_real_model_evaluation": False,
    }

    report_path = Path("mlflow_test_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(report_path))

print("MLflow test run completed.")
print(f"Tracking data: {TRACKING_DIR}")
