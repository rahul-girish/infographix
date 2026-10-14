
"""COCO-style object-detection evaluation for Infographix."""

import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_detections(
    ground_truth_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Evaluate detection predictions against COCO-format ground truth.

    Ground truth must contain images, annotations, and categories.
    Predictions must be a JSON list with image_id, category_id,
    bbox [x, y, width, height], and score for each detection.
    """
    ground_truth_path = Path(ground_truth_path)
    predictions_path = Path(predictions_path)

    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {ground_truth_path}"
        )

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}"
        )

    with ground_truth_path.open(encoding="utf-8") as file:
        ground_truth = json.load(file)

    required = {"images", "annotations", "categories"}
    if not isinstance(ground_truth, dict) or not required.issubset(
        ground_truth
    ):
        raise ValueError(
            "Ground truth must contain images, annotations, and categories."
        )

    with predictions_path.open(encoding="utf-8") as file:
        predictions = json.load(file)

    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a JSON list.")

    image_ids = {image["id"] for image in ground_truth["images"]}
    category_ids = {
        category["id"] for category in ground_truth["categories"]
    }

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(f"Prediction {index} must be an object.")

        required_prediction_fields = {
            "image_id", "category_id", "bbox", "score"
        }
        if not required_prediction_fields.issubset(prediction):
            raise ValueError(
                f"Prediction {index} is missing required fields."
            )

        if prediction["image_id"] not in image_ids:
            raise ValueError(
                f"Prediction {index} refers to an unknown image."
            )

        if prediction["category_id"] not in category_ids:
            raise ValueError(
                f"Prediction {index} refers to an unknown category."
            )

        bbox = prediction["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, (int, float)) for value in bbox)
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            raise ValueError(
                f"Prediction {index} has an invalid bounding box."
            )

        score = prediction["score"]
        if (
            not isinstance(score, (int, float))
            or not 0 <= score <= 1
        ):
            raise ValueError(
                f"Prediction {index} score must be between 0 and 1."
            )

    if not predictions:
        raise ValueError(
            "Predictions are empty; COCO evaluation requires detections."
        )

    coco_gt = COCO(str(ground_truth_path))
    coco_dt = coco_gt.loadRes(str(predictions_path))

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metric_names = [
        "mAP_50_95",
        "mAP_50",
        "mAP_75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
    ]

    metrics = {
        name: float(value)
        for name, value in zip(metric_names, evaluator.stats)
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    return metrics


def log_evaluation_to_mlflow(
    metrics: dict[str, float],
    report_path: str | Path,
    run_name: str = "baseline-evaluation",
) -> None:
    """Log evaluation metrics and a JSON report to MLflow."""
    import mlflow

    report_path = Path(report_path)

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Evaluation report not found: {report_path}"
        )

    mlflow.set_experiment("infographix-baseline")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("evaluation_type", "object_detection")
        mlflow.log_param("metric_source", "COCOeval")

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))

        print(f"Logged {len(metrics)} metrics to MLflow.")
        print(f"Logged report: {report_path}")
