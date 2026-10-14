
import json
import tempfile
import unittest
from pathlib import Path

from evaluation import evaluate_detections, log_evaluation_to_mlflow


class TestEvaluation(unittest.TestCase):
    def test_perfect_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            ground_truth_path = temp / "ground_truth.json"
            predictions_path = temp / "predictions.json"
            output_path = temp / "metrics.json"

            ground_truth = {
                "images": [
                    {"id": 1, "width": 100, "height": 100,
                     "file_name": "poster.png"}
                ],
                "categories": [{"id": 1, "name": "title"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            }

            predictions = [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "score": 0.99,
                }
            ]

            ground_truth_path.write_text(
                json.dumps(ground_truth), encoding="utf-8"
            )
            predictions_path.write_text(
                json.dumps(predictions), encoding="utf-8"
            )

            metrics = evaluate_detections(
                ground_truth_path,
                predictions_path,
                output_path,
            )

            self.assertAlmostEqual(metrics["mAP_50_95"], 1.0)
            self.assertAlmostEqual(metrics["mAP_50"], 1.0)
            self.assertTrue(output_path.exists())

            saved_metrics = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(saved_metrics, metrics)

    def test_mlflow_logging(self):
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            report_path = temp / "metrics.json"
            report_path.write_text(
                json.dumps({"mAP_50_95": 1.0}),
                encoding="utf-8",
            )

            tracking_db = temp / "mlflow.db"
            mlflow.set_tracking_uri(
                f"sqlite:///{tracking_db.as_posix()}"
            )

            try:
                log_evaluation_to_mlflow(
                    metrics={"mAP_50_95": 1.0, "mAP_50": 1.0},
                    report_path=report_path,
                    run_name="integration-test",
                )

                client = MlflowClient()
                experiment = client.get_experiment_by_name(
                    "infographix-baseline"
                )
                self.assertIsNotNone(experiment)

                runs = client.search_runs([experiment.experiment_id])
                self.assertEqual(len(runs), 1)
                self.assertEqual(
                    runs[0].data.metrics["mAP_50_95"], 1.0
                )
                self.assertEqual(
                    runs[0].data.params["evaluation_type"],
                    "object_detection",
                )
            finally:
                # Close the SQLite connection pool before temp cleanup.
                store = mlflow.tracking._tracking_service.utils._get_store(
                    mlflow.get_tracking_uri()
                )
                if isinstance(store, SqlAlchemyStore):
                    store.engine.dispose()


    def test_unknown_image_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            ground_truth_path = temp / "ground_truth.json"
            predictions_path = temp / "predictions.json"

            ground_truth = {
                "images": [
                    {"id": 1, "width": 100, "height": 100,
                     "file_name": "poster.png"}
                ],
                "categories": [{"id": 1, "name": "title"}],
                "annotations": [],
            }

            predictions = [
                {
                    "image_id": 999,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "score": 0.9,
                }
            ]

            ground_truth_path.write_text(
                json.dumps(ground_truth), encoding="utf-8"
            )
            predictions_path.write_text(
                json.dumps(predictions), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "unknown image"):
                evaluate_detections(ground_truth_path, predictions_path)

    def test_unknown_category_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            ground_truth_path = temp / "ground_truth.json"
            predictions_path = temp / "predictions.json"

            ground_truth = {
                "images": [
                    {"id": 1, "width": 100, "height": 100,
                     "file_name": "poster.png"}
                ],
                "categories": [{"id": 1, "name": "title"}],
                "annotations": [],
            }

            predictions = [
                {
                    "image_id": 1,
                    "category_id": 999,
                    "bbox": [10, 10, 20, 20],
                    "score": 0.9,
                }
            ]

            ground_truth_path.write_text(
                json.dumps(ground_truth), encoding="utf-8"
            )
            predictions_path.write_text(
                json.dumps(predictions), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "unknown category"):
                evaluate_detections(ground_truth_path, predictions_path)

    def test_invalid_bounding_box_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            ground_truth_path = temp / "ground_truth.json"
            predictions_path = temp / "predictions.json"

            ground_truth = {
                "images": [
                    {"id": 1, "width": 100, "height": 100,
                     "file_name": "poster.png"}
                ],
                "categories": [{"id": 1, "name": "title"}],
                "annotations": [],
            }

            predictions = [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 0, 20],
                    "score": 0.9,
                }
            ]

            ground_truth_path.write_text(
                json.dumps(ground_truth), encoding="utf-8"
            )
            predictions_path.write_text(
                json.dumps(predictions), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "invalid bounding box"):
                evaluate_detections(ground_truth_path, predictions_path)


if __name__ == "__main__":
    unittest.main()
