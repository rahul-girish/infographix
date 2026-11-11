import base64
import io
import json
import unittest
import uuid
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from PIL import Image

from backend.tests.conftest import create_test_session_factory
from backend.app.api.v1.generation import (
    export_pdf,
    export_training_data,
    generate_poster,
    get_run,
    list_runs,
    update_run,
)
from backend.app.api.v1.health import health_check
from backend.app.core.config import settings
from backend.app.schemas.poster import (
    ContentBlockInput,
    PdfExportRequest,
    PosterGenerateRequest,
    RunUpdateRequest,
)


def _mock_inference_response(engine: str = "adaptation") -> dict:
    return {
        "engine": engine,
        "model_name": (
            "poster_layout_retriever+layout_adapter"
            if engine == "adaptation"
            else "poster_layout_generator"
        ),
        "model_version": "2.1.0" if engine == "adaptation" else "0.2.0",
        "model_id": "5a0101a1cf6340718c7d2b85a15bd900",
        "retriever_version": "2.1.0",
        "retriever_model_id": "5a0101a1cf6340718c7d2b85a15bd900",
        "retrieved_template_id": "3345908124",
        "candidate_templates": [
            {
                "rank": 1,
                "poster_id": "3345908124",
                "orientation": "landscape",
                "estimated_column_count": 2,
                "relevance_score": 0.4,
            }
        ],
        "generation_metrics": {
            "is_feasible": True,
            "composite_quality_score": 100.0,
            "validated_overlap_area": 0.0,
            "latency_ms": 120.5,
        },
        "request_features": {
            "summary": {"orientation": "landscape", "total_element_count": 6},
            "blocks": [],
        },
        "layout": {
            "id": str(uuid.uuid4()),
            "title": "Constraint-Aware Layout Generation for Scientific Posters",
            "width": 1000,
            "height": 700,
            "orientation": "landscape",
            "background": "#ffffff",
            "accentColor": "#3458e8",
            "columns": 2,
            "elements": [
                {
                    "id": "b_title",
                    "type": "text",
                    "name": "Poster title",
                    "text": "Constraint-Aware Layout Generation for Scientific Posters",
                    "x": 50.0,
                    "y": 30.0,
                    "width": 900.0,
                    "height": 50.0,
                    "rotation": 0.0,
                    "layer": 0,
                    "visible": True,
                    "locked": False,
                    "sourceBlockId": "b_title",
                    "semanticRole": "title",
                    "bboxNorm": [0.05, 0.04, 0.9, 0.07],
                },
                {
                    "id": "fig_arch",
                    "type": "rect",
                    "name": "Figure box: System Architecture",
                    "x": 45.0,
                    "y": 250.0,
                    "width": 450.0,
                    "height": 250.0,
                    "rotation": 0.0,
                    "layer": 1,
                    "visible": True,
                    "locked": False,
                    "fill": "#eef2ff",
                    "stroke": "#3458e8",
                    "strokeWidth": 1.5,
                    "sourceBlockId": "fig_arch",
                    "semanticRole": "figure",
                    "bboxNorm": [0.045, 0.357, 0.45, 0.357],
                },
                {
                    "id": "fig_arch_lbl",
                    "type": "text",
                    "name": "Figure placeholder label",
                    "text": "[Figure: System Architecture]",
                    "x": 55.0,
                    "y": 360.0,
                    "width": 430.0,
                    "height": 36.0,
                    "rotation": 0.0,
                    "layer": 2,
                    "visible": True,
                    "locked": False,
                    "sourceBlockId": "fig_arch",
                    "semanticRole": "figure_label",
                },
            ],
            "updatedAt": "2026-11-11T15:56:11+00:00",
        },
    }


class TestGenerationMLOpsAPI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir, self.engine, self.SessionFactory = create_test_session_factory()
        self.db = self.SessionFactory()
        self._orig_url = settings.INFERENCE_SERVICE_URL
        settings.INFERENCE_SERVICE_URL = "https://inference.test.local"

    def tearDown(self):
        settings.INFERENCE_SERVICE_URL = self._orig_url
        self.db.close()
        self.engine.dispose()
        self.tmp_dir.cleanup()

    def _sample_payload(
        self, engine: str = "adaptation", image_data_url: str | None = None
    ) -> PosterGenerateRequest:
        return PosterGenerateRequest(
            title="Constraint-Aware Layout Generation for Scientific Posters",
            authors="Alice Smith, Bob Jones",
            affiliations="MIT CSAIL",
            orientation="landscape",
            preferredColumns=3,
            accentColor="#3458e8",
            background="#ffffff",
            engine=engine,
            blocks=[
                ContentBlockInput(
                    id="sec_intro",
                    type="section",
                    heading="1. Introduction",
                    text="Scientific posters communicate dense technical results across structured columns.",
                ),
                ContentBlockInput(
                    id="fig_arch",
                    type="figure",
                    heading="System Architecture",
                    aspectRatio=1.5,
                    imageDataUrl=image_data_url,
                    caption="Figure 1: Overview of retrieval and HiGHS LP optimization.",
                ),
            ],
        )

    def test_health_endpoint(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"status":"ready"}'
        mock_resp.json.return_value = {"status": "ready", "supported_engines": ["neural", "adaptation"]}

        with patch("backend.app.services.layout_pipeline.requests.get", return_value=mock_resp):
            res = health_check(db=self.db)
        self.assertEqual(res["status"], "healthy")
        self.assertEqual(res["database"], "connected")
        self.assertEqual(res["inference_service"]["status"], "ready")

    def test_generate_edit_and_export_training_data_adaptation(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _mock_inference_response("adaptation")

        img_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        with patch("backend.app.services.layout_pipeline.requests.post", return_value=mock_resp) as mock_post:
            run = generate_poster(
                self._sample_payload("adaptation", image_data_url=img_data), db=self.db
            )

        # Ensure inline image blob was stripped from remote inference call and re-attached locally
        sent_json = mock_post.call_args.kwargs["json"]
        self.assertIsNone(sent_json["request"]["blocks"][1]["imageDataUrl"])
        fig_el = next(el for el in run.poster.elements if el["id"] == "fig_arch")
        self.assertEqual(fig_el["type"], "image")
        self.assertEqual(fig_el["src"], img_data)

        self.assertEqual(run.engine, "adaptation")
        self.assertEqual(run.retrievedTemplateId, "3345908124")
        self.assertFalse(run.wasEdited)

        fetched = get_run(run_id=uuid.UUID(run.id), db=self.db)
        self.assertEqual(fetched.title, run.title)

        runs = list_runs(limit=10, edited_only=False, db=self.db)
        self.assertEqual(len(runs), 1)

        edited_doc = fetched.poster.model_copy(deep=True)
        edited_doc.elements[0]["x"] = float(edited_doc.elements[0]["x"]) + 24.0
        updated = update_run(
            run_id=uuid.UUID(run.id),
            payload=RunUpdateRequest(
                editedLayout=edited_doc.model_dump(),
                exportedFormat="png",
                feedbackRating=5,
                feedbackNotes="Clean layout",
            ),
            db=self.db,
        )
        self.assertTrue(updated.wasEdited)
        self.assertEqual(updated.feedbackRating, 5)
        self.assertIn("png", updated.exportedFormats)

        export_resp = export_training_data(format="jsonl", edited_only=True, limit=50, db=self.db)
        lines = export_resp.body.decode("utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["run_id"], run.id)
        self.assertTrue(record["was_edited"])
        self.assertIsNotNone(record["generated_layout"])
        self.assertIsNotNone(record["edited_layout"])

    def test_unconfigured_and_unreachable_inference_service(self):
        settings.INFERENCE_SERVICE_URL = ""
        with self.assertRaises(HTTPException) as ctx:
            generate_poster(self._sample_payload("adaptation"), db=self.db)
        self.assertEqual(ctx.exception.status_code, 503)

        settings.INFERENCE_SERVICE_URL = "https://inference.test.local"
        import requests as req_lib

        with patch(
            "backend.app.services.layout_pipeline.requests.post",
            side_effect=req_lib.ConnectionError("refused"),
        ):
            with self.assertRaises(HTTPException) as ctx2:
                generate_poster(self._sample_payload("adaptation"), db=self.db)
            self.assertEqual(ctx2.exception.status_code, 502)

    def test_pdf_export_from_canvas_png(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _mock_inference_response("adaptation")
        with patch("backend.app.services.layout_pipeline.requests.post", return_value=mock_resp):
            run = generate_poster(self._sample_payload("adaptation"), db=self.db)

        img = Image.new("RGB", (200, 140), color=(240, 244, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

        pdf_resp = export_pdf(
            PdfExportRequest(title="Test Poster", imageDataUrl=data_url, runId=run.id),
            db=self.db,
        )
        self.assertEqual(pdf_resp.media_type, "application/pdf")
        self.assertTrue(pdf_resp.body.startswith(b"%PDF"))

        fetched = get_run(run_id=uuid.UUID(run.id), db=self.db)
        self.assertIn("pdf", fetched.exportedFormats)


if __name__ == "__main__":
    unittest.main()
