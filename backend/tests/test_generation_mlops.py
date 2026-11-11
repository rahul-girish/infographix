import json
import io
import base64
import unittest
import uuid

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
from backend.app.schemas.poster import (
    ContentBlockInput,
    PdfExportRequest,
    PosterGenerateRequest,
    RunUpdateRequest,
)


class TestGenerationMLOpsAPI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir, self.engine, self.SessionFactory = create_test_session_factory()
        self.db = self.SessionFactory()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        self.tmp_dir.cleanup()

    def _sample_payload(self, engine: str = "adaptation") -> PosterGenerateRequest:
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
                    caption="Figure 1: Overview of retrieval and HiGHS LP optimization.",
                ),
                ContentBlockInput(
                    id="sec_conc",
                    type="section",
                    heading="2. Conclusion",
                    text="Zero-overlap layouts are produced automatically from raw user inputs.",
                ),
            ],
        )

    def test_health_endpoint(self):
        res = health_check(db=self.db)
        self.assertEqual(res["status"], "healthy")
        self.assertEqual(res["database"], "connected")
        self.assertTrue(res["models"]["retrieverLoaded"])

    def test_generate_edit_and_export_training_data_adaptation(self):
        run = generate_poster(self._sample_payload("adaptation"), db=self.db)
        self.assertEqual(run.engine, "adaptation")
        self.assertIsNotNone(run.retrievedTemplateId)
        self.assertGreater(len(run.poster.elements), 4)
        self.assertFalse(run.wasEdited)

        # Fetch via GET /runs/{id} and GET /runs
        fetched = get_run(run_id=uuid.UUID(run.id), db=self.db)
        self.assertEqual(fetched.title, run.title)

        runs = list_runs(limit=10, edited_only=False, db=self.db)
        self.assertEqual(len(runs), 1)

        # Simulate user moving an element in the editor and rating the run
        edited_doc = fetched.poster.model_copy(deep=True)
        edited_doc.elements[0]["x"] = float(edited_doc.elements[0]["x"]) + 24.0
        updated = update_run(
            run_id=uuid.UUID(run.id),
            payload=RunUpdateRequest(
                editedLayout=edited_doc.model_dump(),
                exportedFormat="png",
                feedbackRating=5,
                feedbackNotes="Clean 3-column layout",
            ),
            db=self.db,
        )
        self.assertTrue(updated.wasEdited)
        self.assertEqual(updated.feedbackRating, 5)
        self.assertIn("png", updated.exportedFormats)

        # Export training data JSONL
        export_resp = export_training_data(format="jsonl", edited_only=True, limit=50, db=self.db)
        lines = export_resp.body.decode("utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["run_id"], run.id)
        self.assertTrue(record["was_edited"])
        self.assertIn("summary", record["request_features"])
        self.assertIsNotNone(record["generated_layout"])
        self.assertIsNotNone(record["edited_layout"])

    def test_generate_neural_engine(self):
        run = generate_poster(self._sample_payload("neural"), db=self.db)
        self.assertEqual(run.engine, "neural")
        self.assertEqual(run.modelName, "TemplateConditionedLayoutModel")
        self.assertGreater(len(run.poster.elements), 4)

    def test_pdf_export_from_canvas_png(self):
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
