"""Unit and integration tests for Infographix Stage 3 Adaptation and Generation Pipeline."""

import os
import tempfile
import unittest
from pathlib import Path

# Ensure writable matplotlib temp directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
from src.adaptation.adaptation_evaluation import (
    compute_pairwise_overlap_area,
    evaluate_adapted_layout,
    run_greedy_baseline_placement,
)
from src.adaptation.content_geometry import (
    GeometricDemand,
    compute_content_demands,
    estimate_image_content_geometry,
    estimate_table_content_geometry,
    estimate_text_content_geometry,
)
from src.adaptation.layout_adapter import LayoutAdapter
from src.adaptation.layout_constraints import (
    optimize_column_layout,
    solve_layout_optimization,
)
from src.adaptation.slot_assignment import (
    AssignedSlot,
    assign_content_to_template,
)
from src.adaptation.template_regions import (
    HierarchicalTemplate,
    extract_hierarchical_template,
)
from src.features.request_schema import ContentBlockDemand, PosterRequest
from src.rendering.html_renderer import render_to_html
from src.rendering.pdf_export import export_to_image, export_to_pdf
from src.rendering.svg_renderer import render_to_svg
from src.schemas.layout_document import ElementStyle, PosterCanvas, PosterDocument, PosterElement


class TestStage3Pipeline(unittest.TestCase):
    """Test suite covering schemas, adaptation pipeline, and rendering."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # Mock elements DataFrame conforming to pipeline_v2 element features
        self.mock_elements_df = pd.DataFrame([
            # Header
            {
                "poster_id": 999999, "element_id": 1, "category_slug": "title",
                "x0": 0.1, "y0": 0.05, "w_norm": 0.8, "h_norm": 0.08, "cx": 0.5, "cy": 0.09,
                "area_norm": 0.064, "raw_x": 100, "raw_y": 50, "raw_w": 800, "raw_h": 80,
            },
            {
                "poster_id": 999999, "element_id": 2, "category_slug": "author_info",
                "x0": 0.15, "y0": 0.14, "w_norm": 0.7, "h_norm": 0.04, "cx": 0.5, "cy": 0.16,
                "area_norm": 0.028, "raw_x": 150, "raw_y": 140, "raw_w": 700, "raw_h": 40,
            },
            # Column 0 (x0: 0.05 to 0.33)
            {
                "poster_id": 999999, "element_id": 3, "category_slug": "section",
                "x0": 0.05, "y0": 0.22, "w_norm": 0.28, "h_norm": 0.04, "cx": 0.19, "cy": 0.24,
                "area_norm": 0.011, "raw_x": 50, "raw_y": 220, "raw_w": 280, "raw_h": 40,
            },
            {
                "poster_id": 999999, "element_id": 4, "category_slug": "text",
                "x0": 0.05, "y0": 0.28, "w_norm": 0.28, "h_norm": 0.25, "cx": 0.19, "cy": 0.405,
                "area_norm": 0.07, "raw_x": 50, "raw_y": 280, "raw_w": 280, "raw_h": 250,
            },
            {
                "poster_id": 999999, "element_id": 5, "category_slug": "section",
                "x0": 0.05, "y0": 0.55, "w_norm": 0.28, "h_norm": 0.04, "cx": 0.19, "cy": 0.57,
                "area_norm": 0.011, "raw_x": 50, "raw_y": 550, "raw_w": 280, "raw_h": 40,
            },
            {
                "poster_id": 999999, "element_id": 6, "category_slug": "text",
                "x0": 0.05, "y0": 0.61, "w_norm": 0.28, "h_norm": 0.28, "cx": 0.19, "cy": 0.75,
                "area_norm": 0.078, "raw_x": 50, "raw_y": 610, "raw_w": 280, "raw_h": 280,
            },
            # Column 1 (x0: 0.36 to 0.64)
            {
                "poster_id": 999999, "element_id": 7, "category_slug": "section",
                "x0": 0.36, "y0": 0.22, "w_norm": 0.28, "h_norm": 0.04, "cx": 0.50, "cy": 0.24,
                "area_norm": 0.011, "raw_x": 360, "raw_y": 220, "raw_w": 280, "raw_h": 40,
            },
            {
                "poster_id": 999999, "element_id": 8, "category_slug": "figure",
                "x0": 0.36, "y0": 0.28, "w_norm": 0.28, "h_norm": 0.30, "cx": 0.50, "cy": 0.43,
                "area_norm": 0.084, "raw_x": 360, "raw_y": 280, "raw_w": 280, "raw_h": 300,
            },
            {
                "poster_id": 999999, "element_id": 9, "category_slug": "caption",
                "x0": 0.36, "y0": 0.60, "w_norm": 0.28, "h_norm": 0.06, "cx": 0.50, "cy": 0.63,
                "area_norm": 0.017, "raw_x": 360, "raw_y": 600, "raw_w": 280, "raw_h": 60,
            },
            # Column 2 (x0: 0.67 to 0.95)
            {
                "poster_id": 999999, "element_id": 10, "category_slug": "section",
                "x0": 0.67, "y0": 0.22, "w_norm": 0.28, "h_norm": 0.04, "cx": 0.81, "cy": 0.24,
                "area_norm": 0.011, "raw_x": 670, "raw_y": 220, "raw_w": 280, "raw_h": 40,
            },
            {
                "poster_id": 999999, "element_id": 11, "category_slug": "table",
                "x0": 0.67, "y0": 0.28, "w_norm": 0.28, "h_norm": 0.25, "cx": 0.81, "cy": 0.405,
                "area_norm": 0.07, "raw_x": 670, "raw_y": 280, "raw_w": 280, "raw_h": 250,
            },
            {
                "poster_id": 999999, "element_id": 12, "category_slug": "text",
                "x0": 0.67, "y0": 0.55, "w_norm": 0.28, "h_norm": 0.30, "cx": 0.81, "cy": 0.70,
                "area_norm": 0.084, "raw_x": 670, "raw_y": 550, "raw_w": 280, "raw_h": 300,
            },
        ])

        # Mock poster metadata row
        self.mock_poster_row = pd.Series({
            "poster_id": 999999,
            "canvas_width_px": 1000.0,
            "canvas_height_px": 1000.0,
            "aspect_ratio": 1.0,
            "orientation": "square",
            "estimated_column_count": 3,
            "total_element_count": 12,
        })

        # Mock user PosterRequest
        self.sample_request = PosterRequest(
            request_id="req_test_stage3",
            requested_aspect_ratio=1.0,
            requested_orientation="square",
            blocks=[
                ContentBlockDemand("b_title", "title", "title", word_count=8),
                ContentBlockDemand("b_auth", "author_info", "author_info", word_count=12),
                ContentBlockDemand("b_intro_head", "section", "section", word_count=2),
                ContentBlockDemand("b_intro_body", "text", "text", word_count=50),
                ContentBlockDemand("b_method_head", "section", "section", word_count=2),
                ContentBlockDemand("b_method_body", "text", "text", word_count=65),
                ContentBlockDemand("b_res_head", "section", "section", word_count=3),
                ContentBlockDemand("b_res_fig", "figure", "figure", aspect_ratio=1.33),
                ContentBlockDemand("b_res_cap", "caption", "caption", word_count=18),
                ContentBlockDemand("b_disc_head", "section", "section", word_count=3),
                ContentBlockDemand("b_disc_tbl", "table", "table", table_rows=4, table_cols=3),
                ContentBlockDemand("b_disc_body", "text", "text", word_count=45),
            ],
            author_count=2,
            affiliation_count=1,
            style_preferences={
                "title_text": "Neural Layout Adaptation for Scientific Posters",
                "authors_text": "Dr. Jane Doe, Dr. John Smith",
                "sections": {
                    "b_intro_head": "1. Introduction",
                    "b_intro_body": "Scientific posters require careful visual organization to communicate complex data efficiently.",
                    "b_method_head": "2. Methods",
                    "b_method_body": "We construct hierarchical templates and solve linear programs to optimize placement.",
                    "b_res_head": "3. Results",
                    "b_res_cap": "Figure 1: Objective alignment metrics demonstrating zero overlap.",
                    "b_disc_head": "4. Conclusions",
                    "b_disc_body": "The constraint-based approach reliably produces clean, publication-ready layouts.",
                },
            },
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_poster_document_schema(self):
        """Test document serialization, JSON round-trip, and COCO conversion."""
        canvas = PosterCanvas(width=1200, height=800, background_color="#ffffff")
        doc = PosterDocument(
            poster_id="doc_stage3_001",
            source_template_id=999999,
            canvas=canvas,
            title="Neural Layout Adaptation",
            authors=["Dr. Jane Doe"],
            elements=[
                PosterElement(
                    id="el_title_1",
                    type="title",
                    bbox=[50.0, 40.0, 1100.0, 60.0],
                    bbox_norm=[50 / 1200, 40 / 800, 1100 / 1200, 60 / 800],
                    content="Neural Layout Adaptation",
                    style=ElementStyle(font_size_pt=28.0, font_weight="bold"),
                )
            ],
        )

        # JSON Roundtrip
        out_file = self.tmp_path / "test_doc.json"
        doc.save_json(out_file)
        self.assertTrue(out_file.exists())

        loaded_doc = PosterDocument.from_dict(doc.to_dict())
        self.assertEqual(loaded_doc.poster_id, "doc_stage3_001")
        self.assertEqual(len(loaded_doc.elements), 1)
        self.assertEqual(loaded_doc.elements[0].type, "title")

        # COCO Export
        coco = doc.to_coco()
        self.assertEqual(len(coco.images), 1)
        self.assertEqual(len(coco.annotations), 1)
        self.assertEqual(coco.images[0].width, 1200)

    def test_template_hierarchy_extraction(self):
        """Test hierarchical template extraction, column detection, and pairings."""
        template = extract_hierarchical_template(
            poster_row=self.mock_poster_row,
            elements_df=self.mock_elements_df,
        )

        self.assertEqual(template.poster_id, 999999)
        self.assertEqual(len(template.header_slots), 2)
        self.assertEqual(len(template.columns), 3)

        # Figure-caption pairing detection
        fig_slots = [s for s in template.body_slots if s.category_slug == "figure"]
        self.assertEqual(len(fig_slots), 1)
        self.assertIsNotNone(fig_slots[0].paired_with_id)

    def test_content_geometry_estimation(self):
        """Test text typography, image aspect ratio, and table dimension estimation."""
        # Text wrapping
        text_geom = estimate_text_content_geometry(
            text="Scientific posters communicate research effectively. Our system automates composition.",
            candidate_col_width_norm=0.28,
        )
        self.assertGreater(text_geom["preferred_height"], 0.02)
        self.assertGreaterEqual(text_geom["preferred_height"], text_geom["min_height"])

        # Image locked aspect ratio
        img_geom = estimate_image_content_geometry(
            aspect_ratio=1.5,
            candidate_col_width_norm=0.30,
            canvas_aspect_ratio=1.0,
        )
        self.assertTrue(img_geom["lock_aspect_ratio"])
        self.assertAlmostEqual(img_geom["preferred_height"], 0.30 / 1.5, delta=0.01)

        # Table dimensions
        tbl_geom = estimate_table_content_geometry(
            table_rows=5,
            table_cols=3,
            candidate_col_width_norm=0.28,
        )
        self.assertGreater(tbl_geom["preferred_height"], 0.05)

    def test_slot_assignment_and_region_split(self):
        """Test semantic slot assignment and dynamic splitting/merging."""
        template = extract_hierarchical_template(
            poster_row=self.mock_poster_row,
            elements_df=self.mock_elements_df,
        )
        demands = compute_content_demands(
            request=self.sample_request,
            candidate_col_width_norm=0.28,
            canvas_aspect_ratio=1.0,
        )

        assigned_slots = assign_content_to_template(
            demands=demands,
            template=template,
        )

        self.assertEqual(len(assigned_slots), len(demands))
        # Ensure header demands mapped to header column (None)
        hdr_assigned = [s for s in assigned_slots if s.column_idx is None]
        self.assertEqual(len(hdr_assigned), 2)

    def test_highs_layout_optimization(self):
        """Test SciPy HiGHS LP formulation for column layout."""
        template = extract_hierarchical_template(
            poster_row=self.mock_poster_row,
            elements_df=self.mock_elements_df,
        )
        demands = compute_content_demands(
            request=self.sample_request,
            candidate_col_width_norm=0.28,
            canvas_aspect_ratio=1.0,
        )
        assigned_slots = assign_content_to_template(
            demands=demands,
            template=template,
        )

        opt_res = solve_layout_optimization(
            assigned_slots=assigned_slots,
            canvas_aspect_ratio=1.0,
        )

        self.assertTrue(opt_res.is_feasible)
        self.assertEqual(len(opt_res.adapted_slots), len(assigned_slots))
        # No vertical overlaps
        col_slots = [s for s in opt_res.adapted_slots if s.column_idx == 0]
        col_slots_sorted = sorted(col_slots, key=lambda s: s.target_bbox_norm[1])
        for i in range(len(col_slots_sorted) - 1):
            cur = col_slots_sorted[i]
            nxt = col_slots_sorted[i + 1]
            cur_bottom = cur.target_bbox_norm[1] + cur.target_bbox_norm[3]
            nxt_top = nxt.target_bbox_norm[1]
            self.assertGreaterEqual(nxt_top + 1e-4, cur_bottom)

    def test_adaptation_evaluation(self):
        """Test calculation of 9 objective metrics and comparison with greedy baseline."""
        template = extract_hierarchical_template(
            poster_row=self.mock_poster_row,
            elements_df=self.mock_elements_df,
        )
        demands = compute_content_demands(
            request=self.sample_request,
            candidate_col_width_norm=0.28,
            canvas_aspect_ratio=1.0,
        )
        assigned_slots = assign_content_to_template(demands, template)

        # Baseline placement
        greedy_slots = run_greedy_baseline_placement(demands, template)
        greedy_metrics = evaluate_adapted_layout(demands, greedy_slots, canvas_aspect=1.0)

        # HiGHS placement
        opt_res = solve_layout_optimization(assigned_slots, canvas_aspect_ratio=1.0)
        opt_metrics = evaluate_adapted_layout(demands, opt_res.adapted_slots, canvas_aspect=1.0)

        self.assertEqual(opt_metrics["placement_success_rate"], 1.0)
        self.assertEqual(opt_metrics["content_preservation"], 1.0)
        self.assertLessEqual(opt_metrics["overlap_area"], 1e-4)
        self.assertGreater(opt_metrics["composite_quality_score"], 70.0)

    def test_end_to_end_layout_adapter(self):
        """Test full LayoutAdapter candidate processing."""
        adapter = LayoutAdapter(
            elements_df=self.mock_elements_df,
            canvas_width=1000,
            canvas_height=1000,
        )
        res = adapter.adapt_single_candidate(
            request=self.sample_request,
            candidate_poster_row=self.mock_poster_row,
            candidate_rank=1,
        )

        self.assertTrue(res.is_feasible)
        self.assertIsInstance(res.poster_document, PosterDocument)
        self.assertEqual(len(res.poster_document.elements), len(self.sample_request.blocks))

    def test_renderers_html_svg_pdf(self):
        """Test rendering to HTML, SVG, and vector PDF."""
        adapter = LayoutAdapter(
            elements_df=self.mock_elements_df,
            canvas_width=1000,
            canvas_height=1000,
        )
        res = adapter.adapt_single_candidate(
            request=self.sample_request,
            candidate_poster_row=self.mock_poster_row,
            candidate_rank=1,
        )
        doc = res.poster_document

        # HTML
        html_file = self.tmp_path / "output.html"
        html_str = render_to_html(doc, output_path=html_file)
        self.assertTrue(html_file.exists())
        self.assertIn("<!DOCTYPE html>", html_str)
        self.assertIn("Infographix", html_str)

        # SVG
        svg_file = self.tmp_path / "output.svg"
        svg_str = render_to_svg(doc, output_path=svg_file)
        self.assertTrue(svg_file.exists())
        self.assertIn("<svg", svg_str)
        self.assertIn("</svg>", svg_str)

        # PDF
        pdf_file = self.tmp_path / "output.pdf"
        export_to_pdf(doc, output_path=pdf_file, dpi=100)
        self.assertTrue(pdf_file.exists())
        self.assertGreater(pdf_file.stat().st_size, 1000)

        # Image
        img_file = self.tmp_path / "output.png"
        export_to_image(doc, output_path=img_file, dpi=100)
        self.assertTrue(img_file.exists())
        self.assertGreater(img_file.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
