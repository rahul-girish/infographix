"""Execute notebooks/04_layout_adaptation_and_generation.ipynb and embed rich outputs."""

import base64
import contextlib
import os
import sys
from io import StringIO
from pathlib import Path

import nbformat as nbf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_layout_adaptation_and_generation.ipynb"


def execute_and_embed():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    globals_dict = {"__name__": "__main__"}

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            source = cell.source
            f_out = StringIO()
            try:
                with contextlib.redirect_stdout(f_out):
                    exec(source, globals_dict)
                out_str = f_out.getvalue()
                cell.outputs = []
                if out_str:
                    cell.outputs.append(
                        nbf.v4.new_output(
                            output_type="stream",
                            name="stdout",
                            text=out_str,
                        )
                    )

                # If the cell displays images or tables, embed display_data
                if "metrics_df.pivot" in source:
                    metrics_path = PROJECT_ROOT / "reports" / "adaptation" / "stage3_benchmark_metrics.csv"
                    m_df = pd.read_csv(metrics_path)
                    piv = m_df.pivot(index="scenario", columns="paradigm", values=["composite_quality_score", "overlap_area", "overflow_rate"])
                    cell.outputs.append(
                        nbf.v4.new_output(
                            output_type="display_data",
                            data={
                                "text/plain": piv.to_string(),
                                "text/html": piv.to_html(classes="table table-striped table-bordered"),
                            },
                        )
                    )

                if "scenarios_to_show" in source:
                    fig_dir = PROJECT_ROOT / "reports" / "figures" / "stage3"
                    for s_name, fname in [
                        ("Scenario A (Normal)", "scenario_a_normal_comparison.png"),
                        ("Scenario B (Figure-Heavy)", "scenario_b_figure_heavy_comparison.png"),
                        ("Scenario C (Text-Heavy)", "scenario_c_text_heavy_comparison.png"),
                        ("Scenario D (Sparse)", "scenario_d_sparse_comparison.png"),
                        ("Scenario E (Difficult)", "scenario_e_difficult_comparison.png"),
                    ]:
                        img_p = fig_dir / fname
                        if img_p.exists():
                            b64 = base64.b64encode(img_p.read_bytes()).decode("ascii")
                            cell.outputs.append(
                                nbf.v4.new_output(
                                    output_type="display_data",
                                    data={"image/png": b64, "text/plain": f"<Image: {s_name}>"},
                                )
                            )

                if "summary_chart" in source:
                    chart_p = PROJECT_ROOT / "reports" / "figures" / "stage3" / "tri_paradigm_metrics_comparison.png"
                    if chart_p.exists():
                        b64 = base64.b64encode(chart_p.read_bytes()).decode("ascii")
                        cell.outputs.append(
                            nbf.v4.new_output(
                                output_type="display_data",
                                data={"image/png": b64, "text/plain": "<Image: tri_paradigm_metrics_comparison.png>"},
                            )
                        )

                print(f"Cell {i} executed and outputs recorded successfully.")
            except Exception as e:
                print(f"Cell {i} failed: {e}")
                raise e

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook updated with rich inline visualizations.")


if __name__ == "__main__":
    execute_and_embed()
