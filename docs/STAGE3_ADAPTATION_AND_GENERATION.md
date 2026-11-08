# Stage 3: Content-Aware Layout Adaptation and Poster Generation
**Infographix Production Technical Specification & Benchmarking Report**
*Version: 3.0.0 | Author: Infographix Core Engineering | Dataset: SciPostLayout v2*

---

## 1. Executive Summary & Objective

In **Stage 2**, Infographix established a constraint-aware retrieval engine (`LayoutRetriever` v2.0.0) that takes a structured `PosterRequest` and identifies structurally compatible reference templates from 7,855 scientific posters in the SciPostLayout v2 dataset.

However, retrieved posters are **static geometric templates** containing fixed bounding boxes designed for someone else's content. A real user provides variable scientific content:
- Multi-paragraph abstracts and methodology descriptions of varying word counts.
- Arbitrary collections of figures with non-standard aspect ratios (panoramic charts, tall confocal microscopy micrographs).
- High-density data tables and multi-sentence captions.
- Fixed author affiliations and institution metadata.

**Stage 3** solves the micro-layout adaptation challenge: **adapting retrieved structural templates to arbitrarily conditioned user content** without human intervention, while mathematically guaranteeing zero element collisions, strict typographic readability, and preservation of visual hierarchy.

```
                      [User Content Request]
                      (Text, Figures, Tables)
                                 │
                                 ▼
   [Stage 2: Macro Retrieval] ────────► [Top-K Reference Templates]
                                                │
                                                ▼
                             [1. Hierarchical Template Extraction]
                             (Header, Columns, Section Units, Pairs)
                                                │
                                                ▼
                             [2. Content Geometry Estimation]
                             (Typography Wrapping, Aspect Ratio Lock)
                                                │
                                                ▼
                             [3. Semantic Slot Assignment]
                             (Dynamic Region Splitting & Merging)
                                                │
                                                ▼
                             [4. Constraint Optimization (HiGHS LP)]
                             (Zero Overlap, Bounds, Aspect Ratios)
                                                │
                                                ▼
                             [5. Structured PosterDocument]
                                 (JSON + COCO Compatible)
                                 ┌──────┴──────┐
                                 ▼             ▼
                           [HTML/CSS]      [Vector PDF]
                           [Vector SVG]    [PNG Preview]
```

---

## 2. System Architecture & Core Modules

The Stage 3 implementation consists of five interconnected subsystems located in `src/adaptation/`, `src/schemas/`, and `src/rendering/`:

### 2.1. Intermediate Layout Representation (`src/adaptation/template_regions.py`)
Treating a reference poster as an unorganized set of bounding boxes causes catastrophic failure during resizing. Instead, `extract_hierarchical_template()` parses the reference poster into a semantic tree:
- **Header Band**: Isolates titles, author blocks, and institutional logos located in the top margin ($y_0 < 0.28$ or full-width banners with $w \ge 0.70$).
- **Column Partitions**: Clusters body bounding boxes into $K$ discrete vertical columns using histogram peak analysis and horizontal interval partitioning.
- **Section Groupings**: Associates section headings with following body paragraphs.
- **Coupled Pairs**: Automatically pairs figures with their immediate spatial captions (within vertical distance $\Delta y \le 0.08$ and horizontal overlap $\ge 60\%$).
- **Confidence Scoring**: Computes an aggregate confidence score (0.0 to 1.0) indicating how cleanly the reference poster conforms to standard column geometry.

### 2.2. Content Geometry Demand Estimation (`src/adaptation/content_geometry.py`)
Unlike images, **text does not scale with constant area**. When column width narrows, line length decreases and line count increases, causing text height to expand inversely:
$$\text{chars\_per\_line} = \max\left(10, \frac{w_{\text{col}} \cdot W_{\text{canvas}} - 2 \cdot \text{padding}}{\text{font\_size} \cdot \text{em\_width}}\right)$$
$$\text{estimated\_lines} = \left\lceil \frac{\text{char\_count}}{\text{chars\_per\_line}} \right\rceil$$
$$h_{\text{norm}} = \frac{\text{estimated\_lines} \cdot (\text{font\_size} \cdot \text{line\_height}) + \text{padding}}{H_{\text{canvas}}}$$

- **Images & Figures**: Strictly enforce aspect-ratio preservation. Normalized height is coupled to normalized width via the canvas aspect ratio:
$$h_{\text{norm}} = w_{\text{norm}} \cdot \left(\frac{\text{canvas\_aspect}}{\text{target\_aspect}}\right)$$
- **Tables**: Compute minimum readable cell sizes ($w_{\text{col}} \ge 42\text{ pt}$, $h_{\text{row}} \ge 20\text{ pt}$) to guarantee tabular data is never crushed.

### 2.3. Semantic Slot Assignment & Region Splitting (`src/adaptation/slot_assignment.py`)
Maps incoming user demands to candidate template slots:
- **Coupled Preservation**: Section headings are bound directly to their respective text paragraphs; figures are bound to their respective captions.
- **Dynamic Region Splitting**: When the user provides more figures or text blocks than exist in the template, candidate slots are split along the vertical axis with appropriate inter-element margins ($y_{\text{gap}} = 0.012$).
- **Dynamic Region Merging**: When user content is sparse, unused slots are collapsed, distributing remaining vertical whitespace proportionally across active blocks.

### 2.4. Constraint-Based Layout Optimization (`src/adaptation/layout_constraints.py`)
Formulated as an exact Linear Program (LP) solved via SciPy's HiGHS solver (`scipy.optimize.linprog(method='highs')`):

#### Decision Variables
For each placed element $i \in \{1, \dots, N\}$ in column $c$:
- $y_i \in [0, 1]$: Normalized top vertical coordinate.
- $h_i \in [h_i^{\min}, h_i^{\max}]$: Allocated normalized height.
- $\delta_i^+, \delta_i^- \ge 0$: Slack variables measuring deviations from preferred height.
- $s_{\text{overflow}} \ge 0$: Column overflow slack extending past the canvas bottom boundary.

#### Constraints
1. **Top Canvas Margin**:
   $$y_0 \ge y_{\text{start}} \quad (\text{below header band})$$
2. **Strict Non-Overlap Ordering**:
   $$y_{i+1} \ge y_i + h_i + \text{gap}_i \quad \forall i < N$$
3. **Bottom Canvas Boundary**:
   $$y_{N-1} + h_{N-1} \le y_{\max} + s_{\text{overflow}}$$
4. **Figure-Caption Coupling Equality**:
   $$y_{\text{cap}} = y_{\text{fig}} + h_{\text{fig}} + \text{gap}_{\text{caption}}$$
5. **Image Aspect-Ratio Lock**:
   $$h_{\text{fig}} = w_{\text{col}} \cdot \left(\frac{\text{canvas\_aspect}}{\text{aspect\_ratio}}\right)$$

#### Objective Function
$$\min \sum_{i=1}^N \left( \lambda_{\text{dev}} (\delta_i^+ + \delta_i^-) + \lambda_{\text{pref}} |h_i - h_i^{\text{pref}}| \right) + \lambda_{\text{overflow}} s_{\text{overflow}} + \lambda_{\text{white}} (y_{\max} - y_{N-1} - h_{N-1})$$

Where $\lambda_{\text{overflow}} = 1000.0$ imposes an extreme penalty on canvas clipping, while $\lambda_{\text{dev}} = 1.0$ penalizes deviation from ideal typography.

---

## 3. Editable Document Schema & Renderers

### 3.1. Structured JSON Document (`src/schemas/layout_document.py`)
The canonical `PosterDocument` container stores:
- `canvas`: Physical dimensions, DPI, orientation, background color, margin.
- `elements`: List of `PosterElement` records containing:
  - Stable IDs, semantic category, normalized bounding box $[x_0, y_0, w, h]$, pixel bounding box $[x, y, w, h]$.
  - Actual textual strings, figure URLs/placeholders, structured tabular matrices.
  - Typographic styling (`font_family`, `font_size_pt`, `font_weight`, `text_color`, `line_height`, `padding_px`).
  - Spatial grouping metadata (`column_idx`, `group_id`, `associated_element_ids`).
- Full round-trip compatibility with COCO annotation datasets via `doc.to_coco()`.

### 3.2. Vector Renderers (`src/rendering/`)
- **HTML Renderer (`html_renderer.py`)**: Generates responsive, standalone HTML/CSS with Inter typography, slate palette cards, formatted HTML `<table>` elements with zebra-striping, and SVG placeholder illustrations. Includes interactive inspector attributes (`data-element-id`, `data-type`, `data-col`).
- **SVG Renderer (`svg_renderer.py`)**: Emits layered vector XML (`<defs>`, `<style>`, `<g id="layer-content">`), text wrapping using calculated `<tspan>` coordinates, and clean vector accents.
- **PDF Exporter (`pdf_export.py`)**: Uses Matplotlib's vector PDF backend (`fig.savefig(format='pdf')`) to produce crisp, publication-ready vector PDFs conforming to exact physical print dimensions without headless browser dependencies.

---

## 4. Empirical Evaluation Across 5 Scenarios

We benchmarked three layout paradigms across 5 distinct scientific poster scenarios:
1. **Unmodified Retrieved Template**: Naive placement directly into nearest retrieved template bounding boxes.
2. **Greedy Baseline Sequential Placement**: Top-to-bottom column packing without optimization.
3. **Constraint-Based Optimization (HiGHS LP)**: Our complete Stage 3 mathematical formulation.

### 4.1. Benchmark Metrics Table

| Scenario | Paradigm | Composite Quality (0–100) | Placement Success | Content Preserved | Overflow Rate | Total Overflow Depth | Overlap Area | Aspect Distortion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Normal** | 1. Unmodified Template | 79.3 | 100.0% | 100.0% | 22.2% | 0.013 | 0.0000 | 4.22 |
| *(18 blocks)* | 2. Greedy Baseline | 95.7 | 100.0% | 100.0% | 16.7% | 0.000 | 0.0000 | 0.01 |
| *(5 sec, 3 fig, 1 tbl)* | **3. HiGHS LP (Ours)** | **98.0** | **100.0%** | **100.0%** | **5.6%** | **0.000** | **0.0000** | **0.00** |
| **B. Figure-Heavy** | 1. Unmodified Template | 70.8 | 96.0% | 96.0% | 20.8% | 0.000 | 0.0099 | 3.83 |
| *(25 blocks)* | 2. Greedy Baseline | 74.8 | 100.0% | 100.0% | 80.0% | 9.937 | 0.0000 | 0.02 |
| *(8 fig, 2 tbl)* | **3. HiGHS LP (Ours)** | **74.1\*** | **100.0%** | **100.0%** | **48.0%** | **0.738** | **0.0000** | **0.02** |
| **C. Text-Heavy** | 1. Unmodified Template | 81.3 | 91.3% | 91.3% | 0.0% | 0.000 | 0.0001 | 0.84 |
| *(23 blocks)* | 2. Greedy Baseline | 92.6 | 100.0% | 100.0% | 21.7% | 0.033 | 0.0000 | 0.03 |
| *(8 sec, ~1200 wds)* | **3. HiGHS LP (Ours)** | **99.6** | **100.0%** | **100.0%** | **0.0%** | **0.000** | **0.0000** | **0.00** |
| **D. Sparse** | 1. Unmodified Template | 83.0 | 100.0% | 100.0% | 0.0% | 0.000 | 0.0000 | 2.08 |
| *(10 blocks)* | 2. Greedy Baseline | 100.0 | 100.0% | 100.0% | 0.0% | 0.000 | 0.0000 | 0.00 |
| *(3 sec, 1 fig)* | **3. HiGHS LP (Ours)** | **99.6** | **100.0%** | **100.0%** | **0.0%** | **0.000** | **0.0000** | **0.04** |
| **E. Difficult** | 1. Unmodified Template | 81.8 | 100.0% | 100.0% | 15.4% | 0.011 | 0.0000 | 1.86 |
| *(13 blocks)* | 2. Greedy Baseline | 90.1 | 100.0% | 100.0% | 30.8% | 0.000 | 0.0000 | 0.22 |
| *(2.65:1 pan, 6x6 tbl)* | **3. HiGHS LP (Ours)** | **96.0** | **100.0%** | **100.0%** | **0.0%** | **0.000** | **0.0000** | **0.24** |

*\*Note on Scenario B: The content load of 8 figures + 2 tables exceeds the physical canvas area of a standard landscape poster. While Greedy blindly pushes elements 10x past the bottom boundary (overflow depth 9.937), HiGHS LP compresses slack to the minimum readable limit, reducing total overflow depth by 92.6% (to 0.738) while preserving zero overlap.*

---

## 5. Key Findings & Discussion

### 5.1. The Failure of Unmodified Templates
- **Content Dropping**: In Scenarios B and C, the unmodified template dropped user content blocks (`unplaced_count = 1` in B, `unplaced_count = 2` in C) because the retrieved poster had fewer annotated slots than requested.
- **Aspect Distortion**: Forcing user images into arbitrary template bounding boxes resulted in severe aspect ratio distortion (mean error up to **4.22** in Scenario A), stretching scientific figures beyond recognition.
- **Readability Violations**: Between 38% and 76% of elements violated minimum font-size readability thresholds when squeezed into fixed template boxes.

### 5.2. Greedy vs. Constraint Optimization
- **Zero-Overlap Guarantee**: HiGHS LP achieved **0.0000 pairwise overlap** across all 5 test scenarios.
- **Canvas Overflow Prevention**: In Scenario C (text-heavy, 1200 words), Greedy placement spilled 21.7% of content off the canvas. HiGHS LP compressed vertical gutters and adjusted line heights, achieving **0.0% overflow** and a **99.6 / 100** quality score.
- **Complex Geometries (Scenario E)**: Accommodating a 2.65:1 panoramic figure alongside a 0.48:1 tall microscopy figure and a 6x6 data table caused 30.8% overflow in Greedy placement. HiGHS LP satisfied all aspect constraints with **0.0% overflow** and **96.0 / 100** quality.

---

## 6. Infeasibility Diagnostics & User Guidance

When an extreme content request exceeds the physical capacity of any single poster canvas (as observed in Scenario B):
1. **Explicit Overflow Flagging**: The pipeline records `is_feasible = False` and computes exact `total_overflow_depth` and `overflow_rate`.
2. **Never Drop Content**: The engine preserves 100% of blocks (`content_preservation = 1.0`) and never shrinks text below readable font thresholds (`min_font_size_pt`).
3. **Actionable Suggestions**: The diagnostic response surfaces actionable remediation options:
   - *Option 1 (Canvas Expansion)*: Increase canvas height from $1080\text{ px}$ to $1440\text{ px}$ (or switch to a 4-column landscape / 2-poster symposium presentation).
   - *Option 2 (Figure Sub-Sampling)*: Combine small thumbnail figures into composite multi-panel subfigures (e.g., $2 \times 2$ grid).
   - *Option 3 (Content Condensation)*: Truncate body paragraphs to reach target word-count thresholds.

---

## 7. Deliverables & Artifact Inventory

The Stage 3 implementation includes:
- **Core Engine**:
  - `src/schemas/layout_document.py`: Canonical editable JSON schema & COCO converter.
  - `src/adaptation/template_regions.py`: Hierarchical intermediate representation extractor.
  - `src/adaptation/content_geometry.py`: Typography and non-linear geometry estimator.
  - `src/adaptation/slot_assignment.py`: Semantic slot assignment and region splitting/merging.
  - `src/adaptation/layout_constraints.py`: SciPy HiGHS LP optimization engine.
  - `src/adaptation/adaptation_evaluation.py`: 9-dimension objective evaluation framework.
  - `src/adaptation/layout_adapter.py`: Top-level candidate adapter and selector.
  - `src/rendering/html_renderer.py`: Standalone HTML/CSS poster renderer.
  - `src/rendering/svg_renderer.py`: Pure vector SVG renderer.
  - `src/rendering/pdf_export.py`: Publication vector PDF exporter.
- **Demonstration & Verification**:
  - `notebooks/04_layout_adaptation_and_generation.ipynb`: Fully executed demonstration notebook with embedded base64 plots and interactive tables.
  - `src/test_stage3_pipeline.py`: Comprehensive 8-test unit and integration test suite.
  - `outputs/posters/`: 25 exported deliverables (`.html`, `.svg`, `.pdf`, `.png`, `.json`) across Scenarios A through E.
  - `reports/figures/stage3/`: 4-panel visual comparison figures and tri-paradigm benchmark charts.
  - `reports/adaptation/stage3_benchmark_metrics.csv`: Machine-readable evaluation data.
