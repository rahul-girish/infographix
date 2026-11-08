# Infographix — Feature Engineering & Layout Modeling Architecture (v2.0)

**Version:** `2.0.0`  
**Dataset:** SciPostLayout v2 ($7,855$ posters, $192,697$ element annotations)  
**Source Modules:** [`src/features/`](file:///home/rahul/Projects/infographix/src/features/__init__.py), [`src/models/`](file:///home/rahul/Projects/infographix/src/models/__init__.py), [`src/registry/register_v2_models.py`](file:///home/rahul/Projects/infographix/src/registry/register_v2_models.py)  
**Notebooks:** [`notebooks/02_feature_engineering_v2.ipynb`](file:///home/rahul/Projects/infographix/notebooks/02_feature_engineering_v2.ipynb), [`notebooks/03_layout_model_comparison.ipynb`](file:///home/rahul/Projects/infographix/notebooks/03_layout_model_comparison.ipynb)

---

## 1. Executive Summary & Architectural Shift

In `v1.0.0`, Infographix represented each scientific poster as a single 92-column vector of category counts, median sizes, and standard deviations of center coordinates, followed by PCA ($92 \to 15$) and fixed-$\varepsilon$ DBSCAN clustering (`PosterLayoutDBSCAN`). While DBSCAN reported a test silhouette score of $0.2486$, empirical and visual validation revealed a fatal flaw: **94.6% of all held-out test posters ($473 / 500$) collapsed into a single cluster (`Cluster 0`)**, with normalized cluster size entropy of just $0.0393$, orientation purity of $0.6884$ (matching the base rate of portrait posters), and column-count purity of $0.5621$.

The root cause was twofold:
1. **Lossy Feature Representation:** Reducing an entire poster to category medians and standard deviations erases multi-modal spatial structure (e.g., two vertical text columns have the exact same median $x$-center of $0.50$ as a single centered text block).
2. **Misaligned Modeling Paradigm:** Scientific poster layouts form a continuous manifold across text and figure densities rather than isolated density islands separated by empty space. Furthermore, the ultimate product objective is **conditional layout retrieval, adaptation, and generation** from a user's content specification (`PosterRequest`)—not assigning posters to a handful of arbitrary clusters.

### The v2 Solution
1. **Multi-Table Canonical Representation (`data/processed/features/v2/`):**
   - `posters.parquet` ($7,855$ rows $\times$ $295$ columns — **zero posters dropped**).
   - `elements.parquet` ($192,657$ rows $\times$ $44$ columns — preserving every valid bounding box).
   - `relations.parquet` ($676,383$ rows $\times$ $27$ columns — sparse spatial relationship graph).
   - `rejected_annotations.parquet` ($40$ rows $\times$ $11$ columns — deterministic quality log with explicit rejection reasons).
   - `appearance_sample.parquet` ($180$ rows $\times$ $18$ columns — decoupled pixel-derived visual style features).
2. **Runtime User Content-Demand Bridge (`src/features/request_schema.py`):**
   - Separates `semantic_role` (e.g., `abstract`, `methods`, `results`, `logo`) from `visual_type` (`title`, `author_info`, `section`, `text`, `list`, `figure`, `table`, `caption`).
   - Uses a typography-aware text wrapping estimator (`estimate_rendered_text_geometry`) to convert word/character counts into width-dependent height and area demands.
3. **Two-Pronged Production Model (`v2.0.0` registered in `ModelRegistry`):**
   - **Primary Engine — `poster_layout_retriever` (`LayoutRetriever`):** Group-normalized, constraint-aware nearest-neighbour retrieval with Maximal Marginal Relevance (MMR) diversity reranking. Achieves **100% orientation match**, **97.6% exact top-1 column match**, and **0.9727 spatial occupancy similarity** on held-out test queries.
   - **Macro Taxonomy — `poster_layout_archetypes` (`OrientationConditionedHierarchical`):** Partitions Portrait and Landscape canvases first, then applies Ward-linkage hierarchical clustering ($4$ Landscape + $3$ Portrait = $7$ archetypes), increasing cluster balance entropy from **$0.0393 \to 0.8830$** and orientation purity from **$0.6884 \to 0.9580$** ($100\%$ Portrait vs. Non-Portrait separation).

---

## 2. Audit Corrections from v1 to v2

| v1 Issue | v2 Engineering Fix |
| :--- | :--- |
| **`element_density` duplicated `element_count`** | Removed `element_density`. Added rasterized `union_box_coverage` ($\in [0, 1]$) on a $32\times 32$ normalized grid and explicit $4\times 4$ spatial occupancy grid cells (`occ_grid_r0..3_c0..3`). |
| **`sum_bbox_area_ratio` conflated with occupied area** | Renamed to `sum_box_area_fraction` (which double-counts overlapping annotations) and added `union_box_coverage`, `overlap_excess = max(0, sum_box_area_fraction - union_box_coverage)`, and `unannotated_whitespace_fraction = 1 - union_box_coverage`. |
| **Zero-filled geometry for absent categories** | Added explicit binary presence indicators (`has_<category>`) for all 8 known categories. Absent categories leave raw quantiles as `NaN` in `posters.parquet`, imputed strictly from training medians during matrix preprocessing while retaining `has_<category>`. |
| **Dropped 2 training posters with 1 element** | Zero posters are dropped in v2 ($6,855$ train, $500$ dev, $500$ test all retained). Single-element groups receive zero spread/IQR deterministically. |
| **Raw `pixel_width`, `pixel_height` used in models** | Stored strictly as non-feature metadata (`canvas_width_px`, `canvas_height_px`) for rendering and pixel cropping; excluded from all model feature schemas. |

---

## 3. Comprehensive Feature Specification & Documentation

In accordance with Requirement 3D & 4, every engineered feature group is documented below across five dimensions: **(1) What it measures**, **(2) How it is calculated**, **(3) Why it is useful for poster generation**, **(4) Availability during inference from user input**, and **(5) Limitations**.

### Group A: Poster & Canvas Features (`posters.parquet`)

| Feature Subgroup | 1. What It Measures | 2. How It Is Calculated | 3. Usefulness for Poster Generation | 4. Inference Availability from User Input | 5. Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A1. Canvas Geometry** (`aspect_ratio`, `log_aspect_ratio`, `orientation`, `is_landscape`, `is_portrait`, `known_element_count`, `unknown_element_fraction`) | Canvas shape, orientation category (`portrait` $<0.95$, `landscape` $>1.05$, `square` $[0.95, 1.05]$), and annotation count/reliability. | $W / H$, $\ln(W/H + \epsilon)$, and counts of valid vs. `Unknown` elements. | Hard constraint for template selection; $\ln(W/H)$ makes portrait ($0.707$) and landscape ($1.414$) symmetric around $0$. | **Directly available:** User selects target poster dimensions/orientation, and block count is known from submitted content. | Pixel dimensions reflect export DPI rather than physical inches; only aspect ratio is physically invariant. |
| **A2. Category Inventory** (`count_<cat>`, `log1p_count_<cat>`, `has_<cat>`, `fraction_<cat>`, `visual_element_count`, `textual_element_count`, `heading_element_count`, `media_to_text_count_ratio`) | Exact and log-scaled inventory of each semantic element type (`title`, `author_info`, `section`, `text`, `list`, `figure`, `table`, `caption`). | Grouped counts per poster; `fraction_<cat>` divides by `known_element_count`. | Matches a user's content inventory (e.g., 4 figures, 2 tables, 6 text blocks) to reference posters with compatible region slots. | **Directly available:** Computed directly from the user's submitted content blocks via `VISUAL_TYPE_TO_SCIPOST`. | One user section may map to multiple annotated text boxes, or several short paragraphs may merge into one box during adaptation. |
| **A3 & A4. Box-Size & Category Quantiles** (`w_norm_p10..p90`, `h_norm_p10..p90`, `area_norm_p10..p90`, `area_sum_<cat>`, `{w,h,area,cx,cy}_p25..p75_<cat>`, `visual_area_sum`, `textual_area_sum`, `visual_to_body_area_ratio`) | Non-parametric distribution of normalized element widths, heights, areas, and total area capacity allocated to visual vs. textual content. | Empirical percentiles ($10, 25, 50, 75, 90$) and category area sums over normalized boxes. | Distinguishes posters with many small figures from posters dominated by one or two large central diagrams or wide tables. | **Estimated at runtime:** Derived from user content via typography line-wrapping (`estimate_rendered_text_geometry`), image aspect ratios, and table row/col dimensions (`cx, cy` quantiles are **candidate-only**, not in user request). | SciPostLayout annotations do not store true word counts, so dataset `textual_area_sum` reflects bounding-box area rather than exact character count. |
| **A5. Content Extent & Margins** (`content_{left,right,top,bottom}_margin`, `content_extent_{width,height,area}`, `header_band_bottom_y`) | Outer bounding envelope of all annotated elements and vertical extent of the top title/author header band. | $\min(x_0)$, $1-\max(x_1)$, $\min(y_0)$, $1-\max(y_1)$, and $\max(y_1)$ among header categories with $c_y < 0.35$. | Guides canvas padding, header banner allocation, and usable body area during layout synthesis. | **Candidate-only (or user style default):** Used to evaluate candidate templates and enforce margin constraints during placement. | Unannotated decorative banners, background borders, or institutional logos outside annotations are not captured. |
| **A6. Coverage & Overlap** (`sum_box_area_fraction`, `union_box_coverage`, `overlap_excess`, `unannotated_whitespace_fraction`, `overlap_pair_count`, `max_pair_iou`, `mean_intersection_over_smaller_box`) | True fraction of canvas covered by the union of boxes vs. summed individual box areas and pairwise overlap severity. | $32\times 32$ binary rasterization for union coverage; exact interval intersection for pairwise IoU and intersection-over-minimum-area. | Prevents recommending overcrowded or heavily overlapping layouts; quantifies available whitespace breathing room. | **Partially estimated at runtime:** `sum_box_area_fraction` is estimated from total content demand; overlap features are **candidate-only** quality indicators. | Overlapping boxes in SciPostLayout are sometimes intentional (e.g., a `Caption` or `Section` nested inside a larger `Figure` or panel). |

---

### Group B: Element-Level Features (`elements.parquet`)

| Feature Subgroup | 1. What It Measures | 2. How It Is Calculated | 3. Usefulness for Poster Generation | 4. Inference Availability from User Input | 5. Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1 & B2. Normalized & Log Geometry** (`x0, y0, x1, y1, w_norm, h_norm, cx, cy, area_norm, bbox_aspect_ratio`, `log_{w,h,area}_norm`, `log_bbox_aspect_ratio`) | Exact canvas-normalized position, size, and aspect ratio of every individual region ($192,657$ rows). | Divided by canvas $(W, H)$; natural log applied with $\epsilon = 10^{-6}$ to stabilize skewed size distributions. | Serves as the direct regression target $(x_0, y_0, x_1, y_1)$ for conditional layout generators and region-slot capacity for template adaptation. | **Target / Partially available:** User provides figure `bbox_aspect_ratio` and estimated $(w, h)$ demand; positions $(x_0, y_0, c_x, c_y)$ are the **prediction targets**. | Annotation bounding boxes occasionally have minor padding differences across human annotators. |
| **B3. Edge & Margin Distances** (`left_margin, right_margin, top_margin, bottom_margin, min_horizontal_margin, min_vertical_margin, center_dist_to_edge`) | Proximity of each element to the four canvas borders. | $x_0$, $1-x_1$, $y_0$, $1-y_1$, and their pairwise minima. | Captures whether an element is peripheral/flush to a border (e.g., header, footer, outer column) or interior. | **Candidate / Target feature:** Used to assign user blocks to candidate template slots and evaluate boundary alignment. | Does not account for unannotated background frames. |
| **B4 & B5. Relative Hierarchy & Position Bins** (`poster_area_rank_pct`, `category_area_rank_pct`, `ratio_to_poster_median_area`, `is_top3_largest`, `ratio_{w,h,area}_to_train_cat_median`, `horiz_zone`, `vert_zone`, `grid_4x4_cell`) | Visual dominance of an element within its poster and relative to typical training elements of the same category (`train` split only). | Within-poster percentile ranks; division by training-set category medians; discretization into $3\times 4$ zones and $4\times 4$ grid cells ($0..15$). | Allows matching high-priority user content (`priority > 1.0`, hero figures, main results) to the most prominent regions of a candidate layout. | **Available via user priority & estimated size:** User block priority and estimated area map directly to relative size ranks. | Quantized grid bins (`grid_4x4_cell`) are coarse descriptors and must always be paired with continuous coordinates. |

---

### Group C: Pairwise Spatial Relationship Features (`relations.parquet`)

| Feature Subgroup | 1. What It Measures | 2. How It Is Calculated | 3. Usefulness for Poster Generation | 4. Inference Availability from User Input | 5. Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1 & C2. Pairwise Offsets, Gaps, IoU & Alignment** (`dx, dy, abs_dx, abs_dy, distance, horizontal_gap, vertical_gap, iou, intersection_over_min_area, contains_target, area_ratio, aligned_{left,right,top,bottom,cx,cy}, same_column_hypothesis, same_row_hypothesis, direction, edge_types`) | Directed geometric relationships across $676,383$ sparse graph edges (`knn`, `overlap`, `nearest_below_col`, `nearest_right_row`, `figure_caption_hypothesis`). | Vectorized $N\times N$ interval arithmetic per poster with alignment tolerance $\tau = 0.015$ and column/row interval overlap $\ge 0.50$. | Essential edge attributes for Graph Neural Networks (GNNs), constraint-based layout repair, reading-order inference, and keeping figures paired with captions. | **Partially specified by user constraints:** User provides logical adjacency constraints (`must_be_adjacent_to`, figure-caption links, section ordering); geometric offsets are learned/optimized. | Geometric proximity between a `Figure` and `Caption` is a spatial hypothesis (`figure_caption_hypothesis`), not guaranteed semantic ground truth without OCR text matching. |

---

### Group D: Global Composition & Structure Features (`posters.parquet`)

| Feature Subgroup | 1. What It Measures | 2. How It Is Calculated | 3. Usefulness for Poster Generation | 4. Inference Availability from User Input | 5. Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1 & D2. $4\times 4$ Occupancy Grids & 16-Bin Profiles** (`occ_grid_r0..3_c0..3`, `horiz_occ_00..15`, `vert_occ_00..15`, `text_occ_{top,bottom}_half`, `visual_occ_{left,right,center}`) | Spatial distribution of content across canvas quadrants, horizontal slices, and vertical slices. | Downsampling the $32\times 32$ rasterized canvas union mask (and category-group masks) into $4\times 4$ cells and $16$-bin 1D projections. | Enables fast spatial similarity comparison and Maximal Marginal Relevance (MMR) diversity reranking so top-$k$ retrieved layouts are visually distinct. | **Candidate-only representation:** Used for MMR diversification among retrieved candidates and for evaluating spatial consistency on held-out queries. | Coarse $4\times 4$ grids summarize macro mass distribution rather    than fine element boundaries. |
| **D3. Column & Row Detection** (`estimated_column_count`, `column_assignment_confidence`, `estimated_gutter_fraction`, `estimated_row_band_count`, `full_width_region_count`, `multi_column_span_count`) | Number of vertical reading columns ($1..6$), peak-to-valley gutter contrast confidence, horizontal row bands, and spanning banners. | `scipy.signal.find_peaks` (height $\ge 0.25$, prominence $\ge 0.08$, distance $\ge 4$) on the smoothed 32-bin horizontal body occupancy profile. | Crucial for distinguishing **2-column portrait** from **3-column** and **4-column landscape** layouts. | **Available / Inferred from User Request:** User can specify `preferred_column_count` (or it is inferred from orientation and block count). | Freeform, radial, or highly asymmetric infographic posters have lower `column_assignment_confidence` ($\le 0.45$) because they do not follow a strict vertical grid. |
| **D4 & D5. Alignment Regularity, Balance & Symmetry** (`frac_aligned_{left,right,top,bottom,cx,cy}`, `left_right_area_balance`, `top_bottom_area_balance`, `horiz_profile_symmetry`, `vert_profile_symmetry`, `center_of_mass_{x,y}`) | Grid discipline (fraction of element pairs aligned to common axes) and visual weight symmetry across left/right and top/bottom halves. | Pairwise edge-alignment fractions within $\tau=0.015$; cosine similarity between left and reversed-right 16-bin profiles; area-weighted centroids. | Separates formal, grid-aligned academic templates from asymmetric compositions. | **Candidate style filter:** Can be matched against user `style_preferences` (e.g., formal symmetric vs. asymmetric modern). | High symmetry is a stylistic property, not an intrinsic measure of aesthetic quality. |

---

### Group E: Visual Appearance & Design Style Features (`appearance_sample.parquet`)

| Feature Subgroup | 1. What It Measures | 2. How It Is Calculated | 3. Usefulness for Poster Generation | 4. Inference Availability from User Input | 5. Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1 & E2. Pixel Luminance, Colorfulness, Contrast & Entropy** (`mean_luminance`, `mean_saturation`, `colorfulness_index`, `bg_{r,g,b,luminance}`, `is_dark_background`, `edge_density`, `local_contrast`, `visual_entropy`, `text_region_contrast`, `figure_region_colorfulness`) | Background color (from outer 4% border strip), Hasler-Süsstrunk colorfulness, Sobel gradient edge density, $8\times 8$ block RMS contrast, Shannon entropy, and text vs. figure crop contrast. | Extracted from $256\times 256$ RGB poster images and normalized bounding-box crops in `src/features/visual_features.py`. | Supports optional style-aware filtering (e.g., dark vs. light background, minimalist vs. high-colorfulness templates). | **Optional user style preference only:** Kept strictly separate from structural models so layout retrieval never depends on raster pixels. | Influenced by figure raster content and export compression; does not include OCR typography hierarchy until an OCR pass is run. |

---

## 4. Experimental Comparison of Modeling Strategies

All models were trained strictly on `train` ($N=6,855$) and evaluated on held-out `test` ($N=500$).

### 4.1 Clustering & Structural Archetype Discovery Results

| Model | Feature Set | Train Silhouette | Test Silhouette | Test Davies-Bouldin $\downarrow$ | Clusters | Test Noise % | Largest Cluster Ratio $\downarrow$ | Cluster Balance Entropy $\uparrow$ | Orientation Purity $\uparrow$ | Column Purity $\uparrow$ | Column MAE $\downarrow$ | $4\times 4$ Occupancy Consistency $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=3$, Baseline)** | v1 (92 stats) | 0.0676 | 0.0549 | 3.1464 | 3 | 0.0% | 0.4360 | 0.9489 | 0.6860 | 0.5480 | 0.4980 | 0.9680 |
| **DBSCAN (`PCA-15`, v1.0.0 Prod)** | v1 (92 stats) | 0.2405 | **0.2486** | **1.0222** | 2 | 5.0% | **0.9460** | **0.0393** | 0.6884 | 0.5621 | 0.4821 | 0.9711 |
| **HDBSCAN (Varying Density)** | v2 (34 macro) | 0.1531 | 0.1339 | 2.2201 | 6 | 7.8% | 0.4360 | 0.8418 | **1.0000** | **0.9675** | **0.0325** | **0.9733** |
| **GMM ($k=7$, BIC + 3% Outlier)** | v2 (34 macro) | 0.0780 | 0.0786 | 2.6807 | 7 | 1.2% | **0.2220** | **0.9761** | 0.9514 | 0.8664 | 0.1538 | 0.9731 |
| **Ward Hierarchical ($4\text{L}+3\text{P}$, v2.0.0)** | v2 (34 macro) | 0.1190 | 0.1523 | 1.7185 | 7 | 0.0% | 0.3940 | 0.8830 | **0.9580** | 0.7200 | 0.2840 | 0.9690 |

#### Key Modeling Takeaways
1. **Why Silhouette Score Alone Was Misleading:** `DBSCAN (v1.0.0)` achieved the highest silhouette score ($0.2486$) by placing $94.6\%$ of all test posters into a single cluster (`Cluster Balance Entropy = 0.0393`), failing to separate Portrait from Landscape ($0.6884$ purity = base rate) or 2-column from 3-column layouts ($0.5621$ purity).
2. **Varying-Density HDBSCAN on v2 Features:** Switching to the v2 structural representation with `HDBSCAN` immediately resolves the single-cluster collapse, discovering $6$ clusters with **100% orientation purity** and **96.8% column-count purity** (`Column MAE = 0.0325`) while isolating $7.8\%$ outliers.
3. **GMM & Orientation-Conditioned Ward Hierarchical:**
   - **GMM ($k=7$)** achieves the most uniform cluster distribution (`Cluster Balance Entropy = 0.9761`, largest cluster only $22.2\%$) and provides soft probability distributions over archetypes.
   - **Orientation-Conditioned Ward Hierarchical ($4$ Landscape + $3$ Portrait = $7$ archetypes)** guarantees $100\%$ separation between Portrait and Non-Portrait canvases (`0.9580` 3-class purity including `square`), strong geometric compactness (`Test Silhouette = 0.1523`, `Davies-Bouldin = 1.7185`), and clear, interpretable medoid templates:
     - **Archetype 0:** 3-Column Landscape, Figure-Balanced ($n=59$ test)
     - **Archetype 1:** 2-Column Landscape, Text/Section-Rich ($n=86$ test)
     - **Archetype 2:** 3-Column Landscape, High Visual/Table Density ($n=7$ test)
     - **Archetype 3:** 1-to-2 Column Wide-Banner / Asymmetric Landscape ($n=5$ test)
     - **Archetype 4:** 2-Column Portrait, Standard Academic ($n=197$ test)
     - **Archetype 5:** 3-Column Portrait, Dense Multi-Section ($n=88$ test)
     - **Archetype 6:** 1-to-2 Column Portrait, Large Hero-Figure Dominant ($n=58$ test)

---

### 4.2 Direct Nearest-Neighbour Layout Retrieval & Feature Ablation (`N = 500` Test Queries)

To evaluate the actual application workflow, we hid the spatial coordinates of all $500$ test posters, converted their content inventories and approximate block sizes into runtime `PosterRequest` objects (`simulate_request_from_heldout_poster`), and retrieved top-$5$ training layouts:

| Retrieval Configuration | Orientation Match @ 5 $\uparrow$ | Exact Column Match @ 1 $\uparrow$ | Column MAE @ 5 $\downarrow$ | Category Count L1 Error @ 5 $\downarrow$ | Inventory Cosine Sim @ 5 $\uparrow$ | Capacity Area L1 Error @ 5 $\downarrow$ | Hidden $4\times 4$ Occupancy Sim @ 1 $\uparrow$ | Top-5 Spatial Diversity $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Canvas Only** | 1.0000 | 0.6980 | 0.4292 | 11.0580 | 0.8706 | 0.1602 | 0.9709 | 0.0297 |
| **2. Canvas + Category Inventory** | 1.0000 | 0.7820 | 0.3072 | **4.2696** | **0.9758** | 0.1341 | 0.9701 | 0.0313 |
| **3. Canvas + Inventory + Area Capacity** | 1.0000 | 0.8180 | 0.3048 | 5.2768 | 0.9654 | **0.1019** | 0.9726 | 0.0285 |
| **4. Full Structural (+Columns, No MMR)** | 1.0000 | **0.9760** | **0.0380** | 5.6832 | 0.9608 | 0.1071 | **0.9727** | 0.0293 |
| **5. Full Structural + MMR Diversity (`v2.0.0` Prod)** | **1.0000** | **0.9760** | 0.0544 | 5.6552 | 0.9613 | **0.1071** | **0.9727** | **0.0315** |

Every incremental feature group demonstrably improves its target objective:
- Adding **Category Inventory** cuts element count L1 mismatch by **61%** ($11.06 \to 4.27$).
- Adding **Area Capacity** cuts visual/textual area capacity mismatch by **36%** ($0.1602 \to 0.1019$).
- Adding **Structural Column Features** boosts exact top-1 column match from **$81.8\% \to 97.6\%$** and reduces column MAE by **88%** ($0.3048 \to 0.0380$).
- Adding **MMR Diversity Reranking ($\lambda=0.78$)** preserves top-1 exact column accuracy ($97.6\%$) and spatial occupancy similarity ($0.9727$) while increasing the structural variety among the top-$5$ candidate templates presented to the user.

---

## 5. Known Dataset Limitations & Roadmap to Conditional Layout Generation

### 5.1 What SciPostLayout v2 Cannot Directly Supervise
1. **Absence of True Text Lengths:** SciPostLayout v2 provides bounding boxes and categories, not the raw text strings or font sizes inside each box. Therefore, mapping user word counts to box sizes must rely on deterministic font-metric wrapping (`estimate_rendered_text_geometry`) at runtime until an OCR / PDF-text extraction pass is executed and audited on the training images.
2. **Absence of Semantic Grouping & Reading Order:** Pairwise edges in `relations.parquet` (`figure_caption_hypothesis`, `same_column_hypothesis`) are geometric hypotheses. At inference time, explicit user-supplied links (`must_be_adjacent_to`, section-to-paragraph hierarchy) should take precedence over purely geometric heuristics.
3. **Absence of Aesthetic Preference Labels:** Dataset annotations reflect existing academic posters (which vary in design quality), not human preference rankings.

### 5.2 Staged Progression Toward Conditional Layout Generation
The v2 multi-table schema is purposefully engineered so Infographix can progress cleanly through four stages:
1. **Stage 1 (Implemented & Production-Registered in `v2.0.0`):** Constraint-aware, group-normalized nearest-neighbour layout retrieval (`LayoutRetriever`) + 7-archetype structural taxonomy (`OrientationConditionedHierarchical`).
2. **Stage 2 (Layout Adaptation & Constraint Solver):** Given the top retrieved reference layout (`elements.parquet` + `relations.parquet`) and a user's `PosterRequest`, solve a bipartite matching + quadratic/linear box-refinement program that resizes column heights to fit wrapped text lines, locks figure aspect ratios, and preserves column alignment (`aligned_left`, `aligned_right`) and gutter spacing.
3. **Stage 3 (Learned Candidate Ranker):** Log user template selections and canvas edits in the frontend editor (`convert_canvas_to_coco`), then train a gradient-boosted ranker (`LightGBM` / `CatBoost`) on the 12 compatibility features from `compute_request_candidate_compatibility()`.
4. **Stage 4 (Retrieval-Augmented Conditional Set/Graph Layout Generator):** Train a Set Transformer or Relational Graph Attention Network (RGAT) consuming `elements.parquet` nodes and `relations.parquet` edges conditioned on retrieved reference layouts to predict adapted $(x_0, y_0, w_{\text{norm}}, h_{\text{norm}})$ coordinates directly.
