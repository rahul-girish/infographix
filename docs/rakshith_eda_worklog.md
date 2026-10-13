# Infographix — EDA Work Log

**Contributor:** Rakshith Raghavendra
**Project:** AI-Powered Technical Poster and Infographic Generator
**Responsibility:** Exploratory Data Analysis (EDA)
**Dataset:** SciPostLayout v2
**Repository:** `rahul-girish/infographix`
**Branch:** `eda/rakshith-worklog`
**Status:** EDA notebook completed; final handover pending

---

## 1. Objective

Understand the structure, quality, and characteristics of the SciPostLayout v2 dataset. Identify useful layout features, data-quality issues, and patterns that can inform preprocessing, feature engineering, and baseline modelling for the Infographix project.

## 2. Environment Setup

- [x] Installed Ubuntu 24.04 through WSL2.
- [x] Verified Python 3.12.3.
- [x] Installed `uv` and DVC.
- [x] Cloned the Infographix repository.
- [x] Created and activated the `.venv` virtual environment.
- [x] Installed dependencies from the existing `requirements.txt`.
- [x] Installed and verified Pandas, NumPy, and Matplotlib.
- [x] Installed JupyterLab for notebook-based analysis.

Installed versions at initial verification:

| Package    | Version |
| ---------- | ------- |
| Python     | 3.12.3  |
| Pandas     | 3.0.6   |
| NumPy      | 2.5.4   |
| Matplotlib | 3.11.2  |

## 3. Repository Inspection

- [x] Reviewed `README.md` and `requirements.txt`.
- [x] Inspected the starter notebook `notebooks/01_initial_eda.ipynb`.
- [x] Inspected `.dvc/config` and `.dvcignore`.
- [x] Searched for DVC tracking and pipeline files; none were found within the searched locations.
- [x] Confirmed the original working tree was clean.
- [x] Created the dedicated branch `eda/rakshith-worklog` from `main`.

## 4. Dataset Acquisition

**Dataset:** SciPostLayout v2
**Source:** [https://huggingface.co/datasets/omron-sinicx/scipostlayout_v2](https://huggingface.co/datasets/omron-sinicx/scipostlayout_v2)

- [x] Reviewed the dataset description, file structure, and CC BY 4.0 license information.
- [x] Previewed the Hugging Face download; the archive was reported as approximately 32.2 GB.
- [x] Downloaded `scipostlayout.zip` to `data/raw/scipostlayout/`.
- [x] Tested the ZIP archive using `unzip -t`; no errors were reported.
- [x] Extracted the archive into `data/interim/scipostlayout/`.
- [x] Measured the extracted directory at approximately 33 GB.
- [x] Located the poster image files and corresponding annotation JSON files for all three splits.

## 5. Initial Dataset Inspection

Located the poster annotation files:

- `data/interim/scipostlayout/scipostlayout/poster/png/train.json`
- `data/interim/scipostlayout/scipostlayout/poster/png/dev.json`
- `data/interim/scipostlayout/scipostlayout/poster/png/test.json`

Also located paper metadata JSON files and poster PDF metadata.

- [x] Counted 17,256 files under the extracted `scipostlayout` directory.
- [x] Inspected the root structure of `train.json`.
- [x] Identified the top-level keys: `categories`, `images`, and `annotations`.
- [x] Inspected image metadata and annotation fields.
- [x] Identified the nine layout categories.

### Dataset Split Summary

| Split     |    Images | Annotations | Average annotations per image |
| --------- | --------: | ----------: | ----------------------------: |
| Train     |     6,855 |     168,313 |                         24.55 |
| Dev       |       500 |      12,141 |                         24.28 |
| Test      |       500 |      12,243 |                         24.49 |
| **Total** | **7,855** | **192,697** |                     **24.53** |

The nine categories are Author Info, Title, Figure, Text, List, Section, Caption, Table, and Unknown.

## 6. Data Integrity and Quality Checks

- [x] Checked for duplicate image IDs within each split.
- [x] Checked for duplicate annotation IDs within each split.
- [x] Checked for image ID and annotation ID overlap across splits.
- [x] Checked whether annotations referenced missing image IDs.
- [x] Checked for duplicate filenames within each split.
- [x] Checked for missing required fields in image and annotation records.
- [x] Verified the existence of referenced poster image files.
- [x] Examined bounding boxes for negative coordinates, non-positive dimensions, and image-boundary violations.

### Bounding-Box Validation Findings

The existing dataset loader's category and bounding-box validation rules produced the following results:

| Split     | Total annotations | Valid annotations | Rejected annotations |
| --------- | ----------------: | ----------------: | -------------------: |
| Train     |           168,313 |           168,220 |                   93 |
| Dev       |            12,141 |            12,136 |                    5 |
| Test      |            12,243 |            12,241 |                    2 |
| **Total** |       **192,697** |       **192,597** |              **100** |

All 100 rejected annotations were rejected by the bounding-box validation rules; no annotations were rejected due to an unsupported category.

The rejected annotations represent approximately 0.052% of all annotations.

These results describe annotation-level validation. They do not quantify every image-level filtering decision that the loader may make.

## 7. Category Distribution Analysis

- [x] Calculated category frequencies for train, dev, and test.
- [x] Visualized training-set category counts.
- [x] Compared category percentages across the three splits.
- [x] Reviewed the distribution of rare categories.

### Main Findings

- Text is the most frequent category in all three splits.
- Section is the second most frequent category, followed by Figure.
- The relative category distributions are broadly similar across train, dev, and test.
- Unknown annotations are extremely rare.

The category distribution comparison is descriptive; no formal statistical test was performed.

## 8. Image Dimensions and Orientation

- [x] Calculated minimum, median, and maximum image dimensions.
- [x] Calculated image aspect ratios.
- [x] Grouped images into portrait, square, and landscape orientations using the thresholds defined in the notebook.
- [x] Compared orientation counts across the three splits.
- [x] Visualized the orientation distribution.

### Main Findings

- Approximately 64.6% of training images are portrait-oriented under the selected thresholds.
- The median image aspect ratio is approximately 0.75 in the training and development splits.
- The test split has a median aspect ratio of approximately 0.75 as well.
- Image dimensions vary substantially, which is relevant when designing resizing and preprocessing strategies.

## 9. Annotation Density

- [x] Calculated the minimum, median, mean, and maximum annotation counts per image.
- [x] Counted images with zero annotations.
- [x] Counted images containing more than 50 annotations.

### Main Findings

| Split | Median annotations per image | Mean annotations per image | Images with zero annotations | Images with more than 50 annotations |
| ----- | ---------------------------: | -------------------------: | ---------------------------: | -----------------------------------: |
| Train |                           23 |                      24.55 |                            0 |                                  117 |
| Dev   |                           23 |                      24.28 |                            0 |                                    4 |
| Test  |                           23 |                      24.49 |                            0 |                                   14 |

Most images contain a similar number of annotations, although a smaller subset contains more than 50 annotations.

## 10. Bounding-Box Statistics

- [x] Calculated bounding-box area relative to the corresponding image area.
- [x] Summarized the 25th percentile, median, 75th percentile, 95th percentile, and mean relative box area.
- [x] Excluded invalid or out-of-bounds boxes from the relative-area calculation.

### Main Findings

| Split | Valid boxes | Median relative box area | 95th percentile relative box area |
| ----- | ----------: | -----------------------: | --------------------------------: |
| Train |     168,220 |                   1.428% |                            8.291% |
| Dev   |      12,136 |                   1.486% |                            8.607% |
| Test  |      12,241 |                   1.442% |                            8.577% |

Individual layout elements typically occupy a small fraction of the full poster area. These percentages describe individual bounding boxes and should not be interpreted as total poster coverage.

## 11. Visual Inspection of Annotations

- [x] Visualized a training poster with bounding boxes and category labels.
- [x] Visualized three sampled training posters with bounding-box overlays.
- [x] Checked the visual alignment of sampled annotations against the poster content.

Visual inspection was performed on selected examples rather than on every image in the dataset. It provides a qualitative check of the annotations but does not establish that all annotations are correct.

## 12. Key Findings and Limitations

### Key Findings

1. SciPostLayout v2 contains 7,855 poster images and 192,697 annotations across three splits.
2. The dataset defines nine layout categories.
3. Text, Section, and Figure are the most frequent categories.
4. Category distributions are broadly similar across the three splits.
5. Most training images are portrait-oriented under the chosen thresholds.
6. The median annotation count is 23 per image in every split.
7. Bounding-box validation rejected 100 annotations, a small fraction of the full annotation set.
8. Relative bounding-box area statistics are broadly similar across the splits.

### Limitations

- Visual inspection was sample-based rather than exhaustive.
- Category-distribution similarity was assessed descriptively, without formal statistical testing.
- Bounding-box rejection counts are annotation-level results and do not capture every image-level filtering decision.
- Relative bounding-box area describes individual boxes, not the total area covered by all poster elements.
- The analysis does not establish model performance or prove that any preprocessing strategy improves the downstream model.
- Content-level duplicate images were not established by the image-ID and filename checks.

## 13. Deliverables and Git Status

### Deliverable

- **EDA notebook:** `notebooks/01_initial_eda.ipynb`
- **Notebook commit:** `3c3b098`
- **Commit message:** `feat: complete initial SciPostLayout EDA`
- **Working branch:** `eda/rakshith-worklog`

### Verification Completed

- [x] Executed all 14 notebook cells manually without reported errors.
- [x] Validated the notebook structure using `nbformat`.
- [x] Checked the Git diff for whitespace errors.
- [x] Committed the EDA notebook on the dedicated branch.

### Work Log Status

The initial EDA analysis and notebook are complete. The notebook is committed on `eda/rakshith-worklog`.
