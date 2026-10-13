# Infographix — EDA Work Log

**Contributor:** Rakshith Raghavendra  
**Project:** AI-Powered Technical Poster and Infographic Generator  
**Responsibility:** Exploratory Data Analysis (EDA)  
**Dataset:** SciPostLayout v2  
**Repository:** `rahul-girish/infographix`  
**Branch:** `eda/rakshith-worklog`  
**Status:** In progress

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

Installed versions at initial verification:

| Package | Version |
|---|---|
| Pandas | 3.0.6 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |

## 3. Repository Inspection

- [x] Reviewed `README.md` and `requirements.txt`.
- [x] Inspected the starter notebook `notebooks/01_initial_eda.ipynb`.
- [x] Inspected `.dvc/config` and `.dvcignore`.
- [x] Searched for DVC tracking and pipeline files; none were found within the searched locations.
- [x] Confirmed the original working tree was clean.
- [x] Created the dedicated branch `eda/rakshith-worklog` from `main`.

## 4. Dataset Acquisition

Dataset: SciPostLayout v2  
Source: https://huggingface.co/datasets/omron-sinicx/scipostlayout_v2

- [x] Reviewed the dataset description, file structure, and CC BY 4.0 license information.
- [x] Previewed the Hugging Face download; the archive was reported as approximately 32.2 GB.
- [x] Downloaded `scipostlayout.zip` to `data/raw/scipostlayout/`.
- [x] Tested the ZIP archive using `unzip -t`; no errors were reported.
- [x] Extracted the archive into `data/interim/scipostlayout/`.
- [x] Measured the extracted directory at approximately 33 GB.

## 5. Initial Dataset Inspection

Located the poster annotation files:

- `data/interim/scipostlayout/scipostlayout/poster/png/train.json`
- `data/interim/scipostlayout/scipostlayout/poster/png/dev.json`
- `data/interim/scipostlayout/scipostlayout/poster/png/test.json`

Also located paper metadata JSON files and poster PDF metadata.

- [x] Counted 17,256 files under the extracted `scipostlayout` directory.
- [x] Inspected the root structure of `train.json`.
- [x] Identified the top-level keys: `categories`, `images`, and `annotations`.

### Initial training-set counts

| Component | Count |
|---|---:|
| Categories | 9 |
| Images | 6,855 |
| Annotations | 168,313 |
| Average annotations per image | Approximately 24.55 |

The nine categories are Author Info, Title, Figure, Text, List, Section, Caption, Table, and Unknown.

These counts are initial structural findings, not a complete assessment of data quality.

## 6. Git and Data Management

- [x] Created the EDA branch `eda/rakshith-worklog`.
- [ ] Verify dataset exclusion from Git.
- [ ] Commit EDA work only on this branch.

## 7. EDA Progress Tracker

- [ ] Inspect annotation schema and image metadata.
- [ ] Compare train, dev and test splits.
- [ ] Validate annotations and image references.
- [ ] Check missing values, invalid records and duplicates.
- [ ] Analyze category frequencies and layout distributions.
- [ ] Generate visualizations.
- [ ] Document findings and limitations.
- [ ] Prepare the final EDA notebook and report.

