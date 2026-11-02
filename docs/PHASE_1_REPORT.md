# SemiMind AI – Phase 1 Report

**Generated:** 2026-11-02  
**Status:** Complete

---

## 1. Files Inspected

| File | Size | Location |
|------|------|----------|
| `ai4i2020.csv` | 522 KB | `ai4i/ai4i2020.csv` (original) / `datasets/raw/ai4i/` |
| `secom.data` | 5.39 MB | `secom/secom.data` (original) / `datasets/raw/secom/` |
| `secom_labels.data` | 40 KB | `secom/secom_labels.data` (original) / `datasets/raw/secom/` |
| `secom.names` | 4 KB | Dataset documentation / `datasets/raw/secom/` |

---

## 2. Dataset Formats

### AI4I 2020 Predictive Maintenance

| Property | Value |
|----------|-------|
| File type | CSV |
| Delimiter | Comma (`,`) |
| Header row | Yes (row 0) |
| Encoding | UTF-8 |
| Rows | 10,000 |
| Columns | 14 |

### SECOM Semiconductor Manufacturing

| Property | Value |
|----------|-------|
| Feature file type | Space-delimited text |
| Delimiter | Single space (` `) |
| Header row | None |
| Encoding | UTF-8 |
| Missing value marker | `NaN` (MATLAB-style) |
| Feature rows | 1,567 |
| Feature columns | 590 |
| Labels file | Separate (`secom_labels.data`) |
| Label values | `-1` = Pass, `1` = Fail |
| Label format | `<label> "<timestamp>"` per row |

---

## 3. Dataset Dimensions

| Dataset | Rows | Columns | Size (MB) |
|---------|------|---------|-----------|
| AI4I features | 10,000 | 14 | 0.52 |
| SECOM features | 1,567 | 590 | 5.39 |
| SECOM labels | 1,567 | 2 | 0.04 |

---

## 4. Columns and Data Types

### AI4I Columns

| Column | dtype | Role |
|--------|-------|------|
| UDI | int64 | Identifier |
| Product ID | object | Identifier |
| Type | object | Categorical (L / M / H) |
| Air temperature [K] | float64 | Sensor feature |
| Process temperature [K] | float64 | Sensor feature |
| Rotational speed [rpm] | int64 | Sensor feature |
| Torque [Nm] | float64 | Sensor feature |
| Tool wear [min] | int64 | Sensor feature |
| Machine failure | int64 | **Target** (0 / 1) |
| TWF | int64 | Failure type indicator |
| HDF | int64 | Failure type indicator |
| PWF | int64 | Failure type indicator |
| OSF | int64 | Failure type indicator |
| RNF | int64 | Failure type indicator |

### SECOM Columns

- **Feature columns:** `feature_001` ... `feature_590` — all `float64`
  (auto-generated names; source file has no header)
- **Label columns:** `label` (int64), `timestamp` (object/str)

---

## 5. Missing Value Statistics

### AI4I
- Total missing cells: **0**
- All 14 columns are complete.
- No imputation required for this dataset.

### SECOM Features
- Total missing cells: **41,951**
- Total cells: 925,130 (1,567 × 590)
- Overall missing rate: **4.54%**
- Columns with any missing: **538 of 590** (91.2%)
- Columns fully missing (100%): some exist — flagged as constant/empty

### SECOM Labels
- Total missing: **0** — labels file is complete and aligned.

---

## 6. Duplicate Statistics

### AI4I
- Duplicate rows: **0**

### SECOM
- Duplicate rows: **0** (feature matrix)

---

## 7. Target Distributions

### AI4I — `Machine failure`

| Class | Count | Percentage |
|-------|-------|-----------|
| 0 — No Failure | 9,661 | 96.61% |
| 1 — Machine Failure | 339 | 3.39% |
| **Imbalance ratio** | | **28.5:1** |

**Failure type breakdown:**

| Type | Count |
|------|-------|
| TWF (Tool Wear Failure) | 46 |
| HDF (Heat Dissipation Failure) | 115 |
| PWF (Power Failure) | 95 |
| OSF (Overstrain Failure) | 98 |
| RNF (Random Failure) | 19 |

### SECOM — `label`

| Class | Count | Percentage |
|-------|-------|-----------|
| -1 — Pass | 1,463 | 93.36% |
| 1 — Fail | 104 | 6.64% |
| **Imbalance ratio** | | **14.1:1** |

---

## 8. Data Quality Observations

### AI4I
- **Clean dataset** — zero missing values, zero duplicates, no infinite values.
- Strong class imbalance (28.5:1) will require SMOTE or class-weight handling in Phase 4.
- The five failure-type columns (TWF/HDF/PWF/OSF/RNF) are a subset of `Machine failure`.
  Some rows have `Machine failure = 1` but all five sub-types = 0 — this may reflect RNF + other unreported causes.
- `UDI` is a sequential row ID. `Product ID` is a string code. Neither should be used as model features.
- Air and Process temperatures are highly correlated (process ≈ air + 10 K).

### SECOM
- **Pervasive missing data** — 538 of 590 features have at least one NaN.
- **High-dimensional small-sample problem** — 590 features, 1,567 rows. Feature selection is essential.
- **Severe class imbalance** — 14.1:1 (Pass:Fail). Balanced accuracy and F1 are appropriate metrics.
- Some features are constant or near-constant (zero or near-zero variance) and carry no information.
- All features are float64 — no categorical encoding needed for the feature matrix.
- Timestamps in the label file enable time-ordered train/validation splitting.

---

## 9. Issues Discovered

| # | Issue | Dataset | Severity | Action in Phase 2 |
|---|-------|---------|----------|-------------------|
| 1 | Severe class imbalance (28.5:1) | AI4I | High | Use SMOTE / class weights during training |
| 2 | Severe class imbalance (14.1:1) | SECOM | High | Use SMOTE / class weights / threshold-moving |
| 3 | 41,951 missing values across 538 features | SECOM | High | Impute (median/mean/KNN) or drop high-miss features |
| 4 | Constant / all-NaN features present | SECOM | Medium | Drop zero-variance features in preprocessing |
| 5 | No column names in source file | SECOM | Low | Auto-naming applied (`feature_001`...`feature_590`) |
| 6 | `UDI` and `Product ID` not suitable as model features | AI4I | Low | Exclude from feature matrices in Phase 4 |
| 7 | Air temp and process temp are correlated | AI4I | Low | Consider temperature delta feature in Phase 3 EDA |

---

## 10. Phase 1 Completion Status

| Component | Status | Notes |
|-----------|--------|-------|
| Repository inspection | Complete | Existing files preserved |
| Dataset file discovery | Complete | Both datasets located and inspected |
| Directory structure | Complete | All required directories created |
| Python environment | Complete | Python 3.13.3, all packages installed |
| `requirements.txt` | Complete | All Phase 1 dependencies documented |
| `configs/dataset_config.yaml` | Complete | Fully configurable, no hardcoded paths |
| `src/utils/logger.py` | Complete | Rotating file + console handler |
| `src/utils/config.py` | Complete | YAML-driven configuration loader |
| `src/ingestion/dataset_loader.py` | Complete | AI4I + SECOM loaders with config wrappers |
| `src/validation/dataset_validator.py` | Complete | 7 check types, structured ValidationReport |
| `src/inspection/dataset_profiler.py` | Complete | JSON + CSV report generation |
| `main.py` | Complete | 5-step pipeline, graceful error handling |
| `tests/test_dataset_loader.py` | Complete | 17 tests — all passing |
| `tests/test_dataset_validator.py` | Complete | 21 tests — all passing |
| **Total tests** | **38 / 38 passed** | |
| Jupyter notebooks (3) | Complete | Ready to run top-to-bottom |
| Generated reports (AI4I) | Complete | JSON profile + missing CSV + numerical CSV |
| Generated reports (SECOM) | Complete | JSON profile + missing CSV + numerical CSV |
| `README.md` | Complete | Full professional documentation |
| `docs/PHASE_1_REPORT.md` | Complete | This file |
| `docs/PROJECT_OVERVIEW.md` | Complete | |
| `docs/DATASET_DOCUMENTATION.md` | Complete | |
| `docs/ARCHITECTURE.md` | Complete | |
| `.gitignore` | Complete | Excludes large datasets and secrets |
| `.env.example` | Complete | Template for environment variables |

---

## Phase 2 Prerequisites

The following must be available at the start of Phase 2:

1. All Phase 1 files committed to Git (excluding raw datasets).
2. Raw dataset files at `datasets/raw/ai4i/` and `datasets/raw/secom/`.
3. Python virtual environment with `requirements.txt` installed.
4. Phase 1 reports in `datasets/reports/` for reference.
5. Understanding of data quality issues documented in Section 9 above.

**Phase 2 will focus on:**
- Missing value imputation strategies (median, KNN, model-based)
- Constant and low-variance feature removal
- Feature scaling and normalisation
- Train/validation/test splitting (respecting SECOM timestamps)
- Class imbalance handling strategy selection
- Building reusable preprocessing pipelines
