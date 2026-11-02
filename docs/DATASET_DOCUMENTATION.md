# SemiMind AI – Dataset Documentation

## Dataset A: AI4I 2020 Predictive Maintenance Dataset

**Source:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset  
**File:** `datasets/raw/ai4i/ai4i2020.csv`  
**Format:** CSV, comma-delimited, header row 0, UTF-8

### Column Reference

| Column | dtype | Unit | Description |
|--------|-------|------|-------------|
| UDI | int64 | — | Unique identifier (1–10,000) |
| Product ID | object | — | Product serial number (e.g. M14860) |
| Type | object | — | Machine quality variant: L (low), M (medium), H (high) |
| Air temperature [K] | float64 | Kelvin | Ambient air temperature |
| Process temperature [K] | float64 | Kelvin | Process temperature (≈ air + 10 K) |
| Rotational speed [rpm] | int64 | RPM | Motor rotational speed |
| Torque [Nm] | float64 | Newton-metres | Applied torque |
| Tool wear [min] | int64 | Minutes | Cumulative tool wear |
| Machine failure | int64 | 0/1 | **Target**: 0 = no failure, 1 = failure |
| TWF | int64 | 0/1 | Tool Wear Failure sub-indicator |
| HDF | int64 | 0/1 | Heat Dissipation Failure sub-indicator |
| PWF | int64 | 0/1 | Power Failure sub-indicator |
| OSF | int64 | 0/1 | Overstrain Failure sub-indicator |
| RNF | int64 | 0/1 | Random Failure sub-indicator |

### Key Statistics (Phase 1 Inspection)
- Rows: 10,000 | Columns: 14
- Missing values: 0
- Duplicate rows: 0
- Failure rate: 3.39% (339 / 10,000)
- Class imbalance ratio: 28.5:1

---

## Dataset B: SECOM Semiconductor Manufacturing Dataset

**Source:** https://archive.ics.uci.edu/dataset/179/secom  
**Feature file:** `datasets/raw/secom/secom.data`  
**Labels file:** `datasets/raw/secom/secom_labels.data`  
**Format:** Space-delimited text, no header, UTF-8, NaN for missing

### File Structure

```
secom.data:
  - 1,567 rows x 590 columns
  - Each row = one production entity / wafer
  - Each column = one sensor / measurement signal
  - Missing values represented as NaN (MATLAB convention)
  - No column names in file; assigned feature_001...feature_590

secom_labels.data:
  - 1,567 rows x 2 columns
  - Column 1: label (-1 = Pass, 1 = Fail)
  - Column 2: timestamp in "DD/MM/YYYY HH:MM:SS" format (quoted)
  - Row-aligned with secom.data (rows must match exactly)
```

### Key Statistics (Phase 1 Inspection)
- Rows: 1,567 | Feature columns: 590
- Missing cells: 41,951 (4.54% of all feature cells)
- Columns with any missing: 538 / 590 (91.2%)
- Pass: 1,463 (93.36%) | Fail: 104 (6.64%)
- Class imbalance ratio: 14.1:1
- All features are float64
- Timestamps span from 19/07/2008 onward

### Label Encoding
| Value | Meaning |
|-------|---------|
| -1 | Pass — wafer passed in-house line testing |
| 1 | Fail — wafer failed in-house line testing (yield excursion) |

