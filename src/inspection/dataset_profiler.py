"""
SemiMind AI - Dataset Profiler
================================
Generates a comprehensive statistical profile of each dataset and saves
the results as JSON and CSV report files.

Profile sections
----------------
1. **General information** - name, shape, file size, column names, dtypes.
2. **Data quality**        - missing values, duplicates, infinite values,
                             constant columns.
3. **Numerical summary**   - mean, median, std, min, max, quartiles.
4. **Target information**  - unique values, class distribution, imbalance.

Output files
------------
For AI4I::

    datasets/reports/ai4i/ai4i_dataset_profile.json
    datasets/reports/ai4i/ai4i_missing_values.csv
    datasets/reports/ai4i/ai4i_numerical_summary.csv

For SECOM::

    datasets/reports/secom/secom_dataset_profile.json
    datasets/reports/secom/secom_missing_values.csv
    datasets/reports/secom/secom_numerical_summary.csv

Usage
-----
    from src.inspection.dataset_profiler import DatasetProfiler
    profiler = DatasetProfiler(reports_root=Path("datasets/reports"))
    profiler.profile_ai4i(df)
    profiler.profile_secom(features_df, labels_df)
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# JSON encoder that handles numpy scalars and NaN/Inf values
class _SafeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # type: ignore[override]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return super().default(obj)


# -----------------------------------------------------------------------------
# Profiler
# -----------------------------------------------------------------------------

class DatasetProfiler:
    """
    Generates statistical profiles for SemiMind AI datasets.

    Parameters
    ----------
    reports_root : Path
        Root directory where per-dataset report sub-directories reside.
        E.g. ``Path("datasets/reports")``.
    """

    def __init__(self, reports_root: Path) -> None:
        self.reports_root = Path(reports_root)

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _general_info(
        df: pd.DataFrame,
        name: str,
        raw_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        file_size_bytes: Optional[int] = None
        if raw_file and raw_file.is_file():
            file_size_bytes = raw_file.stat().st_size

        return {
            "dataset_name": name,
            "profiled_at": datetime.now().isoformat(timespec="seconds"),
            "num_rows": int(len(df)),
            "num_columns": int(len(df.columns)),
            "file_size_bytes": file_size_bytes,
            "file_size_mb": (
                round(file_size_bytes / 1_048_576, 3)
                if file_size_bytes
                else None
            ),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

    @staticmethod
    def _data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        # Missing values
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(4)
        total_missing = int(missing_counts.sum())
        total_cells = df.size
        pct_missing_overall = round(total_missing / max(total_cells, 1) * 100, 4)

        # Duplicates
        dup_count = int(df.duplicated().sum())

        # Infinite values (numeric only)
        num_df = df.select_dtypes(include=[np.number])
        inf_counts = num_df.apply(lambda s: np.isinf(s).sum())
        total_inf = int(inf_counts.sum())

        # Constant columns
        constant_cols = [
            col for col in df.columns
            if df[col].nunique(dropna=True) <= 1
        ]

        return {
            "total_missing_values": total_missing,
            "pct_missing_overall": pct_missing_overall,
            "columns_with_missing": int((missing_counts > 0).sum()),
            "duplicate_rows": dup_count,
            "infinite_values": total_inf,
            "constant_columns": constant_cols,
            "constant_column_count": len(constant_cols),
            "missing_per_column": {
                col: {
                    "count": int(missing_counts[col]),
                    "pct": float(missing_pct[col]),
                }
                for col in df.columns
                if missing_counts[col] > 0
            },
        }

    @staticmethod
    def _numerical_summary(df: pd.DataFrame) -> Dict[str, Any]:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return {}
        desc = num_df.describe(percentiles=[0.25, 0.5, 0.75]).T
        summary: Dict[str, Any] = {}
        for col in desc.index:
            summary[col] = {
                "count": float(desc.loc[col, "count"]),
                "mean": float(desc.loc[col, "mean"]) if not math.isnan(desc.loc[col, "mean"]) else None,
                "median": float(num_df[col].median()) if not math.isnan(num_df[col].median()) else None,
                "std": float(desc.loc[col, "std"]) if not math.isnan(desc.loc[col, "std"]) else None,
                "min": float(desc.loc[col, "min"]) if not math.isnan(desc.loc[col, "min"]) else None,
                "q25": float(desc.loc[col, "25%"]) if not math.isnan(desc.loc[col, "25%"]) else None,
                "q50": float(desc.loc[col, "50%"]) if not math.isnan(desc.loc[col, "50%"]) else None,
                "q75": float(desc.loc[col, "75%"]) if not math.isnan(desc.loc[col, "75%"]) else None,
                "max": float(desc.loc[col, "max"]) if not math.isnan(desc.loc[col, "max"]) else None,
            }
        return summary

    @staticmethod
    def _target_info(
        df: pd.DataFrame,
        target_col: str,
        label_description: Optional[Dict[Any, str]] = None,
    ) -> Dict[str, Any]:
        if target_col not in df.columns:
            return {"error": f"Target column '{target_col}' not found."}

        series = df[target_col]
        unique_vals = sorted(series.dropna().unique().tolist())
        total = len(series)
        vc = series.value_counts().to_dict()

        distribution = {}
        for val in unique_vals:
            count = int(vc.get(val, 0))
            pct = round(count / total * 100, 2)
            label_str = (
                label_description.get(val, str(val))
                if label_description
                else str(val)
            )
            distribution[str(val)] = {
                "label": label_str,
                "count": count,
                "pct": pct,
            }

        # Class imbalance: ratio of majority to minority class
        counts_only = [v["count"] for v in distribution.values()]
        imbalance_ratio: Optional[float] = None
        if len(counts_only) >= 2:
            majority = max(counts_only)
            minority = min(counts_only)
            imbalance_ratio = round(majority / max(minority, 1), 2)

        return {
            "target_column": target_col,
            "unique_values": unique_vals,
            "num_classes": len(unique_vals),
            "distribution": distribution,
            "imbalance_ratio": imbalance_ratio,
            "imbalance_note": (
                "Severe class imbalance (ratio > 10)."
                if imbalance_ratio and imbalance_ratio > 10
                else (
                    "Moderate class imbalance (ratio 3-10)."
                    if imbalance_ratio and imbalance_ratio > 3
                    else "Classes are roughly balanced."
                )
            ),
        }

    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, cls=_SafeEncoder, ensure_ascii=False)
        logger.info("[Profiler] Saved JSON report -> %s", path)

    def _save_missing_csv(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(4)
        report = pd.DataFrame({
            "column": df.columns,
            "missing_count": missing_counts.values,
            "missing_pct": missing_pct.values,
        })
        report.to_csv(path, index=False)
        logger.info("[Profiler] Saved missing-values CSV -> %s", path)

    def _save_numerical_csv(self, summary: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not summary:
            return
        rows = []
        for col, stats in summary.items():
            row = {"column": col}
            row.update(stats)
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("[Profiler] Saved numerical summary CSV -> %s", path)

    # -- Public profiling methods -----------------------------------------------

    def profile_ai4i(
        self,
        df: pd.DataFrame,
        raw_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate a full profile for the AI4I dataset and write reports.

        Parameters
        ----------
        df : pd.DataFrame
            Raw AI4I DataFrame.
        raw_file : Path, optional
            Path to the source CSV file (used to report file size).

        Returns
        -------
        dict
            Full profile dictionary (also saved as JSON).
        """
        logger.info("[Profiler] Profiling AI4I dataset ...")
        out_dir = self.reports_root / "ai4i"
        out_dir.mkdir(parents=True, exist_ok=True)

        general   = self._general_info(df, "AI4I 2020 Predictive Maintenance", raw_file)
        quality   = self._data_quality(df)
        numerical = self._numerical_summary(df)
        target    = self._target_info(
            df,
            "Machine failure",
            label_description={0: "No Failure", 1: "Machine Failure"},
        )

        profile = {
            "general_information": general,
            "data_quality": quality,
            "numerical_summary": numerical,
            "target_information": target,
        }

        self._save_json(profile, out_dir / "ai4i_dataset_profile.json")
        self._save_missing_csv(df, out_dir / "ai4i_missing_values.csv")
        self._save_numerical_csv(numerical, out_dir / "ai4i_numerical_summary.csv")

        logger.info("[Profiler] AI4I profiling complete. Reports in: %s", out_dir)
        return profile

    def profile_secom(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        raw_data_file: Optional[Path] = None,
        raw_labels_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate a full profile for the SECOM dataset and write reports.

        Parameters
        ----------
        features_df : pd.DataFrame
            SECOM feature matrix.
        labels_df : pd.DataFrame
            SECOM labels DataFrame (columns: ``label``, ``timestamp``).
        raw_data_file : Path, optional
            Path to ``secom.data`` (used to report file size).
        raw_labels_file : Path, optional
            Path to ``secom_labels.data`` (used to report file size).

        Returns
        -------
        dict
            Full profile dictionary (also saved as JSON).
        """
        logger.info("[Profiler] Profiling SECOM dataset ...")
        out_dir = self.reports_root / "secom"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Combined size when two files exist
        total_size: Optional[int] = None
        if raw_data_file and raw_labels_file:
            s1 = raw_data_file.stat().st_size if raw_data_file.is_file() else 0
            s2 = raw_labels_file.stat().st_size if raw_labels_file.is_file() else 0
            total_size = s1 + s2

        general = self._general_info(features_df, "SECOM Semiconductor Manufacturing")
        if total_size:
            general["file_size_bytes"] = total_size
            general["file_size_mb"] = round(total_size / 1_048_576, 3)
        general["label_file_rows"] = int(len(labels_df))

        quality   = self._data_quality(features_df)
        numerical = self._numerical_summary(features_df)
        target    = self._target_info(
            labels_df,
            "label",
            label_description={-1: "Pass", 1: "Fail"},
        )

        # Identify constant features (zero variance)
        constant_features = [
            col for col in features_df.columns
            if features_df[col].nunique(dropna=True) <= 1
        ]

        profile = {
            "general_information": general,
            "data_quality": quality,
            "numerical_summary": numerical,
            "target_information": target,
            "constant_features": {
                "count": len(constant_features),
                "columns": constant_features,
            },
        }

        self._save_json(profile, out_dir / "secom_dataset_profile.json")
        self._save_missing_csv(features_df, out_dir / "secom_missing_values.csv")
        self._save_numerical_csv(numerical, out_dir / "secom_numerical_summary.csv")

        logger.info("[Profiler] SECOM profiling complete. Reports in: %s", out_dir)
        return profile
