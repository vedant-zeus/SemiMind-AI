"""
SemiMind AI - Dataset Validator
================================
Implements a suite of data-quality checks for the Phase 1 datasets.

**Phase 1 scope**: *identification only* - this module flags quality issues
and returns structured results.  It does **not** impute, remove, or
transform any data.

Checks performed
----------------
* Empty dataset detection
* Duplicate row detection
* Missing value analysis (per column)
* Infinite value detection (numerical columns)
* Invalid / unexpected column names
* Inconsistent data types
* Unexpected target variable values
* Row alignment between feature and label files (SECOM)

Usage
-----
    from src.validation.dataset_validator import DatasetValidator, ValidationReport
    validator = DatasetValidator()
    report = validator.validate_ai4i(df)
    print(report.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    severity: str          # "INFO" | "WARNING" | "ERROR"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregated validation results for one dataset."""
    dataset_name: str
    total_rows: int
    total_columns: int
    checks: List[CheckResult] = field(default_factory=list)

    # -- Accessors -------------------------------------------------------------

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def errors(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "ERROR"]

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "WARNING"]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"ValidationReport -> {self.dataset_name}",
            f"  Rows: {self.total_rows}  |  Columns: {self.total_columns}",
            f"  Checks: {len(self.checks)} total  |  "
            f"Passed: {self.passed_count}  |  Failed: {self.failed_count}",
        ]
        if self.errors:
            lines.append("  --- ERRORS ---------------------------------")
            for c in self.errors:
                lines.append(f"  [FAIL] [{c.check_name}] {c.message}")
        if self.warnings:
            lines.append("  --- WARNINGS --------------------------------")
            for c in self.warnings:
                lines.append(f"  ⚠ [{c.check_name}] {c.message}")
        passed_names = [c.check_name for c in self.checks if c.passed]
        if passed_names:
            lines.append("  --- PASSED ----------------------------------")
            for name in passed_names:
                lines.append(f"  [OK] {name}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dictionary (JSON-compatible)."""
        return {
            "dataset_name": self.dataset_name,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "checks": [
                {
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# -----------------------------------------------------------------------------
# Validator
# -----------------------------------------------------------------------------

class DatasetValidator:
    """
    Runs a standard set of data-quality checks on a DataFrame.

    All public methods return a :class:`ValidationReport`.
    """

    # -- Generic checks (reusable across datasets) -----------------------------

    def _check_not_empty(self, df: pd.DataFrame, name: str) -> CheckResult:
        passed = len(df) > 0 and len(df.columns) > 0
        return CheckResult(
            check_name="EmptyDataset",
            passed=passed,
            severity="ERROR",
            message=(
                "Dataset is not empty."
                if passed
                else f"Dataset '{name}' is empty - 0 rows or 0 columns."
            ),
            details={"rows": len(df), "columns": len(df.columns)},
        )

    def _check_duplicates(self, df: pd.DataFrame) -> CheckResult:
        dup_count = int(df.duplicated().sum())
        passed = dup_count == 0
        return CheckResult(
            check_name="DuplicateRows",
            passed=passed,
            severity="WARNING",
            message=(
                "No duplicate rows detected."
                if passed
                else f"{dup_count} duplicate row(s) detected."
            ),
            details={"duplicate_count": dup_count},
        )

    def _check_missing_values(self, df: pd.DataFrame) -> CheckResult:
        missing_per_col = df.isnull().sum()
        total_missing = int(missing_per_col.sum())
        cols_with_missing = missing_per_col[missing_per_col > 0]
        passed = total_missing == 0
        pct_missing = round(total_missing / max(df.size, 1) * 100, 4)
        return CheckResult(
            check_name="MissingValues",
            passed=passed,
            severity="WARNING",
            message=(
                "No missing values detected."
                if passed
                else (
                    f"{total_missing} missing value(s) across "
                    f"{len(cols_with_missing)} column(s) "
                    f"({pct_missing}% of all cells)."
                )
            ),
            details={
                "total_missing": total_missing,
                "columns_affected": int(len(cols_with_missing)),
                "pct_missing": pct_missing,
                "per_column": {
                    col: int(cnt)
                    for col, cnt in cols_with_missing.items()
                },
            },
        )

    def _check_infinite_values(self, df: pd.DataFrame) -> CheckResult:
        num_df = df.select_dtypes(include=[np.number])
        inf_per_col = num_df.apply(lambda s: np.isinf(s).sum())
        total_inf = int(inf_per_col.sum())
        cols_with_inf = inf_per_col[inf_per_col > 0]
        passed = total_inf == 0
        return CheckResult(
            check_name="InfiniteValues",
            passed=passed,
            severity="WARNING",
            message=(
                "No infinite values detected."
                if passed
                else (
                    f"{total_inf} infinite value(s) across "
                    f"{len(cols_with_inf)} column(s)."
                )
            ),
            details={
                "total_infinite": total_inf,
                "columns_affected": int(len(cols_with_inf)),
                "per_column": {col: int(cnt) for col, cnt in cols_with_inf.items()},
            },
        )

    def _check_column_names(
        self, df: pd.DataFrame, expected: Optional[Set[str]] = None
    ) -> CheckResult:
        actual = set(df.columns.tolist())
        has_unnamed = any(str(c).startswith("Unnamed") for c in df.columns)
        if expected:
            missing_cols = expected - actual
            extra_cols = actual - expected
            passed = not missing_cols and not has_unnamed
            msg = (
                "All expected columns present."
                if passed
                else (
                    f"Column name issues - missing: {missing_cols}, "
                    f"unexpected: {extra_cols}."
                )
            )
        else:
            passed = not has_unnamed
            msg = (
                "Column names look valid."
                if passed
                else "Unnamed columns detected - possible header parsing issue."
            )
        return CheckResult(
            check_name="ColumnNames",
            passed=passed,
            severity="WARNING",
            message=msg,
            details={
                "actual_columns": sorted(actual),
                "has_unnamed_columns": has_unnamed,
            },
        )

    def _check_dtypes(self, df: pd.DataFrame) -> CheckResult:
        """Flag columns whose dtype is entirely object but contain numeric data."""
        suspect: List[str] = []
        for col in df.select_dtypes(include="object").columns:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                non_null_original = df[col].dropna()
                if len(non_null_original) > 0 and converted.notna().sum() / len(non_null_original) > 0.9:
                    suspect.append(col)
            except Exception:
                pass
        passed = len(suspect) == 0
        return CheckResult(
            check_name="DataTypes",
            passed=passed,
            severity="WARNING",
            message=(
                "Data types appear consistent."
                if passed
                else (
                    f"{len(suspect)} object column(s) appear to contain "
                    f"numeric data: {suspect}"
                )
            ),
            details={"suspect_columns": suspect},
        )

    def _check_target_values(
        self,
        df: pd.DataFrame,
        target_col: str,
        expected_values: Optional[List[Any]] = None,
    ) -> CheckResult:
        if target_col not in df.columns:
            return CheckResult(
                check_name="TargetValues",
                passed=False,
                severity="ERROR",
                message=f"Target column '{target_col}' not found in dataset.",
                details={"target_column": target_col},
            )
        actual_values = sorted(df[target_col].dropna().unique().tolist())
        value_counts = df[target_col].value_counts().to_dict()
        total = len(df)
        distribution = {
            str(k): {
                "count": int(v),
                "pct": round(v / total * 100, 2),
            }
            for k, v in value_counts.items()
        }
        if expected_values is not None:
            unexpected = [v for v in actual_values if v not in expected_values]
            passed = len(unexpected) == 0
            msg = (
                f"Target column '{target_col}' contains expected values only."
                if passed
                else (
                    f"Unexpected values in '{target_col}': {unexpected}. "
                    f"Expected: {expected_values}."
                )
            )
        else:
            passed = True
            msg = (
                f"Target column '{target_col}' has {len(actual_values)} unique "
                f"value(s): {actual_values}."
            )
        return CheckResult(
            check_name="TargetValues",
            passed=passed,
            severity="WARNING",
            message=msg,
            details={
                "target_column": target_col,
                "unique_values": actual_values,
                "distribution": distribution,
            },
        )

    def _check_row_alignment(
        self, df_features: pd.DataFrame, df_labels: pd.DataFrame
    ) -> CheckResult:
        passed = len(df_features) == len(df_labels)
        return CheckResult(
            check_name="RowAlignment",
            passed=passed,
            severity="ERROR",
            message=(
                f"Feature and label files are aligned ({len(df_features)} rows each)."
                if passed
                else (
                    f"Row count mismatch - features: {len(df_features)}, "
                    f"labels: {len(df_labels)}."
                )
            ),
            details={
                "feature_rows": len(df_features),
                "label_rows": len(df_labels),
            },
        )

    # -- Dataset-specific validators -------------------------------------------

    def validate_ai4i(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all quality checks against the AI4I dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Raw AI4I DataFrame as returned by
            :func:`src.ingestion.dataset_loader.load_ai4i`.

        Returns
        -------
        ValidationReport
        """
        name = "AI4I 2020 Predictive Maintenance"
        logger.info("[Validator] Starting validation for: %s", name)
        report = ValidationReport(
            dataset_name=name,
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        report.checks.append(self._check_not_empty(df, name))
        report.checks.append(self._check_duplicates(df))
        report.checks.append(self._check_missing_values(df))
        report.checks.append(self._check_infinite_values(df))
        report.checks.append(self._check_column_names(df))
        report.checks.append(self._check_dtypes(df))
        report.checks.append(
            self._check_target_values(df, "Machine failure", expected_values=[0, 1])
        )

        logger.info(
            "[Validator] AI4I validation complete -> %d passed, %d failed.",
            report.passed_count, report.failed_count,
        )
        return report

    def validate_secom(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> ValidationReport:
        """
        Run all quality checks against the SECOM dataset.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix as returned by
            :func:`src.ingestion.dataset_loader.load_secom`.
        labels_df : pd.DataFrame
            Labels DataFrame as returned by
            :func:`src.ingestion.dataset_loader.load_secom`.

        Returns
        -------
        ValidationReport
        """
        name = "SECOM Semiconductor Manufacturing"
        logger.info("[Validator] Starting validation for: %s", name)
        report = ValidationReport(
            dataset_name=name,
            total_rows=len(features_df),
            total_columns=len(features_df.columns),
        )

        report.checks.append(self._check_not_empty(features_df, name))
        report.checks.append(self._check_row_alignment(features_df, labels_df))
        report.checks.append(self._check_duplicates(features_df))
        report.checks.append(self._check_missing_values(features_df))
        report.checks.append(self._check_infinite_values(features_df))
        report.checks.append(self._check_column_names(features_df))
        report.checks.append(
            self._check_target_values(labels_df, "label", expected_values=[-1, 1])
        )

        logger.info(
            "[Validator] SECOM validation complete -> %d passed, %d failed.",
            report.passed_count, report.failed_count,
        )
        return report
