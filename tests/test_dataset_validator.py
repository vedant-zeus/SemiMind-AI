"""
SemiMind AI – Tests for Dataset Validator
==========================================
Unit tests for src/validation/dataset_validator.py

Uses small synthetic fixtures; does NOT depend on the full raw datasets.
Run with: pytest tests/test_dataset_validator.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation.dataset_validator import DatasetValidator, ValidationReport


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a clean AI4I-like DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _make_ai4i(rows: int = 5, include_failure: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "UDI": range(1, rows + 1),
        "Product ID": [f"M{i}" for i in range(rows)],
        "Type": rng.choice(["L", "M", "H"], size=rows).tolist(),
        "Air temperature [K]": rng.uniform(295, 305, rows).round(1).tolist(),
        "Process temperature [K]": rng.uniform(305, 315, rows).round(1).tolist(),
        "Rotational speed [rpm]": rng.integers(1000, 3000, rows).tolist(),
        "Torque [Nm]": rng.uniform(20, 80, rows).round(1).tolist(),
        "Tool wear [min]": rng.integers(0, 300, rows).tolist(),
        "Machine failure": rng.integers(0, 2, rows).tolist(),
        "TWF": [0] * rows,
        "HDF": [0] * rows,
        "PWF": [0] * rows,
        "OSF": [0] * rows,
        "RNF": [0] * rows,
    }
    return pd.DataFrame(data)


def _make_secom_features(rows: int = 5, n_features: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cols = [f"feature_{i+1}" for i in range(n_features)]
    data = rng.random((rows, n_features))
    return pd.DataFrame(data, columns=cols)


def _make_secom_labels(rows: int = 5) -> pd.DataFrame:
    labels = [-1] * (rows - 1) + [1]
    timestamps = [f"01/01/2008 {10+i}:00:00" for i in range(rows)]
    return pd.DataFrame({"label": labels, "timestamp": timestamps})


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAI4IValidation:
    """Tests for DatasetValidator.validate_ai4i."""

    def test_returns_validation_report(self) -> None:
        df = _make_ai4i()
        v = DatasetValidator()
        report = v.validate_ai4i(df)
        assert isinstance(report, ValidationReport)

    def test_clean_dataset_no_errors(self) -> None:
        df = _make_ai4i()
        # Fix Machine failure to only 0/1
        df["Machine failure"] = 0
        report = DatasetValidator().validate_ai4i(df)
        assert len(report.errors) == 0

    def test_empty_dataset_detected(self) -> None:
        df = pd.DataFrame(columns=["UDI", "Machine failure"])
        report = DatasetValidator().validate_ai4i(df)
        empty_checks = [c for c in report.checks if c.check_name == "EmptyDataset"]
        assert len(empty_checks) == 1
        assert not empty_checks[0].passed

    def test_duplicate_rows_detected(self) -> None:
        df = _make_ai4i(rows=4)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add one duplicate
        report = DatasetValidator().validate_ai4i(df)
        dup_checks = [c for c in report.checks if c.check_name == "DuplicateRows"]
        assert len(dup_checks) == 1
        assert not dup_checks[0].passed
        assert dup_checks[0].details["duplicate_count"] == 1

    def test_missing_values_detected(self) -> None:
        df = _make_ai4i()
        df.loc[0, "Air temperature [K]"] = np.nan
        df.loc[2, "Torque [Nm]"] = np.nan
        report = DatasetValidator().validate_ai4i(df)
        mv_checks = [c for c in report.checks if c.check_name == "MissingValues"]
        assert len(mv_checks) == 1
        assert not mv_checks[0].passed
        assert mv_checks[0].details["total_missing"] == 2

    def test_infinite_values_detected(self) -> None:
        df = _make_ai4i()
        df.loc[1, "Rotational speed [rpm]"] = np.inf
        report = DatasetValidator().validate_ai4i(df)
        inf_checks = [c for c in report.checks if c.check_name == "InfiniteValues"]
        assert len(inf_checks) == 1
        assert not inf_checks[0].passed

    def test_unexpected_target_value_detected(self) -> None:
        df = _make_ai4i()
        df.loc[0, "Machine failure"] = 99   # invalid value
        report = DatasetValidator().validate_ai4i(df)
        tv_checks = [c for c in report.checks if c.check_name == "TargetValues"]
        assert len(tv_checks) == 1
        assert not tv_checks[0].passed

    def test_missing_target_column_detected(self) -> None:
        df = _make_ai4i().drop(columns=["Machine failure"])
        report = DatasetValidator().validate_ai4i(df)
        tv_checks = [c for c in report.checks if c.check_name == "TargetValues"]
        assert not tv_checks[0].passed

    def test_no_missing_values_passes(self) -> None:
        df = _make_ai4i()
        report = DatasetValidator().validate_ai4i(df)
        mv_checks = [c for c in report.checks if c.check_name == "MissingValues"]
        assert mv_checks[0].passed


class TestSECOMValidation:
    """Tests for DatasetValidator.validate_secom."""

    def test_returns_validation_report(self) -> None:
        features = _make_secom_features()
        labels = _make_secom_labels()
        report = DatasetValidator().validate_secom(features, labels)
        assert isinstance(report, ValidationReport)

    def test_clean_secom_no_errors(self) -> None:
        features = _make_secom_features()
        labels = _make_secom_labels()
        report = DatasetValidator().validate_secom(features, labels)
        assert len(report.errors) == 0

    def test_row_alignment_mismatch_detected(self) -> None:
        features = _make_secom_features(rows=5)
        labels = _make_secom_labels(rows=3)   # intentional mismatch
        report = DatasetValidator().validate_secom(features, labels)
        align_checks = [c for c in report.checks if c.check_name == "RowAlignment"]
        assert not align_checks[0].passed

    def test_missing_values_detected_in_features(self) -> None:
        features = _make_secom_features()
        features.iloc[0, 0] = np.nan
        labels = _make_secom_labels()
        report = DatasetValidator().validate_secom(features, labels)
        mv_checks = [c for c in report.checks if c.check_name == "MissingValues"]
        assert not mv_checks[0].passed

    def test_label_distribution_reported(self) -> None:
        features = _make_secom_features()
        labels = _make_secom_labels()
        report = DatasetValidator().validate_secom(features, labels)
        tv_checks = [c for c in report.checks if c.check_name == "TargetValues"]
        assert len(tv_checks) == 1
        assert "-1" in tv_checks[0].details["distribution"]

    def test_unexpected_label_value_detected(self) -> None:
        features = _make_secom_features()
        labels = _make_secom_labels()
        labels.loc[0, "label"] = 99   # invalid
        report = DatasetValidator().validate_secom(features, labels)
        tv_checks = [c for c in report.checks if c.check_name == "TargetValues"]
        assert not tv_checks[0].passed


class TestValidationReport:
    """Tests for the ValidationReport helper methods."""

    def test_summary_string_is_nonempty(self) -> None:
        df = _make_ai4i()
        report = DatasetValidator().validate_ai4i(df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_to_dict_is_serialisable(self) -> None:
        import json
        df = _make_ai4i()
        report = DatasetValidator().validate_ai4i(df)
        d = report.to_dict()
        # Must not raise
        json_str = json.dumps(d)
        assert "dataset_name" in json_str

    def test_passed_failed_counts_are_consistent(self) -> None:
        df = _make_ai4i()
        report = DatasetValidator().validate_ai4i(df)
        assert report.passed_count + report.failed_count == len(report.checks)


class TestConfigLoading:
    """Test that the configuration can be loaded from the default path."""

    def test_config_loads_without_error(self) -> None:
        from src.utils.config import load_config
        cfg = load_config()
        assert cfg is not None
        assert cfg.raw_data_root.is_dir()

    def test_config_has_ai4i_section(self) -> None:
        from src.utils.config import load_config
        cfg = load_config()
        assert "name" in cfg.ai4i
        assert "raw_file" in cfg.ai4i

    def test_config_has_secom_section(self) -> None:
        from src.utils.config import load_config
        cfg = load_config()
        assert "name" in cfg.secom
        assert "raw_data_file" in cfg.secom
        assert "raw_labels_file" in cfg.secom
