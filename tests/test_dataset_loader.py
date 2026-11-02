"""
SemiMind AI – Tests for Dataset Loader
========================================
Unit tests for src/ingestion/dataset_loader.py

Uses small synthetic fixtures; does NOT depend on the full raw datasets.
Run with: pytest tests/test_dataset_loader.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on path when running from any directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.dataset_loader import load_ai4i, load_secom


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def ai4i_csv(tmp_path: Path) -> Path:
    """Minimal AI4I-style CSV fixture."""
    content = (
        "UDI,Product ID,Type,Air temperature [K],Process temperature [K],"
        "Rotational speed [rpm],Torque [Nm],Tool wear [min],"
        "Machine failure,TWF,HDF,PWF,OSF,RNF\n"
        "1,M14860,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0\n"
        "2,L47181,L,298.2,308.7,1408,46.3,3,0,0,0,0,0,0\n"
        "3,H47182,H,298.1,308.5,1498,49.4,5,1,0,0,1,0,0\n"
    )
    p = tmp_path / "ai4i2020.csv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def secom_data(tmp_path: Path) -> Path:
    """Minimal SECOM space-delimited feature fixture (3 rows × 4 features)."""
    rows = [
        "1.0 2.0 NaN 4.0",
        "5.0 6.0 7.0 8.0",
        "NaN 10.0 11.0 12.0",
    ]
    p = tmp_path / "secom.data"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


@pytest.fixture()
def secom_labels(tmp_path: Path) -> Path:
    """Minimal SECOM labels fixture aligned to 3 rows."""
    rows = [
        '-1 "01/01/2008 10:00:00"',
        '-1 "01/01/2008 11:00:00"',
        '1 "01/01/2008 12:00:00"',
    ]
    p = tmp_path / "secom_labels.data"
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


@pytest.fixture()
def empty_csv(tmp_path: Path) -> Path:
    """CSV with header only (zero data rows)."""
    p = tmp_path / "empty.csv"
    p.write_text("col_a,col_b,col_c\n", encoding="utf-8")
    return p


@pytest.fixture()
def duplicate_csv(tmp_path: Path) -> Path:
    """CSV with duplicate rows."""
    content = "a,b,c\n1,2,3\n1,2,3\n4,5,6\n"
    p = tmp_path / "dup.csv"
    p.write_text(content, encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# AI4I loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAI4I:
    def test_returns_dataframe(self, ai4i_csv: Path) -> None:
        df = load_ai4i(ai4i_csv)
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self, ai4i_csv: Path) -> None:
        df = load_ai4i(ai4i_csv)
        assert df.shape == (3, 14)

    def test_expected_columns(self, ai4i_csv: Path) -> None:
        df = load_ai4i(ai4i_csv)
        assert "Machine failure" in df.columns
        assert "UDI" in df.columns
        assert "Type" in df.columns

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_file.csv"
        with pytest.raises(FileNotFoundError):
            load_ai4i(missing)

    def test_machine_failure_values(self, ai4i_csv: Path) -> None:
        df = load_ai4i(ai4i_csv)
        assert set(df["Machine failure"].unique()).issubset({0, 1})

    def test_custom_separator(self, tmp_path: Path) -> None:
        """Loader accepts a tab-separated file when sep is overridden."""
        content = "col_a\tcol_b\n1\t2\n3\t4\n"
        p = tmp_path / "tab.csv"
        p.write_text(content, encoding="utf-8")
        df = load_ai4i(p, sep="\t")
        assert df.shape == (2, 2)

    def test_empty_csv_loads_zero_rows(self, empty_csv: Path) -> None:
        df = load_ai4i(empty_csv)
        assert len(df) == 0
        assert "col_a" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# SECOM loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSECOM:
    def test_returns_tuple_of_dataframes(
        self, secom_data: Path, secom_labels: Path
    ) -> None:
        features, labels = load_secom(secom_data, secom_labels)
        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.DataFrame)

    def test_features_shape(self, secom_data: Path, secom_labels: Path) -> None:
        features, _ = load_secom(secom_data, secom_labels)
        assert features.shape == (3, 4)

    def test_labels_shape(self, secom_data: Path, secom_labels: Path) -> None:
        _, labels = load_secom(secom_data, secom_labels)
        assert labels.shape == (3, 2)

    def test_labels_columns(self, secom_data: Path, secom_labels: Path) -> None:
        _, labels = load_secom(secom_data, secom_labels)
        assert list(labels.columns) == ["label", "timestamp"]

    def test_label_values(self, secom_data: Path, secom_labels: Path) -> None:
        _, labels = load_secom(secom_data, secom_labels)
        assert set(labels["label"].unique()).issubset({-1, 1})

    def test_nan_preserved_as_missing(
        self, secom_data: Path, secom_labels: Path
    ) -> None:
        features, _ = load_secom(secom_data, secom_labels)
        total_missing = features.isnull().sum().sum()
        assert total_missing == 2  # fixture has 2 NaN values

    def test_feature_columns_auto_named(
        self, secom_data: Path, secom_labels: Path
    ) -> None:
        features, _ = load_secom(secom_data, secom_labels)
        assert features.columns[0] == "feature_1"
        assert features.columns[-1] == "feature_4"

    def test_row_alignment_error(self, secom_data: Path, tmp_path: Path) -> None:
        """Mismatched row counts must raise ValueError."""
        short_labels = tmp_path / "short_labels.data"
        short_labels.write_text('-1 "01/01/2008 10:00:00"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Row count mismatch"):
            load_secom(secom_data, short_labels)

    def test_data_file_not_found(self, tmp_path: Path, secom_labels: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_secom(tmp_path / "missing.data", secom_labels)

    def test_labels_file_not_found(self, secom_data: Path, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_secom(secom_data, tmp_path / "missing_labels.data")

