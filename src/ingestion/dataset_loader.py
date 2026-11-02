"""
SemiMind AI - Dataset Loader
==============================
Provides reusable, configuration-driven loaders for the two Phase 1
datasets:

* **AI4I 2020 Predictive Maintenance Dataset** (single CSV file)
* **SECOM Semiconductor Manufacturing Dataset** (space-delimited feature
  matrix + separate labels file)

Design principles
-----------------
* Never modifies raw dataset files.
* Accepts configurable paths -- no hardcoded absolute paths.
* Detects file existence before attempting to read.
* Handles encoding errors safely.
* Supports configurable separator, header row, and encoding.
* Returns plain ``pandas.DataFrame`` objects.
* Emits structured log messages at each step.

Discovered file formats (Phase 1 inspection)
---------------------------------------------
AI4I:
    - File      : ai4i2020.csv
    - Delimiter : comma (``,``)
    - Header    : row 0
    - Rows      : 10 000 (excluding header)
    - Columns   : 14

SECOM:
    - Feature file  : secom.data  (space-delimited, no header, 1 567 × 590)
    - Labels file   : secom_labels.data (space-delimited, no header,
                      columns: label [-1 / 1], timestamp [quoted string])
    - Missing values: represented as ``NaN`` strings in the raw files
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _verify_file(path: Path, label: str) -> None:
    """
    Raise ``FileNotFoundError`` with a helpful message if *path* is missing.

    Parameters
    ----------
    path : Path
        File path to verify.
    label : str
        Human-readable description used in the error message.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[{label}] Dataset file not found: {path}\n"
            f"  -> Place the raw file at the expected location and re-run."
        )
    if not path.is_file():
        raise IsADirectoryError(
            f"[{label}] Expected a file but found a directory: {path}"
        )


def _read_csv_safe(
    path: Path,
    sep: str = ",",
    header: Optional[int] = 0,
    encoding: str = "utf-8",
    na_values: Optional[list] = None,
    label: str = "dataset",
) -> pd.DataFrame:
    """
    Read a CSV/delimited text file into a DataFrame with safe error handling.

    Parameters
    ----------
    path : Path
        Absolute path to the file.
    sep : str
        Column delimiter.
    header : int or None
        Row number to use as column names, or ``None`` if there is no header.
    encoding : str
        File encoding (e.g. ``'utf-8'``, ``'latin-1'``).
    na_values : list, optional
        Additional strings to recognise as ``NaN``.
    label : str
        Dataset label for log messages.

    Returns
    -------
    pd.DataFrame
    """
    _verify_file(path, label)
    logger.info("[%s] Reading file: %s (sep=%r, header=%s, encoding=%s)",
                label, path, sep, header, encoding)
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            header=header,
            encoding=encoding,
            na_values=na_values or [],
            keep_default_na=True,
            low_memory=False,
        )
    except UnicodeDecodeError:
        logger.warning(
            "[%s] UTF-8 decoding failed - retrying with 'latin-1'.", label
        )
        df = pd.read_csv(
            path,
            sep=sep,
            header=header,
            encoding="latin-1",
            na_values=na_values or [],
            keep_default_na=True,
            low_memory=False,
        )
    logger.info("[%s] Loaded -> shape: %s", label, df.shape)
    return df


# -----------------------------------------------------------------------------
# Public loaders
# -----------------------------------------------------------------------------

def load_ai4i(
    file_path: str | Path,
    sep: str = ",",
    header: int = 0,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load the AI4I 2020 Predictive Maintenance dataset.

    The raw file is a standard comma-delimited CSV with a single header row.

    Parameters
    ----------
    file_path : str or Path
        Absolute or relative path to ``ai4i2020.csv``.
    sep : str
        Delimiter character.  Default: ``','``.
    header : int
        Index of the header row.  Default: ``0``.
    encoding : str
        File encoding.  Default: ``'utf-8'``.

    Returns
    -------
    pd.DataFrame
        Raw AI4I DataFrame with all original columns preserved.

    Columns (discovered during Phase 1 inspection)
    -----------------------------------------------
    UDI, Product ID, Type,
    Air temperature [K], Process temperature [K],
    Rotational speed [rpm], Torque [Nm], Tool wear [min],
    Machine failure, TWF, HDF, PWF, OSF, RNF
    """
    path = Path(file_path)
    df = _read_csv_safe(
        path=path,
        sep=sep,
        header=header,
        encoding=encoding,
        label="AI4I",
    )
    logger.info(
        "[AI4I] Dataset summary -> rows: %d, cols: %d, missing: %d",
        len(df), len(df.columns), df.isnull().sum().sum(),
    )
    return df


def load_secom(
    data_path: str | Path,
    labels_path: str | Path,
    sep: str = " ",
    encoding: str = "utf-8",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the SECOM Semiconductor Manufacturing dataset.

    The dataset consists of **two** space-delimited files with no column
    headers:

    * **Feature file** (``secom.data``) - 1 567 rows × 590 float columns.
      Missing values are stored as the string ``NaN``.
    * **Labels file** (``secom_labels.data``) - 1 567 rows with columns
      ``[label, timestamp]``.  Labels: ``-1`` = pass, ``1`` = fail.

    Row alignment between the two files is preserved; this function does
    **not** merge them so downstream code retains full control.

    Parameters
    ----------
    data_path : str or Path
        Absolute or relative path to ``secom.data``.
    labels_path : str or Path
        Absolute or relative path to ``secom_labels.data``.
    sep : str
        Delimiter character.  Default: ``' '`` (single space).
    encoding : str
        File encoding.  Default: ``'utf-8'``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(features_df, labels_df)``

        * ``features_df`` - shape *(1 567, 590)*.  Column names are
          auto-generated as ``feature_001 ... feature_590``.
        * ``labels_df``   - shape *(1 567, 2)*.  Columns: ``label``
          (int), ``timestamp`` (str).

    Raises
    ------
    ValueError
        If the number of rows in the feature file and labels file differ.
    """
    data_path = Path(data_path)
    labels_path = Path(labels_path)

    # -- Load feature matrix ---------------------------------------------------
    features_df = _read_csv_safe(
        path=data_path,
        sep=sep,
        header=None,
        encoding=encoding,
        na_values=["NaN"],
        label="SECOM-data",
    )

    # Auto-generate readable column names: feature_001, feature_002, ...
    n_features = features_df.shape[1]
    width = len(str(n_features))
    features_df.columns = [f"feature_{str(i + 1).zfill(width)}" for i in range(n_features)]
    logger.info("[SECOM-data] Columns renamed to feature_001 ... feature_%s", str(n_features).zfill(width))

    # -- Load labels -----------------------------------------------------------
    labels_df = _read_csv_safe(
        path=labels_path,
        sep=sep,
        header=None,
        encoding=encoding,
        label="SECOM-labels",
    )
    labels_df.columns = ["label", "timestamp"]
    # Strip surrounding quotes from timestamp if present
    labels_df["timestamp"] = labels_df["timestamp"].str.strip('"')
    logger.info(
        "[SECOM-labels] Label distribution: %s",
        labels_df["label"].value_counts().to_dict(),
    )

    # -- Validate row alignment ------------------------------------------------
    if len(features_df) != len(labels_df):
        raise ValueError(
            f"[SECOM] Row count mismatch - features: {len(features_df)}, "
            f"labels: {len(labels_df)}. "
            f"Files must have identical row counts for alignment."
        )
    logger.info(
        "[SECOM] Row alignment verified -> %d rows in both files.", len(features_df)
    )

    logger.info(
        "[SECOM] Dataset summary -> features shape: %s, labels shape: %s, "
        "total missing: %d",
        features_df.shape, labels_df.shape, features_df.isnull().sum().sum(),
    )
    return features_df, labels_df


def load_ai4i_from_config(cfg: object) -> pd.DataFrame:
    """
    Convenience wrapper that loads the AI4I dataset using a ``Config`` object.

    Parameters
    ----------
    cfg : Config
        A ``Config`` instance returned by ``src.utils.config.load_config``.

    Returns
    -------
    pd.DataFrame
    """
    ai4i_cfg = cfg.ai4i  # type: ignore[attr-defined]
    return load_ai4i(
        file_path=cfg.ai4i_raw_path(),  # type: ignore[attr-defined]
        sep=ai4i_cfg.get("separator", ","),
        header=ai4i_cfg.get("header", 0),
        encoding=ai4i_cfg.get("encoding", "utf-8"),
    )


def load_secom_from_config(cfg: object) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper that loads the SECOM dataset using a ``Config`` object.

    Parameters
    ----------
    cfg : Config
        A ``Config`` instance returned by ``src.utils.config.load_config``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(features_df, labels_df)``
    """
    secom_cfg = cfg.secom  # type: ignore[attr-defined]
    return load_secom(
        data_path=cfg.secom_data_path(),    # type: ignore[attr-defined]
        labels_path=cfg.secom_labels_path(), # type: ignore[attr-defined]
        sep=secom_cfg.get("separator", " "),
        encoding=secom_cfg.get("encoding", "utf-8"),
    )
