"""
SemiMind AI – Configuration Utility
=====================================
Loads and validates ``configs/dataset_config.yaml`` and provides a single
``Config`` dataclass that is reused across all modules.

Usage
-----
    from src.utils.config import load_config
    cfg = load_config()
    print(cfg.raw_data_root)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Load .env if present (silently skip when absent)
load_dotenv(override=False)

# Project root is two levels above this file:
# src/utils/config.py → src/utils → src → project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"


# ── Sub-configuration dataclasses ────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    description: str
    source: str
    encoding: str = "utf-8"
    separator: str = ","
    header: Optional[int] = 0
    target_column: Optional[str] = None
    # Fields used only for specific datasets (populated dynamically)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Top-level project configuration."""
    project_root: Path
    raw_data_root: Path
    processed_data_root: Path
    reports_root: Path
    logs_dir: Path
    log_level: str
    log_to_file: bool
    log_filename: str
    ai4i: Dict[str, Any]
    secom: Dict[str, Any]
    _raw: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def ai4i_raw_path(self) -> Path:
        """Absolute path to the AI4I raw CSV file."""
        return self.raw_data_root / self.ai4i["raw_file"]

    def secom_data_path(self) -> Path:
        """Absolute path to the SECOM feature data file."""
        return self.raw_data_root / self.secom["raw_data_file"]

    def secom_labels_path(self) -> Path:
        """Absolute path to the SECOM labels file."""
        return self.raw_data_root / self.secom["raw_labels_file"]

    def ai4i_reports_dir(self) -> Path:
        """Absolute path to the AI4I reports directory."""
        return self.reports_root / "ai4i"

    def secom_reports_dir(self) -> Path:
        """Absolute path to the SECOM reports directory."""
        return self.reports_root / "secom"


def load_config(config_path: Optional[Path | str] = None) -> Config:
    """
    Load and parse the YAML configuration file.

    Parameters
    ----------
    config_path : Path or str, optional
        Path to the YAML config file.  Defaults to
        ``configs/dataset_config.yaml`` at the project root.

    Returns
    -------
    Config
        Populated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    ValueError
        If the YAML file is malformed or missing required keys.
    """
    resolved = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {resolved}\n"
            f"Expected location: {DEFAULT_CONFIG_PATH}"
        )

    with resolved.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    if not raw:
        raise ValueError(f"Configuration file is empty: {resolved}")

    # ── Validate required top-level keys ─────────────────────────────────────
    required_keys = {"datasets", "paths", "logging"}
    missing = required_keys - raw.keys()
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    paths_cfg = raw["paths"]
    log_cfg = raw["logging"]

    # ── Resolve absolute paths ────────────────────────────────────────────────
    # Allow SEMIMIND_DATA_ROOT env-var to override the data root prefix
    data_root_override = os.getenv("SEMIMIND_DATA_ROOT")

    def _abs(rel: str) -> Path:
        base = Path(data_root_override) if data_root_override else PROJECT_ROOT
        p = base / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    return Config(
        project_root=PROJECT_ROOT,
        raw_data_root=_abs(paths_cfg["raw_data_root"]),
        processed_data_root=_abs(paths_cfg["processed_data_root"]),
        reports_root=_abs(paths_cfg["reports_root"]),
        logs_dir=_abs(paths_cfg["logs_dir"]),
        log_level=log_cfg.get("level", "INFO"),
        log_to_file=log_cfg.get("log_to_file", True),
        log_filename=log_cfg.get("log_filename", "semimind_phase1.log"),
        ai4i=raw["datasets"]["ai4i"],
        secom=raw["datasets"]["secom"],
        _raw=raw,
    )
