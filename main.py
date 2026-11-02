"""
SemiMind AI - Main Execution Script
=====================================
Entry point for Phase 1: loads configuration, initialises logging,
loads both datasets, runs validation and profiling, saves reports, and
prints an execution summary.

Run
---
    python main.py

The script fails gracefully with actionable error messages if dataset
paths are incorrect or configuration is missing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# -- Bootstrap: make sure project root is on sys.path -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging
from src.ingestion.dataset_loader import load_ai4i_from_config, load_secom_from_config
from src.validation.dataset_validator import DatasetValidator
from src.inspection.dataset_profiler import DatasetProfiler


def main() -> int:
    """
    Phase 1 pipeline: Load -> Validate -> Profile.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    start_time = time.time()

    # -- 1. Load configuration -------------------------------------------------
    print("\n" + "=" * 70)
    print("  SemiMind AI - Phase 1: Dataset Inspection & Profiling")
    print("=" * 70 + "\n")

    try:
        cfg = load_config()
    except FileNotFoundError as exc:
        print(f"[ERROR] Configuration file not found:\n  {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] Failed to load configuration:\n  {exc}")
        return 1

    # -- 2. Initialise logging -------------------------------------------------
    setup_logging(
        log_dir=cfg.logs_dir,
        log_filename=cfg.log_filename,
        level=cfg.log_level,
        log_to_file=cfg.log_to_file,
    )
    logger = get_logger(__name__)
    logger.info("SemiMind AI Phase 1 pipeline started.")
    logger.info("Project root: %s", cfg.project_root)

    # -- 3. Verify dataset file paths ------------------------------------------
    print("Step 1/5 - Verifying dataset file paths ...")
    files_to_check = {
        "AI4I CSV": cfg.ai4i_raw_path(),
        "SECOM data": cfg.secom_data_path(),
        "SECOM labels": cfg.secom_labels_path(),
    }
    all_found = True
    for label, path in files_to_check.items():
        exists = path.is_file()
        status = "[OK]     " if exists else "[MISSING]"
        print(f"  {status}: {label} -> {path}")
        if not exists:
            all_found = False

    if not all_found:
        print(
            "\n[ERROR] One or more dataset files are missing.\n"
            "  -> Place the raw files according to configs/dataset_config.yaml\n"
            "    and ensure datasets/raw/ai4i/ and datasets/raw/secom/ exist.\n"
            "  -> If files are in a different location, update configs/dataset_config.yaml."
        )
        logger.error("Dataset file verification failed - missing files detected.")
        return 1

    # -- 4. Load datasets ------------------------------------------------------
    print("\nStep 2/5 - Loading datasets ...")
    try:
        ai4i_df = load_ai4i_from_config(cfg)
        print(f"  [OK] AI4I loaded  - shape: {ai4i_df.shape}")
    except Exception as exc:
        print(f"  [FAIL] AI4I loading failed: {exc}")
        logger.exception("AI4I loading failed.")
        return 1

    try:
        secom_features_df, secom_labels_df = load_secom_from_config(cfg)
        print(f"  [OK] SECOM loaded - features: {secom_features_df.shape}, "
              f"labels: {secom_labels_df.shape}")
    except Exception as exc:
        print(f"  [FAIL] SECOM loading failed: {exc}")
        logger.exception("SECOM loading failed.")
        return 1

    # -- 5. Run validation -----------------------------------------------------
    print("\nStep 3/5 - Running data validation ...")
    validator = DatasetValidator()

    ai4i_report = validator.validate_ai4i(ai4i_df)
    print(f"\n  AI4I Validation -> {ai4i_report.passed_count} passed, "
          f"{ai4i_report.failed_count} failed")
    if ai4i_report.errors:
        for c in ai4i_report.errors:
            print(f"    [ERROR]   {c.message}")
    if ai4i_report.warnings:
        for c in ai4i_report.warnings:
            print(f"    [WARNING] {c.message}")

    secom_report = validator.validate_secom(secom_features_df, secom_labels_df)
    print(f"\n  SECOM Validation -> {secom_report.passed_count} passed, "
          f"{secom_report.failed_count} failed")
    if secom_report.errors:
        for c in secom_report.errors:
            print(f"    [ERROR]   {c.message}")
    if secom_report.warnings:
        for c in secom_report.warnings:
            print(f"    [WARNING] {c.message}")

    # -- 6. Generate profiles and save reports ---------------------------------
    print("\nStep 4/5 - Generating dataset profiles and saving reports ...")
    profiler = DatasetProfiler(reports_root=cfg.reports_root)

    try:
        ai4i_profile = profiler.profile_ai4i(
            ai4i_df, raw_file=cfg.ai4i_raw_path()
        )
        print(f"  [OK] AI4I profile saved - {cfg.ai4i_reports_dir()}")
    except Exception as exc:
        print(f"  [FAIL] AI4I profiling failed: {exc}")
        logger.exception("AI4I profiling failed.")

    try:
        secom_profile = profiler.profile_secom(
            secom_features_df,
            secom_labels_df,
            raw_data_file=cfg.secom_data_path(),
            raw_labels_file=cfg.secom_labels_path(),
        )
        print(f"  [OK] SECOM profile saved - {cfg.secom_reports_dir()}")
    except Exception as exc:
        print(f"  [FAIL] SECOM profiling failed: {exc}")
        logger.exception("SECOM profiling failed.")

    # -- 7. Execution summary --------------------------------------------------
    elapsed = round(time.time() - start_time, 2)
    print("\nStep 5/5 -> Execution summary")
    print("-" * 70)
    print(f"  AI4I dataset")
    print(f"    Rows     : {len(ai4i_df):,}")
    print(f"    Columns  : {len(ai4i_df.columns)}")
    print(f"    Missing  : {ai4i_df.isnull().sum().sum():,}")
    print(f"    Failures : {ai4i_df['Machine failure'].sum():,} "
          f"({ai4i_df['Machine failure'].mean() * 100:.2f}%)")
    print(f"\n  SECOM dataset")
    print(f"    Rows     : {len(secom_features_df):,}")
    print(f"    Features : {len(secom_features_df.columns)}")
    print(f"    Missing  : {secom_features_df.isnull().sum().sum():,}")
    fail_count = (secom_labels_df["label"] == 1).sum()
    print(f"    Fails    : {fail_count:,} "
          f"({fail_count / len(secom_labels_df) * 100:.2f}%)")
    print(f"\n  Reports saved to : {cfg.reports_root}")
    print(f"  Logs saved to    : {cfg.logs_dir}")
    print(f"  Total elapsed    : {elapsed}s")
    print("-" * 70)
    print("  Phase 1 pipeline completed successfully.")
    print("=" * 70 + "\n")

    logger.info("Phase 1 pipeline completed in %.2f seconds.", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
