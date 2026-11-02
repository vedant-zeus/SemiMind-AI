"""
SemiMind AI - Logger Utility
============================
Configures a project-wide Python logger that writes to both the console
and a rotating log file under the ``logs/`` directory.

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Dataset loaded successfully.")
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


# -- Default values (overridden by config at runtime) -------------------------
# logger.py lives at src/utils/logger.py -> parents[2] is the project root
_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_DEFAULT_LOG_FILE = "semimind_phase1.log"
_DEFAULT_LEVEL = logging.INFO
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track whether the root logger has been configured to avoid duplicate handlers.
_configured = False


def setup_logging(
    log_dir: str | Path | None = None,
    log_filename: str = _DEFAULT_LOG_FILE,
    level: str | int = "INFO",
    log_to_file: bool = True,
) -> None:
    """
    Configure the root logger for the SemiMind AI project.

    Parameters
    ----------
    log_dir : str or Path, optional
        Directory where log files are written.  Defaults to ``logs/`` at the
        project root.
    log_filename : str
        Name of the log file.
    level : str or int
        Logging level (``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``).
    log_to_file : bool
        When *True*, attach a rotating file handler in addition to the console
        handler.
    """
    global _configured
    if _configured:
        return

    # Resolve numeric level
    numeric_level: int = (
        level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # -- Console handler -------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # -- File handler ----------------------------------------------------------
    if log_to_file:
        resolved_log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = resolved_log_dir / log_filename

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    If ``setup_logging`` has not been called yet, this function initialises
    logging with default settings so callers never receive an unconfigured
    logger.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
