"""
src/utils/logger.py
-------------------
Logging setup with MLflow integration.

Usage:
    from src.utils.logger import get_logger, init_mlflow
    log = get_logger(__name__)
    run_id = init_mlflow(cfg, run_name="train_tabular")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a named logger with a consistent format.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def init_mlflow(cfg: DictConfig, run_name: Optional[str] = None) -> str:
    """
    Initialise an MLflow run.

    Parameters
    ----------
    cfg : DictConfig
        Loaded project config (must have ``cfg.mlflow`` section).
    run_name : str, optional
        Human-readable run name.

    Returns
    -------
    str
        Active MLflow run ID.
    """
    try:
        import mlflow

        Path(cfg.mlflow.tracking_uri).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        run = mlflow.start_run(run_name=run_name)
        return run.info.run_id
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("MLflow not installed — skipping experiment tracking.")
        return "no_mlflow"
