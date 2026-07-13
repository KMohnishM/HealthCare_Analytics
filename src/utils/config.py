"""
src/utils/config.py
-------------------
OmegaConf-based config loader. Loads config/config.yaml and merges
with any CLI overrides.

Usage:
    from src.utils.config import load_config
    cfg = load_config()                        # uses default config.yaml
    cfg = load_config("config/custom.yaml")   # custom path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> DictConfig:
    """
    Load project configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML config file. Defaults to
        ``<project_root>/config/config.yaml``.
    overrides : dict, optional
        Key-value pairs to override loaded config values.
        Keys use dot-notation, e.g. ``{"tabular.xgb.max_depth": 8}``.

    Returns
    -------
    DictConfig
        OmegaConf configuration object (supports dot-access).
    """
    if config_path is None:
        # Resolve relative to this file's location (src/utils/ -> project root)
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config" / "config.yaml"

    cfg: DictConfig = OmegaConf.load(str(config_path))

    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def ensure_dirs(cfg: DictConfig) -> None:
    """
    Create all output directories defined in cfg.paths if they don't exist.

    Parameters
    ----------
    cfg : DictConfig
        Loaded project configuration.
    """
    path_keys = [
        "cohort_dir",
        "processed_dir",
        "models_dir",
        "results_dir",
        "figures_dir",
        "logs_dir",
    ]
    for key in path_keys:
        p = Path(cfg.paths[key])
        p.mkdir(parents=True, exist_ok=True)
