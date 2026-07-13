"""
src/utils/seed.py
-----------------
Global random seed setter for reproducibility across
Python, NumPy, PyTorch (CPU + CUDA), and XGBoost.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility.

    Covers: Python ``random``, NumPy, PyTorch CPU, PyTorch CUDA,
    and sets the ``PYTHONHASHSEED`` environment variable.

    Parameters
    ----------
    seed : int
        The global random seed. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic CUDA ops (slight performance cost)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
