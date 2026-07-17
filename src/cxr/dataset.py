"""
src/cxr/dataset.py
-------------------
PyTorch Dataset and DataLoader factory for chest radiographs.

Usage:
    from src.cxr.dataset import CXRDataset, make_cxr_loader
    ds     = CXRDataset(cxr_index, cohort, cfg, is_train=True)
    loader = make_cxr_loader(ds, batch_size=32, shuffle=True)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.cxr.preprocess import load_and_preprocess_cxr
from src.utils.logger import get_logger

log = get_logger(__name__)


class CXRDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-CXR chest radiographs.

    For patients without a qualifying CXR, returns a zero tensor
    and sets ``available = 0``.

    Parameters
    ----------
    cxr_index : pd.DataFrame
        Output of ``build_cxr_index`` — maps hadm_id -> cxr_path.
    cohort : pd.DataFrame
        Cohort split with ``hadm_id`` and ``readmitted_30d``.
    cfg : DictConfig
        Project configuration.
    is_train : bool
        Enables augmentation transforms when True.
    """

    def __init__(
        self,
        cxr_index: pd.DataFrame,
        cohort: pd.DataFrame,
        cfg: DictConfig,
        is_train: bool = False,
    ) -> None:
        self.cfg        = cfg
        self.is_train   = is_train
        self.image_size = cfg.cxr.image_size

        self.path_map: dict[int, str] = dict(
            zip(cxr_index["hadm_id"], cxr_index["cxr_path"])
        )
        self.hadm_ids = cohort["hadm_id"].values
        self.labels   = cohort["readmitted_30d"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.hadm_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hadm_id = int(self.hadm_ids[idx])
        label   = float(self.labels[idx])

        path = self.path_map.get(hadm_id)
        if path is not None:
            img = load_and_preprocess_cxr(
                path, is_train=self.is_train, image_size=self.image_size
            )
        else:
            img = None

        if img is not None:
            image     = img                        # (3, H, W)
            available = torch.tensor(1.0)
        else:
            image     = torch.zeros(3, self.image_size, self.image_size)
            available = torch.tensor(0.0)

        return {
            "image":     image,
            "available": available,
            "label":     torch.tensor(label, dtype=torch.float32),
            "hadm_id":   torch.tensor(hadm_id, dtype=torch.long),
        }


def make_cxr_loader(
    dataset: CXRDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    """
    Create a DataLoader for CXRDataset.

    Parameters
    ----------
    dataset : CXRDataset
    batch_size : int
    shuffle : bool
    num_workers : int
        Use 0 on Colab if you encounter multiprocessing errors.

    Returns
    -------
    DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
