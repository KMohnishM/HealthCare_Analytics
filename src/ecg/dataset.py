"""
src/ecg/dataset.py
-------------------
PyTorch Dataset and DataLoader factory for ECG waveforms.

Usage:
    from src.ecg.dataset import ECGDataset, make_ecg_loader
    ds     = ECGDataset(ecg_index, cohort, cfg)
    loader = make_ecg_loader(ds, batch_size=64, shuffle=True)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.ecg.preprocess import load_and_preprocess_ecg
from src.utils.logger import get_logger

log = get_logger(__name__)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for 12-lead ECG waveforms.

    For patients without a qualifying ECG, returns a zero tensor and
    sets ``available = 0`` so the fusion layer can mask them out.

    Parameters
    ----------
    ecg_index : pd.DataFrame
        Output of ``build_ecg_index`` — maps hadm_id -> ecg_record_path.
    cohort : pd.DataFrame
        Cohort split with ``hadm_id`` and ``readmitted_30d``.
    cfg : DictConfig
        Project configuration.
    cache : bool
        If True, pre-load all waveforms into RAM (faster if RAM allows).
    """

    def __init__(
        self,
        ecg_index: pd.DataFrame,
        cohort: pd.DataFrame,
        cfg: DictConfig,
        cache: bool = False,
    ) -> None:
        self.cfg       = cfg
        self.target_len = cfg.ecg.target_len
        self.n_leads   = cfg.ecg.n_leads

        # Build hadm_id -> record path lookup
        self.path_map: dict[int, str] = dict(
            zip(ecg_index["hadm_id"], ecg_index["ecg_record_path"])
        )

        self.hadm_ids = cohort["hadm_id"].values
        self.labels   = cohort["readmitted_30d"].values.astype(np.float32)

        self._cache: dict[int, Optional[np.ndarray]] = {}
        if cache:
            log.info("Pre-loading ECG waveforms into cache ...")
            for hadm_id in self.hadm_ids:
                self._cache[hadm_id] = self._load(hadm_id)

    def _load(self, hadm_id: int) -> Optional[np.ndarray]:
        """Load and preprocess waveform for one admission."""
        path = self.path_map.get(hadm_id)
        if path is None:
            return None
        return load_and_preprocess_ecg(path, self.cfg)

    def __len__(self) -> int:
        return len(self.hadm_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hadm_id = int(self.hadm_ids[idx])
        label   = float(self.labels[idx])

        if hadm_id in self._cache:
            sig = self._cache[hadm_id]
        else:
            sig = self._load(hadm_id)

        if sig is not None:
            waveform  = torch.tensor(sig, dtype=torch.float32)  # (12, 5000)
            available = torch.tensor(1.0)
        else:
            waveform  = torch.zeros(self.n_leads, self.target_len, dtype=torch.float32)
            available = torch.tensor(0.0)

        return {
            "waveform":  waveform,
            "available": available,
            "label":     torch.tensor(label, dtype=torch.float32),
            "hadm_id":   torch.tensor(hadm_id, dtype=torch.long),
        }


def make_ecg_loader(
    dataset: ECGDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for an ECGDataset.

    Parameters
    ----------
    dataset : ECGDataset
    batch_size : int
    shuffle : bool
        Set True for training, False for val/test.
    num_workers : int
        0 is required on Colab (multiprocessing issues with WFDB).

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
