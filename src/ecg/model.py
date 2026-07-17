"""
src/ecg/model.py
-----------------
1D ResNet for 12-lead ECG classification with MC-Dropout uncertainty.

Architecture:
    Input (B, 12, 5000)
    -> Stem Conv + MaxPool
    -> 4× ResBlock1D (64->128->256->embed_dim channels, stride-2 downsampling)
    -> AdaptiveAvgPool -> Dropout -> FC -> embed vector (B, embed_dim)

A separate head maps embed -> scalar risk score (logit).
MC-Dropout is applied at inference by enabling Dropout layers while
keeping BatchNorm in eval mode.

Usage:
    from src.ecg.model import ECGResNet, mc_predict_ecg
    model = ECGResNet(cfg)
    logit, embed = model(waveform)      # training
    result = mc_predict_ecg(model, waveform_tensor, n_passes=50)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


# ── Building blocks ──────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    1D residual block with optional stride-2 downsampling.

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    kernel : int
        Convolution kernel size (odd number recommended).
    stride : int
        Stride for the first conv (use 2 for downsampling).
    """

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1
    ) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_block(x) + self.downsample(x))


# ── Full model ───────────────────────────────────────────────────────────────

class ECGResNet(nn.Module):
    """
    1D ResNet for 12-lead ECG binary classification.

    Forward pass returns (logit, embedding):
    - logit     : (B, 1) raw score (no sigmoid) for BCEWithLogitsLoss
    - embedding : (B, embed_dim) for use in the fusion layer

    Parameters
    ----------
    cfg : DictConfig
        Project configuration (reads cfg.ecg section).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        embed_dim = cfg.ecg.embed_dim
        dropout   = cfg.ecg.dropout

        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = ResBlock1D(64,  64,  kernel=7, stride=1)
        self.layer2 = ResBlock1D(64,  128, kernel=7, stride=2)
        self.layer3 = ResBlock1D(128, 256, kernel=5, stride=2)
        self.layer4 = ResBlock1D(256, embed_dim, kernel=5, stride=2)

        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, 12, 5000)

        Returns
        -------
        logit : (B, 1)
        embed : (B, embed_dim)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        embed = self.pool(x).squeeze(-1)  # (B, embed_dim)
        embed = self.dropout(embed)
        logit = self.head(embed)          # (B, 1)
        return logit, embed


# ── MC-Dropout inference ──────────────────────────────────────────────────────

def _enable_dropout(model: nn.Module) -> None:
    """Enable only Dropout layers (keep BatchNorm in eval mode)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_predict_ecg(
    model: ECGResNet,
    waveform: torch.Tensor,
    n_passes: int = 50,
    device: str | None = None,
    global_max_std: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    MC-Dropout inference for uncertainty estimation.

    Parameters
    ----------
    model : ECGResNet
        Trained model.
    waveform : torch.Tensor, shape (B, 12, 5000)
        Preprocessed ECG waveforms.
    n_passes : int
        Number of stochastic forward passes.
    device : str, optional
        Auto-detects CUDA/CPU if None.
    global_max_std : float
        Static scaling factor for normalization. Defaults to 0.5 (theoretical limit).

    Returns
    -------
    dict with keys:
        - ``prob``       : mean predicted probability (B,)
        - ``std``        : std across passes (B,)
        - ``confidence`` : 1 − (std / global_max_std) clamped to [0,1] (B,)
        - ``embed``      : mean embedding (B, embed_dim) from last pass
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    waveform = waveform.to(device)

    model.eval()
    _enable_dropout(model)

    probs_list = []
    embeds_list = []

    with torch.no_grad():
        for _ in range(n_passes):
            logit, embed = model(waveform)
            prob = torch.sigmoid(logit).squeeze(-1)  # (B,)
            probs_list.append(prob.cpu().numpy())
            embeds_list.append(embed.cpu().numpy())

    probs  = np.stack(probs_list, axis=0)   # (T, B)
    embeds = np.stack(embeds_list, axis=0)  # (T, B, D)

    mean_prob  = probs.mean(axis=0)
    std_prob   = probs.std(axis=0)
    mean_embed = embeds.mean(axis=0)

    confidence = np.clip(1.0 - (std_prob / global_max_std), 0.0, 1.0)

    return {
        "prob":       mean_prob,
        "std":        std_prob,
        "confidence": confidence,
        "embed":      mean_embed,
    }
