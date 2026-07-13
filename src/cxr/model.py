"""
src/cxr/model.py
-----------------
DenseNet-121 transfer learning encoder for CXR classification with MC-Dropout.

Architecture:
    DenseNet-121 backbone (frozen) → feature vector (1024-d)
    → projection head: Linear(1024 → 256) → ReLU → Dropout
    → classification head: Linear(256 → 1) logit

The backbone is loaded from timm with pretrained ImageNet weights.
Only the projection + classification heads are trained.

Usage:
    from src.cxr.model import CXREncoder, mc_predict_cxr
    model  = CXREncoder(cfg)
    logit, embed = model(image_tensor)      # (B, 1), (B, 256)
    result = mc_predict_cxr(model, images, n_passes=50)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


class CXREncoder(nn.Module):
    """
    Transfer-learning CXR encoder using DenseNet-121 from timm.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration (reads cfg.cxr section).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        import timm

        backbone_name = cfg.cxr.backbone       # "densenet121"
        embed_dim     = cfg.cxr.embed_dim      # 256
        dropout       = cfg.cxr.dropout        # 0.3

        # Load pretrained backbone without classifier
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=cfg.cxr.pretrained,
            num_classes=0,       # returns raw feature vector
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features  # 1024 for densenet121

        # Freeze backbone if configured
        if cfg.cxr.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Trainable projection head
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 224, 224)

        Returns
        -------
        logit : (B, 1)
        embed : (B, embed_dim)
        """
        feat  = self.backbone(x)   # (B, 1024)
        embed = self.proj(feat)    # (B, 256)
        logit = self.head(embed)   # (B, 1)
        return logit, embed

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights for fine-tuning (optional stage 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ── MC-Dropout inference ──────────────────────────────────────────────────────

def _enable_dropout(model: nn.Module) -> None:
    """Enable Dropout layers while keeping BatchNorm/LayerNorm in eval mode."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def mc_predict_cxr(
    model: CXREncoder,
    images: torch.Tensor,
    n_passes: int = 50,
    device: str | None = None,
) -> Dict[str, np.ndarray]:
    """
    MC-Dropout inference for uncertainty estimation on CXR.

    Parameters
    ----------
    model : CXREncoder
        Trained model.
    images : torch.Tensor, shape (B, 3, 224, 224)
        Preprocessed CXR images.
    n_passes : int
        Number of stochastic forward passes.
    device : str, optional
        Auto-detects CUDA/CPU if None.

    Returns
    -------
    dict with keys:
        - ``prob``       : mean predicted probability (B,)
        - ``std``        : std across passes (B,)
        - ``confidence`` : 1 − normalized_std (B,)
        - ``embed``      : mean embedding (B, embed_dim)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    images = images.to(device)

    model.eval()
    _enable_dropout(model)

    probs_list  = []
    embeds_list = []

    with torch.no_grad():
        for _ in range(n_passes):
            logit, embed = model(images)
            prob = torch.sigmoid(logit).squeeze(-1)
            probs_list.append(prob.cpu().numpy())
            embeds_list.append(embed.cpu().numpy())

    probs  = np.stack(probs_list,  axis=0)  # (T, B)
    embeds = np.stack(embeds_list, axis=0)  # (T, B, D)

    mean_prob  = probs.mean(axis=0)
    std_prob   = probs.std(axis=0)
    mean_embed = embeds.mean(axis=0)

    max_std    = max(std_prob.max(), 1e-8)
    confidence = 1.0 - (std_prob / max_std)

    return {
        "prob":       mean_prob,
        "std":        std_prob,
        "confidence": confidence,
        "embed":      mean_embed,
    }
