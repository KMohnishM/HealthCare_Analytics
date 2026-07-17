"""
src/fusion/learned_gate.py
---------------------------
Learned gating fusion with modality-dropout training.

Architecture:
    Input = per-modality (embedding, confidence, availability_flag) triplets
    Gate MLP -> weights over available modalities
    Weighted sum of branch embeddings -> classifier head -> logit

Training trick: random modality dropout forces the gate to learn
how to reweight when modalities are genuinely missing at inference,
not just noisy.

Usage:
    from src.fusion.learned_gate import GatedFusionModel, train_fusion
    model = GatedFusionModel(cfg)
    train_fusion(model, train_loader, val_loader, cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


# ── Gated Fusion Model ────────────────────────────────────────────────────────

class GatedFusionModel(nn.Module):
    """
    Learned gating fusion over three modality embeddings.

    Each modality contributes: embedding (D,) + confidence (1,) + available (1,).
    The gate takes the concatenated modality representations and outputs
    a soft weight per modality; unavailable modalities are masked to -inf
    before softmax.

    Parameters
    ----------
    cfg : DictConfig
        Project config. Reads:
        - cfg.ecg.embed_dim    (D_ecg)
        - cfg.cxr.embed_dim    (D_cxr)
        - cfg.fusion.d_fuse
        - cfg.fusion.gate_hidden
        - cfg.fusion.dropout
    tab_dim : int
        Tabular embedding dimension (from XGBoost -> small FC head).
    """

    def __init__(self, cfg: DictConfig, tab_dim: int = 64) -> None:
        super().__init__()

        d_ecg  = cfg.ecg.embed_dim     # 256
        d_cxr  = cfg.cxr.embed_dim     # 256
        d_fuse = cfg.fusion.d_fuse     # 256
        d_gate = cfg.fusion.gate_hidden # 64
        drop   = cfg.fusion.dropout    # 0.4

        # Per-modality projectors to common dimension
        self.proj_tab = nn.Sequential(
            nn.Linear(tab_dim, d_fuse), nn.LayerNorm(d_fuse), nn.ReLU()
        )
        self.proj_ecg = nn.Sequential(
            nn.Linear(d_ecg, d_fuse), nn.LayerNorm(d_fuse), nn.ReLU()
        )
        self.proj_cxr = nn.Sequential(
            nn.Linear(d_cxr, d_fuse), nn.LayerNorm(d_fuse), nn.ReLU()
        )

        # Gate: takes projected features + confidence + avail for each modality
        # Input size = 3 * (d_fuse + 1 + 1) = 3 * (d_fuse + 2)
        gate_in = 3 * (d_fuse + 2)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, d_gate), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_gate, 3),      # logits over 3 modalities
        )

        # Final classifier on fused representation
        self.classifier = nn.Sequential(
            nn.Linear(d_fuse, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        tab_embed: torch.Tensor,    # (B, tab_dim)
        ecg_embed: torch.Tensor,    # (B, d_ecg)
        cxr_embed: torch.Tensor,    # (B, d_cxr)
        tab_conf: torch.Tensor,     # (B, 1)
        ecg_conf: torch.Tensor,     # (B, 1)
        cxr_conf: torch.Tensor,     # (B, 1)
        availability: torch.Tensor, # (B, 3)  — [tab_avail, ecg_avail, cxr_avail]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logit : (B, 1) — raw score for BCEWithLogitsLoss
        weights : (B, 3) — attention weights per modality
        """
        ft = self.proj_tab(tab_embed)   # (B, d_fuse)
        fe = self.proj_ecg(ecg_embed)   # (B, d_fuse)
        fc = self.proj_cxr(cxr_embed)   # (B, d_fuse)

        # Concat [embed, confidence, availability] per modality
        tab_in = torch.cat([ft, tab_conf, availability[:, 0:1]], dim=-1)
        ecg_in = torch.cat([fe, ecg_conf, availability[:, 1:2]], dim=-1)
        cxr_in = torch.cat([fc, cxr_conf, availability[:, 2:3]], dim=-1)

        gate_input = torch.cat([tab_in, ecg_in, cxr_in], dim=-1)
        gate_logits = self.gate(gate_input)  # (B, 3)

        # Mask unavailable modalities: set their logit to -1e9
        mask = (availability < 0.5)          # (B, 3) — True where unavailable
        gate_logits = gate_logits.masked_fill(mask, -1e9)
        weights = torch.softmax(gate_logits, dim=-1)  # (B, 3)

        # Weighted sum of projected embeddings
        stacked = torch.stack([ft, fe, fc], dim=1)    # (B, 3, d_fuse)
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_fuse)

        logit = self.classifier(fused)   # (B, 1)
        return logit, weights


# ── Modality-dropout augmentation ────────────────────────────────────────────

def apply_modality_dropout(
    tab_embed: torch.Tensor,
    ecg_embed: torch.Tensor,
    cxr_embed: torch.Tensor,
    tab_conf: torch.Tensor,
    ecg_conf: torch.Tensor,
    cxr_conf: torch.Tensor,
    availability: torch.Tensor,
    p_drop: float = 0.3,
) -> Tuple[torch.Tensor, ...]:
    """
    Randomly zero out modalities during training to simulate missingness.

    Tabular is NEVER dropped (always available assumption).
    ECG and CXR are each dropped with probability ``p_drop``.
    At least one neural modality may be dropped per sample.

    Parameters
    ----------
    p_drop : float
        Probability of masking each of ECG and CXR.

    Returns
    -------
    Augmented versions of all six tensors + updated availability.
    """
    B = tab_embed.shape[0]
    device = tab_embed.device

    # Draw drop masks for ECG and CXR
    ecg_keep = (torch.rand(B, device=device) > p_drop).float().unsqueeze(-1)  # (B,1)
    cxr_keep = (torch.rand(B, device=device) > p_drop).float().unsqueeze(-1)

    # Zero embeddings and confidence for dropped modalities
    ecg_embed_aug = ecg_embed * ecg_keep
    cxr_embed_aug = cxr_embed * cxr_keep
    ecg_conf_aug  = ecg_conf  * ecg_keep
    cxr_conf_aug  = cxr_conf  * cxr_keep

    # Update availability mask
    avail_aug = availability.clone()
    avail_aug[:, 1] = avail_aug[:, 1] * ecg_keep.squeeze(-1)  # ECG
    avail_aug[:, 2] = avail_aug[:, 2] * cxr_keep.squeeze(-1)  # CXR

    return (
        tab_embed, ecg_embed_aug, cxr_embed_aug,
        tab_conf, ecg_conf_aug, cxr_conf_aug,
        avail_aug,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_fusion(
    model: GatedFusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    save_path: Optional[str] = None,
) -> Dict[str, list]:
    """
    Train the gated fusion model with modality-dropout augmentation.

    The DataLoader is expected to yield batches containing:
    tab_embed, ecg_embed, cxr_embed, tab_conf, ecg_conf, cxr_conf,
    availability (B,3), labels (B,).

    Parameters
    ----------
    model : GatedFusionModel
    train_loader : DataLoader
    val_loader : DataLoader
    cfg : DictConfig
    save_path : str, optional
        Where to save the best checkpoint.

    Returns
    -------
    dict
        Training history: {'train_loss', 'val_loss', 'val_auroc'}.
    """
    set_seed(cfg.fusion.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg.fusion.lr),
        weight_decay=float(cfg.fusion.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.fusion.epochs, eta_min=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_val_auroc = 0.0
    best_state = None

    for epoch in range(1, cfg.fusion.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            (tab_e, ecg_e, cxr_e,
             tab_c, ecg_c, cxr_c,
             avail, labels) = [b.to(device) for b in batch]

            # Apply modality dropout augmentation during training
            (tab_e, ecg_e, cxr_e,
             tab_c, ecg_c, cxr_c, avail) = apply_modality_dropout(
                tab_e, ecg_e, cxr_e,
                tab_c, ecg_c, cxr_c,
                avail, p_drop=cfg.fusion.modality_drop_p,
            )

            optimizer.zero_grad()
            logit, _ = model(tab_e, ecg_e, cxr_e, tab_c, ecg_c, cxr_c, avail)
            loss = criterion(logit.squeeze(-1), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        val_loss, val_auroc = _eval_fusion(model, val_loader, criterion, device)
        avg_train_loss = total_loss / len(train_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_AUROC=%.4f",
                epoch, cfg.fusion.epochs, avg_train_loss, val_loss, val_auroc,
            )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Loaded best model (val_AUROC=%.4f)", best_val_auroc)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "history": history}, path)
        log.info("Fusion model saved -> %s", path)

    return history


def _eval_fusion(
    model: GatedFusionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate fusion model on a dataloader. Returns (loss, AUROC)."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            (tab_e, ecg_e, cxr_e,
             tab_c, ecg_c, cxr_c,
             avail, labels) = [b.to(device) for b in batch]
            logit, _ = model(tab_e, ecg_e, cxr_e, tab_c, ecg_c, cxr_c, avail)
            loss = criterion(logit.squeeze(-1), labels)
            total_loss += loss.item()
            all_logits.append(logit.squeeze(-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_score = np.concatenate(all_logits)
    y_true  = np.concatenate(all_labels)
    probs   = 1 / (1 + np.exp(-y_score))

    try:
        auroc = roc_auc_score(y_true, probs)
    except Exception:
        auroc = 0.5

    return total_loss / len(loader), auroc


def make_fusion_dataset(
    tab_embeds:   np.ndarray,
    ecg_embeds:   np.ndarray,
    cxr_embeds:   np.ndarray,
    tab_confs:    np.ndarray,
    ecg_confs:    np.ndarray,
    cxr_confs:    np.ndarray,
    availability: np.ndarray,
    labels:       np.ndarray,
) -> TensorDataset:
    """
    Assemble a TensorDataset from pre-computed branch outputs.

    Parameters
    ----------
    tab_embeds : (N, D_tab)
    ecg_embeds : (N, D_ecg)
    cxr_embeds : (N, D_cxr)
    tab_confs  : (N,)
    ecg_confs  : (N,)
    cxr_confs  : (N,)
    availability : (N, 3) — [tab_avail, ecg_avail, cxr_avail]
    labels     : (N,)

    Returns
    -------
    TensorDataset of all 8 tensors.
    """
    def t(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(np.array(a), dtype=dtype)

    return TensorDataset(
        t(tab_embeds),
        t(ecg_embeds),
        t(cxr_embeds),
        t(tab_confs).unsqueeze(-1),
        t(ecg_confs).unsqueeze(-1),
        t(cxr_confs).unsqueeze(-1),
        t(availability),
        t(labels),
    )
