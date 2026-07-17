"""
src/evaluation/missingness_sweep.py
-------------------------------------
Core experiment: evaluate both fusion methods across all 7 modality combinations.

For each combination of available modalities, branch scores/confidences for
the unavailable ones are zeroed out (availability flag = 0) and both fusion
strategies are evaluated on AUROC, AUPRC, Brier, ECE.

The resulting table is the headline result of the project showing whether the
learned gate degrades more gracefully than fixed-weight fusion as modalities
are removed.

Usage:
    from src.evaluation.missingness_sweep import run_missingness_sweep
    results_df = run_missingness_sweep(branch_outputs, y_true, gate_model, cfg)
    results_df.to_csv("outputs/results/missingness_sweep.csv")
"""

from __future__ import annotations

from itertools import chain, combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.evaluation.metrics import evaluate_all
from src.fusion.fixed_weight import confidence_weighted_fusion
from src.utils.logger import get_logger

log = get_logger(__name__)

MODALITIES = ["tabular", "ecg", "cxr"]


def _all_subsets(modalities: List[str]) -> List[Tuple[str, ...]]:
    """Return all non-empty subsets of modalities (7 combinations for 3)."""
    return list(
        chain.from_iterable(
            combinations(modalities, r) for r in range(1, len(modalities) + 1)
        )
    )


def _make_availability_mask(
    subset: Tuple[str, ...],
    n: int,
    modalities: List[str] = MODALITIES,
) -> Dict[str, np.ndarray]:
    """
    Create binary availability masks for a given subset of modalities.

    Parameters
    ----------
    subset : tuple of str
        The modalities that are available.
    n : int
        Number of patients.
    modalities : list of str
        All modality names.

    Returns
    -------
    dict: modality -> (n,) binary array.
    """
    return {
        mod: np.ones(n, dtype=np.float32) if mod in subset else np.zeros(n, dtype=np.float32)
        for mod in modalities
    }


def _gated_fusion_predict(
    gate_model: torch.nn.Module,
    branch_outputs: Dict[str, Dict],
    availability_mask: Dict[str, np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """
    Run gated fusion model for a given availability mask.

    Parameters
    ----------
    gate_model : GatedFusionModel
        Trained fusion model.
    branch_outputs : dict
        Keys = modality names. Values = dicts with 'embed', 'confidence'.
    availability_mask : dict
        Binary availability per modality.
    device : str

    Returns
    -------
    np.ndarray of shape (N,) — predicted probabilities.
    """
    gate_model.eval()
    gate_model.to(device)

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).to(device)

    tab_embed = _t(branch_outputs["tabular"]["embed"])
    ecg_embed = _t(branch_outputs["ecg"]["embed"])
    cxr_embed = _t(branch_outputs["cxr"]["embed"])

    tab_conf  = _t(branch_outputs["tabular"]["confidence"]).unsqueeze(-1)
    ecg_conf  = _t(branch_outputs["ecg"]["confidence"]).unsqueeze(-1)
    cxr_conf  = _t(branch_outputs["cxr"]["confidence"]).unsqueeze(-1)

    avail = torch.stack([
        _t(availability_mask["tabular"]),
        _t(availability_mask["ecg"]),
        _t(availability_mask["cxr"]),
    ], dim=1)  # (N, 3)

    # Zero out embeddings for unavailable modalities
    ecg_embed = ecg_embed * _t(availability_mask["ecg"]).unsqueeze(-1)
    cxr_embed = cxr_embed * _t(availability_mask["cxr"]).unsqueeze(-1)
    ecg_conf  = ecg_conf  * _t(availability_mask["ecg"]).unsqueeze(-1)
    cxr_conf  = cxr_conf  * _t(availability_mask["cxr"]).unsqueeze(-1)

    with torch.no_grad():
        logit, _ = gate_model(
            tab_embed, ecg_embed, cxr_embed,
            tab_conf,  ecg_conf,  cxr_conf,
            avail,
        )
    prob = torch.sigmoid(logit.squeeze(-1)).cpu().numpy()
    return prob


def run_missingness_sweep(
    branch_outputs: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    gate_model: Optional[torch.nn.Module] = None,
    cfg: Optional[DictConfig] = None,
) -> pd.DataFrame:
    """
    Evaluate both fusion methods across all 7 modality subsets.

    Parameters
    ----------
    branch_outputs : dict
        Keys = modality names ('tabular', 'ecg', 'cxr').
        Values = dicts containing:
          - 'score'      : (N,) probability
          - 'confidence' : (N,) confidence in [0,1]
          - 'embed'      : (N, D) embedding (needed for learned gate)
    y_true : np.ndarray, shape (N,)
        Binary readmission labels.
    gate_model : GatedFusionModel, optional
        Trained learned gate. If None, only fixed-weight results reported.
    cfg : DictConfig, optional
        Project config.

    Returns
    -------
    pd.DataFrame
        Rows = one per (modality_subset × fusion_method) combination.
        Columns = subset_label, n_modalities, fusion_method, AUROC, AUPRC, Brier, ECE.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(y_true)
    subsets = _all_subsets(MODALITIES)

    records = []
    log.info("Running missingness sweep over %d modality combinations ...", len(subsets))

    for subset in subsets:
        subset_label  = "+".join(s[:3].upper() for s in subset)
        n_mod = len(subset)
        avail = _make_availability_mask(subset, n)

        # ── Fixed-weight fusion ──────────────────────────────────────────────
        fw_score = confidence_weighted_fusion(
            scores={m: branch_outputs[m]["score"] for m in MODALITIES},
            confidences={m: branch_outputs[m]["confidence"] for m in MODALITIES},
            available=avail,
        )
        fw_metrics = evaluate_all(y_true, fw_score)
        records.append({
            "subset":       subset_label,
            "modalities":   list(subset),
            "n_modalities": n_mod,
            "fusion":       "fixed_weight",
            **fw_metrics,
        })

        # ── Learned gate ─────────────────────────────────────────────────────
        if gate_model is not None:
            gate_score = _gated_fusion_predict(gate_model, branch_outputs, avail, device)
            gate_metrics = evaluate_all(y_true, gate_score)
            records.append({
                "subset":       subset_label,
                "modalities":   list(subset),
                "n_modalities": n_mod,
                "fusion":       "learned_gate",
                **gate_metrics,
            })

        log.info(
            "  %-20s | fixed_AUROC=%.4f | gate_AUROC=%s",
            subset_label,
            fw_metrics["AUROC"],
            f"{gate_metrics['AUROC']:.4f}" if gate_model else "N/A",
        )

    df = pd.DataFrame(records)
    df = df.sort_values(["n_modalities", "subset", "fusion"]).reset_index(drop=True)
    log.info("Missingness sweep complete — %d rows", len(df))
    return df


def plot_missingness_sweep(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Generate a heatmap comparing learned gate vs fixed-weight fusion
    across all modality subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``run_missingness_sweep``.
    save_path : str, optional
        Path to save the figure (PNG).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = ["AUROC", "AUPRC", "Brier", "ECE"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    subset_order = (
        df.sort_values("n_modalities")
        .drop_duplicates("subset")["subset"]
        .tolist()
    )

    for ax, metric in zip(axes, metrics):
        pivot = df.pivot(index="subset", columns="fusion", values=metric)
        pivot = pivot.reindex(subset_order)
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn" if metric == "AUROC" else "RdYlGn_r",
            ax=ax, linewidths=0.5,
            vmin=pivot.min().min() * 0.95,
            vmax=pivot.max().max() * 1.05,
        )
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Modality Subset" if ax == axes[0] else "")
        ax.set_xlabel("")

    plt.suptitle("Missingness Sweep: Learned Gate vs Fixed-Weight Fusion",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Missingness sweep figure saved -> %s", save_path)

    plt.show()
