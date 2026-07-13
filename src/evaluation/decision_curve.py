"""
src/evaluation/decision_curve.py
----------------------------------
Decision Curve Analysis (DCA): net clinical benefit vs treat-all / treat-none
over a range of probability thresholds.

DCA answers: "At a given risk threshold, does using this model to decide
who to treat produce better outcomes than treating everyone or no one?"

Net Benefit = TP/N − FP/N × (pt / (1 − pt))

Usage:
    from src.evaluation.decision_curve import run_dca, plot_dca
    dca_df = run_dca(y_true, predictions_dict)
    plot_dca(dca_df, save_path="outputs/figures/dca.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """
    Net benefit of a model at a single probability threshold.

    NB = TP/N − FP/N × (pt / (1 − pt))

    Parameters
    ----------
    y_true   : (N,) binary labels.
    y_prob   : (N,) predicted probabilities.
    threshold: Decision threshold pt ∈ (0, 1).

    Returns
    -------
    float — net benefit (can be negative).
    """
    if threshold <= 0.0 or threshold >= 1.0:
        return float("nan")
    y_pred = (y_prob >= threshold).astype(int)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    n  = float(len(y_true))
    return tp / n - fp / n * (threshold / (1.0 - threshold))


def treat_all_net_benefit(
    y_true: np.ndarray,
    threshold: float,
) -> float:
    """Net benefit of the 'treat all' strategy at threshold pt."""
    if threshold <= 0.0 or threshold >= 1.0:
        return float("nan")
    prevalence = float(y_true.mean())
    return prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold))


def run_dca(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    thresh_min: float = 0.0,
    thresh_max: float = 0.5,
    thresh_step: float = 0.01,
) -> pd.DataFrame:
    """
    Run DCA for multiple prediction models.

    Parameters
    ----------
    y_true : np.ndarray, binary labels.
    predictions : dict
        Keys = model/strategy names, values = (N,) probability arrays.
    thresh_min, thresh_max, thresh_step : float
        Threshold range parameters.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, model_name, net_benefit.
        Also includes 'treat_all' and 'treat_none' strategies.
    """
    thresholds = np.arange(thresh_min + thresh_step, thresh_max, thresh_step)
    records = []

    for pt in thresholds:
        # Treat none always = 0
        records.append({"threshold": pt, "strategy": "treat_none", "net_benefit": 0.0})
        # Treat all
        records.append({
            "threshold":   pt,
            "strategy":    "treat_all",
            "net_benefit": treat_all_net_benefit(y_true, pt),
        })
        # Models
        for name, probs in predictions.items():
            records.append({
                "threshold":   pt,
                "strategy":    name,
                "net_benefit": net_benefit(y_true, probs, pt),
            })

    return pd.DataFrame(records)


def plot_dca(
    dca_df: pd.DataFrame,
    title: str = "Decision Curve Analysis",
    save_path: Optional[str] = None,
    y_limits: tuple[float, float] = (-0.05, 0.30),
) -> None:
    """
    Plot net benefit curves for all strategies.

    Parameters
    ----------
    dca_df : pd.DataFrame
        Output of ``run_dca``.
    title : str
    save_path : str, optional
    y_limits : tuple
        Y-axis range for the plot.
    """
    import matplotlib.pyplot as plt

    strategies = dca_df["strategy"].unique()

    style_map = {
        "treat_all":   {"color": "gray",   "linestyle": "--", "lw": 1.5},
        "treat_none":  {"color": "black",  "linestyle": ":",  "lw": 1.5},
        "lace":        {"color": "#e74c3c", "linestyle": "-",  "lw": 2.0},
        "hospital":    {"color": "#e67e22", "linestyle": "-",  "lw": 2.0},
        "tabular_only":{"color": "#3498db", "linestyle": "-",  "lw": 2.0},
        "fixed_weight":{"color": "#9b59b6", "linestyle": "-",  "lw": 2.0},
        "learned_gate":{"color": "#27ae60", "linestyle": "-",  "lw": 2.5},
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for strategy in strategies:
        subset = dca_df[dca_df["strategy"] == strategy]
        style  = style_map.get(strategy, {"color": "steelblue", "linestyle": "-", "lw": 1.5})
        ax.plot(
            subset["threshold"],
            subset["net_benefit"],
            label=strategy.replace("_", " ").title(),
            **style,
        )

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlim(dca_df["threshold"].min(), dca_df["threshold"].max())
    ax.set_ylim(y_limits)
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit",           fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
