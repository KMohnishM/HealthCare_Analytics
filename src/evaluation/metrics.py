"""
src/evaluation/metrics.py
--------------------------
Core evaluation metrics: AUROC, AUPRC, Brier Score, ECE.

All functions accept numpy arrays and return scalar floats.

Usage:
    from src.evaluation.metrics import evaluate_all
    results = evaluate_all(y_true, y_prob)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area Under the Precision-Recall Curve (average precision)."""
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score: mean squared error of probability predictions."""
    return float(brier_score_loss(y_true, y_prob))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / N) × |accuracy(B_b) − confidence(B_b)|

    Parameters
    ----------
    y_true : np.ndarray of shape (N,), binary.
    y_prob : np.ndarray of shape (N,), predicted probabilities in [0,1].
    n_bins : int
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        ECE in [0, 1]. Lower is better.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])  # 0..n_bins-1

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return ece


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute data for a reliability (calibration) diagram.

    Returns
    -------
    dict with keys:
        - 'bin_centers' : midpoint of each bin
        - 'fraction_pos': empirical positive fraction per bin
        - 'mean_pred'   : mean predicted probability per bin
        - 'bin_counts'  : number of samples in each bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])

    centers, fracs, means, counts = [], [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        centers.append((bins[b] + bins[b + 1]) / 2)
        fracs.append(float(y_true[mask].mean()))
        means.append(float(y_prob[mask].mean()))
        counts.append(int(mask.sum()))

    return {
        "bin_centers":  np.array(centers),
        "fraction_pos": np.array(fracs),
        "mean_pred":    np.array(means),
        "bin_counts":   np.array(counts),
    }


def evaluate_all(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute all core metrics in one call.

    Parameters
    ----------
    y_true : np.ndarray, binary labels.
    y_prob : np.ndarray, predicted probabilities in [0,1].
    n_bins : int, ECE bins.
    prefix : str, optional prefix for metric keys.

    Returns
    -------
    dict with keys: AUROC, AUPRC, Brier, ECE (prefixed if requested).
    """
    p = prefix + "_" if prefix else ""
    return {
        f"{p}AUROC": auroc(y_true, y_prob),
        f"{p}AUPRC": auprc(y_true, y_prob),
        f"{p}Brier": brier(y_true, y_prob),
        f"{p}ECE":   expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }
