"""
src/fusion/fixed_weight.py
---------------------------
Confidence-weighted average fusion baseline.

For each patient, the fused score is the weighted average of
available branch scores, where weights = branch confidences.
Missing modalities are excluded from numerator and denominator.

This serves as the interpretable baseline against which the
learned gate is evaluated in the missingness sweep.

Usage:
    from src.fusion.fixed_weight import confidence_weighted_fusion
    fused = confidence_weighted_fusion(scores, confidences, available)
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def confidence_weighted_fusion(
    scores:      Dict[str, np.ndarray],
    confidences: Dict[str, np.ndarray],
    available:   Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute confidence-weighted average fusion score.

    Parameters
    ----------
    scores : dict
        Branch risk scores. Keys are modality names
        (e.g. 'tabular', 'ecg', 'cxr'), values are (N,) arrays in [0,1].
    confidences : dict
        Branch confidence estimates (N,) arrays in [0,1].
        Same keys as ``scores``.
    available : dict
        Binary availability flags (N,) arrays in {0, 1}.
        Same keys as ``scores``.

    Returns
    -------
    np.ndarray
        Fused risk score per patient, shape (N,).
        Falls back to tabular score for patients with all modalities missing
        (should not happen in practice since tabular is always available).
    """
    modalities = list(scores.keys())
    N = len(next(iter(scores.values())))

    numerator   = np.zeros(N, dtype=np.float64)
    denominator = np.zeros(N, dtype=np.float64)

    for mod in modalities:
        s = np.array(scores[mod],      dtype=np.float64)
        c = np.array(confidences[mod], dtype=np.float64)
        a = np.array(available[mod],   dtype=np.float64)

        numerator   += a * c * s
        denominator += a * c

    # Avoid division by zero (fall back to 0.5 for truly empty rows)
    safe_denom = np.where(denominator > 1e-8, denominator, 1.0)
    fused = np.where(denominator > 1e-8, numerator / safe_denom, 0.5)

    return fused.astype(np.float32)


def fixed_fusion_predict(
    branch_results: Dict[str, Dict[str, np.ndarray]],
    avail_flags: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper that extracts scores and confidences from
    branch result dicts and calls ``confidence_weighted_fusion``.

    Parameters
    ----------
    branch_results : dict
        Keys = modality names. Values = dicts with at least:
        - ``'score'`` or ``'prob'`` (N,) — risk probability
        - ``'confidence'`` (N,) — confidence estimate

    avail_flags : dict
        Keys = modality names. Values = (N,) binary arrays.

    Returns
    -------
    dict with keys:
        - ``'score'``     : fused probability (N,)
        - ``'method'``    : 'fixed_weight'
    """
    scores      = {}
    confidences = {}

    for mod, res in branch_results.items():
        scores[mod]      = res.get("score", res.get("prob", np.zeros(1)))
        confidences[mod] = res.get("confidence", np.ones_like(scores[mod]))

    fused = confidence_weighted_fusion(scores, confidences, avail_flags)
    return {"score": fused, "method": "fixed_weight"}
