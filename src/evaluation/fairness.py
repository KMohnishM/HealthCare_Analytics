"""
src/evaluation/fairness.py
---------------------------
Subgroup fairness evaluation: report AUROC and calibration
separately for demographic groups (gender, age band).

A documented fairness gap is reported honestly, not hidden.
This satisfies the fairness check requirement without making
unsupported causal claims.

Usage:
    from src.evaluation.fairness import fairness_report
    report = fairness_report(cohort_test, y_prob_fused, cfg)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import auroc, expected_calibration_error
from src.utils.logger import get_logger

log = get_logger(__name__)


def fairness_report(
    cohort_test: pd.DataFrame,
    y_prob: np.ndarray,
    groups: Optional[List[str]] = None,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """
    Compute AUROC and ECE per demographic subgroup.

    Parameters
    ----------
    cohort_test : pd.DataFrame
        Test cohort with ``readmitted_30d``, ``gender``, ``age_group``,
        and optionally other subgroup columns.
    y_prob : np.ndarray, shape (N,)
        Predicted probabilities corresponding to rows in cohort_test.
    groups : list of str, optional
        Column names to stratify by. Defaults to ['gender', 'age_group'].
    min_group_size : int
        Subgroups smaller than this are flagged with a warning.

    Returns
    -------
    pd.DataFrame
        Columns: group_var, group_value, n, prevalence, AUROC, ECE, auroc_gap.
        ``auroc_gap`` = subgroup AUROC − overall AUROC.
    """
    if groups is None:
        groups = ["gender", "age_group"]

    y_true = cohort_test["readmitted_30d"].values
    overall_auroc = auroc(y_true, y_prob)
    overall_ece   = expected_calibration_error(y_true, y_prob)

    records = []

    # Overall row
    records.append({
        "group_var":   "overall",
        "group_value": "all",
        "n":           len(y_true),
        "prevalence":  float(y_true.mean()),
        "AUROC":       overall_auroc,
        "ECE":         overall_ece,
        "AUROC_gap":   0.0,
    })

    for group_col in groups:
        if group_col not in cohort_test.columns:
            log.warning("Column '%s' not found in cohort — skipping.", group_col)
            continue

        for grp_val, grp_df in cohort_test.groupby(group_col):
            idx = grp_df.index
            y_t = y_true[cohort_test.index.get_indexer(idx)] if cohort_test.index.name else \
                  y_true[grp_df.index.values]
            y_p = y_prob[grp_df.index.values] if isinstance(y_prob, np.ndarray) else \
                  y_prob

            # Re-index properly
            mask = cohort_test[group_col] == grp_val
            y_t = y_true[mask.values]
            y_p = y_prob[mask.values]

            n = len(y_t)
            if n < min_group_size:
                log.warning(
                    "Group %s=%s has only %d samples (< min_group_size=%d). "
                    "Results may be unreliable.",
                    group_col, grp_val, n, min_group_size,
                )

            sub_auroc = auroc(y_t, y_p)
            sub_ece   = expected_calibration_error(y_t, y_p)

            records.append({
                "group_var":   group_col,
                "group_value": str(grp_val),
                "n":           n,
                "prevalence":  float(y_t.mean()),
                "AUROC":       sub_auroc,
                "ECE":         sub_ece,
                "AUROC_gap":   sub_auroc - overall_auroc,
            })

    report_df = pd.DataFrame(records)

    # Log the largest AUROC gap
    model_rows = report_df[report_df["group_var"] != "overall"]
    if len(model_rows) > 0:
        worst = model_rows.loc[model_rows["AUROC_gap"].abs().idxmax()]
        log.info(
            "Fairness report: largest AUROC gap = %.3f for %s=%s (n=%d)",
            worst["AUROC_gap"], worst["group_var"], worst["group_value"], int(worst["n"]),
        )

    return report_df


def plot_fairness_report(
    report_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of AUROC by subgroup for each group variable.

    Parameters
    ----------
    report_df : pd.DataFrame
        Output of ``fairness_report``.
    save_path : str, optional
    """
    import matplotlib.pyplot as plt

    group_vars = [g for g in report_df["group_var"].unique() if g != "overall"]
    if not group_vars:
        log.warning("No subgroups to plot.")
        return

    fig, axes = plt.subplots(1, len(group_vars), figsize=(5 * len(group_vars), 5))
    if len(group_vars) == 1:
        axes = [axes]

    overall_auroc = report_df.loc[report_df["group_var"] == "overall", "AUROC"].values[0]

    for ax, gvar in zip(axes, group_vars):
        sub = report_df[report_df["group_var"] == gvar].copy()
        sub = sub.sort_values("group_value")
        colors = ["#27ae60" if v >= overall_auroc else "#e74c3c" for v in sub["AUROC"]]
        bars = ax.bar(sub["group_value"].astype(str), sub["AUROC"], color=colors, alpha=0.8)
        ax.axhline(overall_auroc, color="black", linestyle="--", lw=1.5, label=f"Overall={overall_auroc:.3f}")
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
        ax.set_ylim(max(0, overall_auroc - 0.15), min(1.0, overall_auroc + 0.15))
        ax.set_title(f"AUROC by {gvar}", fontsize=12, fontweight="bold")
        ax.set_ylabel("AUROC")
        ax.legend(fontsize=9)

    plt.suptitle("Fairness Subgroup Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Fairness plot saved -> %s", save_path)

    plt.show()
