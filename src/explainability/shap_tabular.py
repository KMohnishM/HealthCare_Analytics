"""
src/explainability/shap_tabular.py
------------------------------------
SHAP-based explainability for the XGBoost tabular branch.

Generates:
  - Global beeswarm plot (feature importance + direction)
  - Global bar plot (mean |SHAP|)
  - Local waterfall plot for individual patients
  - SHAP value CSV for downstream analysis

Usage:
    from src.explainability.shap_tabular import explain_tabular
    shap_out = explain_tabular(ensemble, X_test, y_test, cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger(__name__)


def explain_tabular(
    ensemble,           # TabularEnsemble instance
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: DictConfig,
    n_samples_global: int = 500,
    patient_hadm_ids: Optional[list] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute SHAP values and generate explanation plots.

    Parameters
    ----------
    ensemble : TabularEnsemble
        Fitted XGBoost ensemble (uses the first model for SHAP).
    X_test : pd.DataFrame
        Test feature matrix (imputed, no NaN).
    y_test : pd.Series
        Binary test labels.
    cfg : DictConfig
        Project configuration.
    n_samples_global : int
        Number of test samples to use for global SHAP plots
        (random subsample for speed).
    patient_hadm_ids : list, optional
        Specific hadm_ids to generate waterfall plots for.
    save_dir : str, optional
        Directory to save plots and CSV.

    Returns
    -------
    dict with keys:
        - 'shap_values'  : (N, F) array of SHAP values for each sample
        - 'base_value'   : scalar expected output
        - 'feature_names': list of feature names
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install SHAP: pip install shap")

    import matplotlib.pyplot as plt

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    # Use the first bootstrap model for SHAP (most representative)
    base_model = ensemble.models[0]
    explainer = shap.TreeExplainer(base_model)

    log.info("Computing SHAP values for %d test samples ...", len(X_test))
    shap_explanation = explainer(X_test)  # shap.Explanation object
    shap_values      = shap_explanation.values       # (N, F)
    base_value       = explainer.expected_value

    # ── Global: beeswarm ─────────────────────────────────────────────────────
    idx_sample = np.random.choice(len(X_test), min(n_samples_global, len(X_test)), replace=False)
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(
        shap_explanation[idx_sample],
        max_display=20,
        show=False,
    )
    plt.title("SHAP Beeswarm — Tabular Branch (Top 20 Features)", fontsize=13)
    plt.tight_layout()
    if save_dir:
        fig1.savefig(save_path / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        log.info("Saved beeswarm -> %s", save_path / "shap_beeswarm.png")
    plt.show()
    plt.close()

    # ── Global: bar plot (mean |SHAP|) ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    shap.plots.bar(shap_explanation[idx_sample], max_display=20, show=False)
    plt.title("Mean |SHAP| — Top 20 Features", fontsize=13)
    plt.tight_layout()
    if save_dir:
        fig2.savefig(save_path / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    # ── Local: waterfall for selected patients ───────────────────────────────
    if patient_hadm_ids:
        for hadm_id in patient_hadm_ids:
            if hadm_id not in X_test.index:
                log.warning("hadm_id %s not in X_test index — skipping.", hadm_id)
                continue
            i = X_test.index.get_loc(hadm_id)
            true_label = int(y_test.loc[hadm_id]) if hadm_id in y_test.index else "?"

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation[i], max_display=15, show=False)
            plt.title(
                f"SHAP Waterfall — Patient hadm_id={hadm_id} | label={true_label}",
                fontsize=12,
            )
            plt.tight_layout()
            if save_dir:
                fname = save_path / f"shap_waterfall_{hadm_id}.png"
                fig3.savefig(fname, dpi=150, bbox_inches="tight")
                log.info("Saved waterfall -> %s", fname)
            plt.show()
            plt.close()

    # ── Save SHAP values as CSV ──────────────────────────────────────────────
    if save_dir:
        shap_df = pd.DataFrame(
            shap_values,
            index=X_test.index,
            columns=X_test.columns,
        )
        shap_df.to_csv(save_path / "shap_values.csv")
        log.info("SHAP values CSV saved -> %s", save_path / "shap_values.csv")

    # ── Summary table ────────────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":        X_test.columns,
        "mean_abs_shap":  mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    log.info("Top 10 features by |SHAP|:\n%s", importance_df.head(10).to_string(index=False))

    return {
        "shap_values":   shap_values,
        "base_value":    base_value,
        "feature_names": X_test.columns.tolist(),
        "importance_df": importance_df,
    }
