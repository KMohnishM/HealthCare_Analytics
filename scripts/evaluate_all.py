"""
scripts/evaluate_all.py
------------------------
Full evaluation pipeline: missingness sweep + baselines + DCA + fairness.

Prerequisites: all branch models and fusion layer must be trained.

Run from project root:
    python scripts/evaluate_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import evaluate_all
from src.evaluation.missingness_sweep import run_missingness_sweep, plot_missingness_sweep
from src.evaluation.decision_curve import run_dca, plot_dca
from src.evaluation.fairness import fairness_report, plot_fairness_report
from src.fusion.learned_gate import GatedFusionModel
from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    proc_dir    = Path(cfg.paths.processed_dir)
    results_dir = Path(cfg.paths.results_dir)
    figures_dir = Path(cfg.paths.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test split and predictions ──────────────────────────────────────
    cohort_dir = Path(cfg.paths.cohort_dir)
    test_df    = pd.read_parquet(cohort_dir / "test.parquet")

    fusion_preds = pd.read_csv(results_dir / "fusion_test_preds.csv")
    y_test       = fusion_preds["label"].values

    # ── Load branch embeddings for missingness sweep ──────────────────────────
    log.info("Loading branch outputs for missingness sweep ...")
    branch_outputs = {}
    for mod in ["tabular", "ecg", "cxr"]:
        preds = pd.read_csv(proc_dir / f"{mod}_preds_test.csv")
        embed = np.load(proc_dir / f"{mod}_embed_test.npy")
        avail = preds.get("available", pd.Series(np.ones(len(preds)))).values.astype("float32")
        if mod == "tabular":
            avail = np.ones(len(preds), dtype="float32")  # always available
        branch_outputs[mod] = {
            "score":      preds["score"].values.astype("float32"),
            "confidence": preds["confidence"].values.astype("float32"),
            "embed":      embed,
            "available":  avail,
        }

    # ── Load trained gate model ───────────────────────────────────────────────
    gate_model = None
    gate_path  = Path(cfg.paths.models_dir) / "fusion_gate.pt"
    if gate_path.exists():
        tab_dim = branch_outputs["tabular"]["embed"].shape[1]
        gate_model = GatedFusionModel(cfg, tab_dim=tab_dim)
        ckpt = torch.load(gate_path, map_location="cpu")
        gate_model.load_state_dict(ckpt["model_state"])
        gate_model.eval()
        log.info("Loaded trained gate model from %s", gate_path)
    else:
        log.warning("Fusion gate model not found at %s — running fixed-weight only.", gate_path)

    # ── 1. Missingness Sweep (CORE EXPERIMENT) ────────────────────────────────
    log.info("\n%s\nMISSINGNESS SWEEP\n%s", "=" * 60, "=" * 60)
    sweep_df = run_missingness_sweep(branch_outputs, y_test, gate_model=gate_model, cfg=cfg)
    sweep_df.to_csv(results_dir / "missingness_sweep.csv", index=False)
    log.info("Missingness sweep results:\n%s",
             sweep_df[["subset", "fusion", "AUROC", "AUPRC", "Brier", "ECE"]].to_string(index=False))

    plot_missingness_sweep(
        sweep_df,
        save_path=str(figures_dir / "missingness_sweep_heatmap.png"),
    )

    # ── 2. Clinical Baselines Comparison ─────────────────────────────────────
    log.info("\n%s\nBASELINE COMPARISON\n%s", "=" * 60, "=" * 60)

    baseline_preds = {}
    for score_col in ["lace_prob", "hospital_prob"]:
        if score_col in fusion_preds.columns:
            baseline_preds[score_col.replace("_prob", "")] = fusion_preds[score_col].values

    all_predictions = {
        "tabular_only":  branch_outputs["tabular"]["score"],
        "fixed_weight":  fusion_preds["fixed_weight_score"].values,
        **baseline_preds,
    }
    if "learned_gate_score" in fusion_preds.columns:
        all_predictions["learned_gate"] = fusion_preds["learned_gate_score"].values

    comparison_records = []
    for name, probs in all_predictions.items():
        m = evaluate_all(y_test, probs, prefix=name)
        comparison_records.append({"model": name, **{k.replace(name + "_", ""): v for k, v in m.items()}})
        log.info("  %-20s AUROC=%.4f  AUPRC=%.4f  Brier=%.4f  ECE=%.4f",
                 name,
                 m.get(f"{name}_AUROC", float("nan")),
                 m.get(f"{name}_AUPRC", float("nan")),
                 m.get(f"{name}_Brier", float("nan")),
                 m.get(f"{name}_ECE",   float("nan")))

    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)

    # ── 3. Decision Curve Analysis ────────────────────────────────────────────
    log.info("\n%s\nDECISION CURVE ANALYSIS\n%s", "=" * 60, "=" * 60)
    dca_preds = {k: v for k, v in all_predictions.items()}
    dca_df = run_dca(
        y_test,
        dca_preds,
        thresh_min  = cfg.evaluation.dca_thresh_min,
        thresh_max  = cfg.evaluation.dca_thresh_max,
        thresh_step = cfg.evaluation.dca_thresh_step,
    )
    dca_df.to_csv(results_dir / "dca_results.csv", index=False)
    plot_dca(
        dca_df,
        title="Decision Curve Analysis — HF 30-Day Readmission",
        save_path=str(figures_dir / "decision_curve.png"),
    )

    # ── 4. Fairness Subgroup Analysis ─────────────────────────────────────────
    log.info("\n%s\nFAIRNESS ANALYSIS\n%s", "=" * 60, "=" * 60)

    gate_probs = all_predictions.get("learned_gate", all_predictions["fixed_weight"])

    # Align cohort with predictions
    test_cohort = test_df.set_index("hadm_id").reindex(fusion_preds["hadm_id"]).reset_index()
    test_cohort["readmitted_30d"] = y_test

    fairness_df = fairness_report(
        cohort_test = test_cohort.reset_index(drop=True),
        y_prob      = gate_probs,
        groups      = list(cfg.evaluation.fairness_groups),
    )
    fairness_df.to_csv(results_dir / "fairness_report.csv", index=False)
    log.info("Fairness report:\n%s", fairness_df.to_string(index=False))

    plot_fairness_report(
        fairness_df,
        save_path=str(figures_dir / "fairness_subgroups.png"),
    )

    log.info("\nAll evaluations complete. Results saved to %s", results_dir)
    log.info("Figures saved to %s", figures_dir)


if __name__ == "__main__":
    main()
