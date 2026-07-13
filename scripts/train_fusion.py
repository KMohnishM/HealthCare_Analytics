"""
scripts/train_fusion.py
------------------------
Training script for the learned gating fusion layer.

Prerequisites: run train_tabular.py, train_ecg.py, train_cxr.py first
to generate branch embeddings in data/processed/.

Run from project root:
    python scripts/train_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.fusion.learned_gate import GatedFusionModel, make_fusion_dataset, train_fusion
from src.fusion.fixed_weight import fixed_fusion_predict
from src.evaluation.metrics import evaluate_all
from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger, init_mlflow
from src.utils.seed import set_seed

log = get_logger(__name__)


def load_branch_outputs(proc_dir: Path, split: str) -> dict:
    """
    Load pre-computed branch embeddings, scores, and confidences.

    Parameters
    ----------
    proc_dir : Path
        Path to data/processed/ directory.
    split : str
        One of 'train', 'val', 'test'.

    Returns
    -------
    dict with keys 'tabular', 'ecg', 'cxr', each containing
    'embed', 'score', 'confidence', 'available', 'label', 'hadm_id'.
    """
    out = {}
    for mod in ["tabular", "ecg", "cxr"]:
        preds = pd.read_csv(proc_dir / f"{mod}_preds_{split}.csv")
        embed = np.load(proc_dir / f"{mod}_embed_{split}.npy")
        out[mod] = {
            "embed":      embed,
            "score":      preds["score"].values.astype("float32"),
            "confidence": preds["confidence"].values.astype("float32"),
            "available":  preds.get("available", pd.Series(np.ones(len(preds)))).values.astype("float32"),
            "label":      preds["label"].values.astype("float32"),
            "hadm_id":    preds["hadm_id"].values,
        }
    return out


def main() -> None:
    cfg    = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.fusion.random_seed)

    run_id = init_mlflow(cfg, run_name="train_fusion")

    proc_dir = Path(cfg.paths.processed_dir)

    # ── Load branch outputs ───────────────────────────────────────────────────
    log.info("Loading branch outputs …")
    train_out = load_branch_outputs(proc_dir, "train")
    val_out   = load_branch_outputs(proc_dir, "val")
    test_out  = load_branch_outputs(proc_dir, "test")

    y_train = train_out["tabular"]["label"]
    y_val   = val_out["tabular"]["label"]
    y_test  = test_out["tabular"]["label"]

    # ── Build availability masks (natural missingness from the data) ──────────
    def make_avail(out_dict: dict) -> np.ndarray:
        """(N, 3) availability matrix [tab, ecg, cxr]."""
        return np.stack([
            out_dict["tabular"]["available"],
            out_dict["ecg"]["available"],
            out_dict["cxr"]["available"],
        ], axis=1).astype("float32")

    # Tabular is always available (1.0) — set explicitly
    for out_d in [train_out, val_out, test_out]:
        out_d["tabular"]["available"] = np.ones(len(out_d["tabular"]["label"]), dtype="float32")

    # ── Determine tabular embedding dim ──────────────────────────────────────
    tab_dim = train_out["tabular"]["embed"].shape[1]
    log.info("Tab embed dim = %d", tab_dim)

    # ── TensorDatasets ───────────────────────────────────────────────────────
    def make_ds(out: dict) -> torch.utils.data.TensorDataset:
        avail = make_avail(out)
        return make_fusion_dataset(
            tab_embeds   = out["tabular"]["embed"],
            ecg_embeds   = out["ecg"]["embed"],
            cxr_embeds   = out["cxr"]["embed"],
            tab_confs    = out["tabular"]["confidence"],
            ecg_confs    = out["ecg"]["confidence"],
            cxr_confs    = out["cxr"]["confidence"],
            availability = avail,
            labels       = out["tabular"]["label"],
        )

    train_ds = make_ds(train_out)
    val_ds   = make_ds(val_out)
    test_ds  = make_ds(test_out)

    train_loader = DataLoader(train_ds, batch_size=cfg.fusion.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.fusion.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.fusion.batch_size)

    # ── Train learned gate ───────────────────────────────────────────────────
    log.info("Training learned gating fusion model …")
    gate_model = GatedFusionModel(cfg, tab_dim=tab_dim)
    save_path  = str(Path(cfg.paths.models_dir) / "fusion_gate.pt")

    history = train_fusion(gate_model, train_loader, val_loader, cfg, save_path=save_path)

    # ── Evaluate: fixed-weight vs learned gate on test ───────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate_model.to(device).eval()

    test_avail = make_avail(test_out)
    test_avail_dict = {
        "tabular": test_avail[:, 0],
        "ecg":     test_avail[:, 1],
        "cxr":     test_avail[:, 2],
    }

    # Fixed-weight predictions
    fw_result = fixed_fusion_predict(
        branch_results=test_out,
        avail_flags=test_avail_dict,
    )

    # Learned gate predictions
    all_gate_probs = []
    for batch in test_loader:
        (tab_e, ecg_e, cxr_e, tab_c, ecg_c, cxr_c, avail, _) = [b.to(device) for b in batch]
        with torch.no_grad():
            logit, _ = gate_model(tab_e, ecg_e, cxr_e, tab_c, ecg_c, cxr_c, avail)
        all_gate_probs.append(torch.sigmoid(logit.squeeze(-1)).cpu().numpy())
    gate_probs = np.concatenate(all_gate_probs)

    fw_metrics   = evaluate_all(y_test, fw_result["score"],   prefix="fixed_weight")
    gate_metrics = evaluate_all(y_test, gate_probs,           prefix="learned_gate")

    log.info("=" * 60)
    log.info("TEST SET RESULTS — Fusion Layer")
    for k, v in {**fw_metrics, **gate_metrics}.items():
        log.info("  %s: %.4f", k, v)
    log.info("=" * 60)

    # ── Save test predictions for evaluation pipeline ─────────────────────────
    out_df = pd.DataFrame({
        "hadm_id":           test_out["tabular"]["hadm_id"],
        "label":             y_test,
        "tabular_score":     test_out["tabular"]["score"],
        "ecg_score":         test_out["ecg"]["score"],
        "cxr_score":         test_out["cxr"]["score"],
        "fixed_weight_score": fw_result["score"],
        "learned_gate_score": gate_probs,
        "ecg_available":     test_avail[:, 1],
        "cxr_available":     test_avail[:, 2],
    })
    out_path = Path(cfg.paths.results_dir) / "fusion_test_preds.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info("Fusion predictions saved → %s", out_path)

    log.info("Fusion training complete ✓")


if __name__ == "__main__":
    main()
