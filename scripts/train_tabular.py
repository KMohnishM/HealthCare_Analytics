"""
scripts/train_tabular.py
-------------------------
End-to-end training script for the XGBoost tabular branch.

Run from project root:
    python scripts/train_tabular.py
    # With config override:
    python scripts/train_tabular.py paths.base_dir=/content/drive/MyDrive/HA

On Google Colab, mount Drive first:
    from google.colab import drive
    drive.mount('/content/drive')
    !python scripts/train_tabular.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.tabular.features import build_feature_matrix
from src.tabular.impute import fit_imputer, apply_imputer, save_imputer, missingness_report
from src.tabular.model import TabularEnsemble
from src.evaluation.metrics import evaluate_all
from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger, init_mlflow
from src.utils.seed import set_seed

log = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg.cohort.random_seed)

    run_id = init_mlflow(cfg, run_name="train_tabular")
    log.info("MLflow run_id: %s", run_id)

    # ── Load cohort splits ────────────────────────────────────────────────────
    cohort_dir = Path(cfg.paths.cohort_dir)
    log.info("Loading cohort splits from %s …", cohort_dir)
    train_df = pd.read_parquet(cohort_dir / "train.parquet")
    val_df   = pd.read_parquet(cohort_dir / "val.parquet")
    test_df  = pd.read_parquet(cohort_dir / "test.parquet")

    log.info(
        "Splits: train=%d | val=%d | test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Load MIMIC tables (loaded once, shared across splits) ─────────────────
    mimic_hosp = Path(cfg.paths.mimic_iv_dir) / "hosp"
    mimic_icu  = Path(cfg.paths.mimic_iv_dir) / "icu"

    log.info("Loading labevents …")
    lab_path = mimic_hosp / "labevents.csv.gz"
    labevents = pd.read_csv(lab_path if lab_path.exists() else mimic_hosp / "labevents.csv",
                            low_memory=False)

    log.info("Loading chartevents (selected columns only) …")
    ce_path = mimic_icu / "chartevents.csv.gz"
    chartevents = pd.read_csv(
        ce_path if ce_path.exists() else mimic_icu / "chartevents.csv",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        low_memory=False,
    )

    # ── Build feature matrices ────────────────────────────────────────────────
    log.info("Building feature matrices …")
    X_train, y_train, features = build_feature_matrix(train_df, cfg, labevents, chartevents)
    X_val,   y_val,   _        = build_feature_matrix(val_df,   cfg, labevents, chartevents)
    X_test,  y_test,  _        = build_feature_matrix(test_df,  cfg, labevents, chartevents)

    # ── Imputation ────────────────────────────────────────────────────────────
    log.info("Fitting imputer on training data …")
    miss_report = missingness_report(X_train)
    log.info("Top 5 missing features:\n%s", miss_report.head(5).to_string(index=False))

    imputer  = fit_imputer(X_train)
    X_train  = apply_imputer(imputer, X_train)
    X_val    = apply_imputer(imputer, X_val)
    X_test   = apply_imputer(imputer, X_test)

    # Save imputer
    imp_path = Path(cfg.paths.models_dir) / "tabular_imputer.pkl"
    save_imputer(imputer, imp_path)

    # ── Train ensemble ────────────────────────────────────────────────────────
    model = TabularEnsemble(cfg)
    model.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("Evaluating on test set …")
    test_result = model.predict(X_test)
    metrics = evaluate_all(y_test.values, test_result["score"])

    log.info("=" * 60)
    log.info("TEST SET RESULTS — Tabular Branch (XGBoost Ensemble)")
    for k, v in metrics.items():
        log.info("  %s: %.4f", k, v)
    log.info("=" * 60)

    # Log to MLflow
    try:
        import mlflow
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_bootstrap", cfg.tabular.xgb.n_bootstrap)
        mlflow.log_param("n_features", len(features))
    except Exception:
        pass

    # ── Save model and results ────────────────────────────────────────────────
    model_path = Path(cfg.paths.models_dir) / "tabular_ensemble.pkl"
    model.save(model_path)

    # Save test predictions and embeddings (for fusion layer)
    results_dir = Path(cfg.paths.processed_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for split_name, X_sp, y_sp, cohort_sp in [
        ("train", X_train, y_train, train_df),
        ("val",   X_val,   y_val,   val_df),
        ("test",  X_test,  y_test,  test_df),
    ]:
        result = model.predict(X_sp)
        out = pd.DataFrame({
            "hadm_id":       X_sp.index,
            "score":         result["score"],
            "confidence":    result["confidence"],
            "std":           result["std"],
            "label":         y_sp.values,
        })
        # Also save feature matrix as embedding proxy for the fusion layer
        embed_path = results_dir / f"tabular_embed_{split_name}.npy"
        # Use the raw feature matrix as the "embedding" for tabular branch
        # The fusion layer will project this via its proj_tab layer
        import numpy as np
        np.save(embed_path, X_sp.values.astype("float32"))

        out.to_csv(results_dir / f"tabular_preds_{split_name}.csv", index=False)
        log.info("Saved %s predictions → %s", split_name, results_dir)

    # Feature importance
    importance_df = model.get_feature_importance()
    log.info("Top 10 features:\n%s", importance_df.head(10).to_string(index=False))
    importance_df.to_csv(
        Path(cfg.paths.results_dir) / "tabular_feature_importance.csv", index=False
    )

    log.info("Tabular branch training complete ✓")


if __name__ == "__main__":
    main()
