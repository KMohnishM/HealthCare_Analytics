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

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

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

    # ── Log Missingness (XGBoost handles NaN natively) ─────────────────────────
    miss_report = missingness_report(X_train)
    log.info("Top 5 missing features:\n%s", miss_report.head(5).to_string(index=False))

    # ── Out-Of-Fold (OOF) Generation for Stacking (Fusion) ───────────────────
    log.info("Generating Out-Of-Fold predictions via 5-fold CV to prevent Stacking Leakage …")
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.cohort.random_seed)
    
    oof_scores = np.zeros(len(X_train))
    oof_confs = np.zeros(len(X_train))
    oof_stds = np.zeros(len(X_train))
    # Tabular "embeddings" are the raw features. OOF embeddings are just the features.
    oof_embeds = X_train.values.copy() 

    # We copy the config overrides to avoid modifying the global config
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        log.info(f"--- Fold {fold + 1}/5 ---")
        X_tr_fold, y_tr_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va_fold, y_va_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        fold_model = TabularEnsemble(cfg)
        fold_model.fit(X_tr_fold, y_tr_fold, X_va_fold, y_va_fold)
        
        fold_pred = fold_model.predict(X_va_fold)
        oof_scores[val_idx] = fold_pred["score"]
        oof_confs[val_idx] = fold_pred["confidence"]
        oof_stds[val_idx] = fold_pred["std"]

    log.info("OOF prediction generation complete.")

    # ── Train final ensemble on full training set ────────────────────────────
    log.info("Training final tabular ensemble on full training set …")
    model = TabularEnsemble(cfg)
    model.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("Evaluating final model on test set …")
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

    # Generate predictions on val/test using the final trained model
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    splits_data = {
        "train": {
            "hadm_id": X_train.index,
            "score": oof_scores,
            "confidence": oof_confs,
            "std": oof_stds,
            "label": y_train.values,
            "embed": oof_embeds
        },
        "val": {
            "hadm_id": X_val.index,
            "score": val_pred["score"],
            "confidence": val_pred["confidence"],
            "std": val_pred["std"],
            "label": y_val.values,
            "embed": X_val.values
        },
        "test": {
            "hadm_id": X_test.index,
            "score": test_pred["score"],
            "confidence": test_pred["confidence"],
            "std": test_pred["std"],
            "label": y_test.values,
            "embed": X_test.values
        }
    }

    for split_name, data in splits_data.items():
        out = pd.DataFrame({
            "hadm_id": data["hadm_id"],
            "score": data["score"],
            "confidence": data["confidence"],
            "std": data["std"],
            "label": data["label"],
        })
        embed_path = results_dir / f"tabular_embed_{split_name}.npy"
        # Neural networks cannot process NaNs natively, so we impute with 0.0 for fusion input
        embed_imputed = np.nan_to_num(data["embed"], nan=0.0)
        np.save(embed_path, embed_imputed.astype("float32"))
        out.to_csv(results_dir / f"tabular_preds_{split_name}.csv", index=False)
        log.info("Saved %s predictions (OOF for train) → %s", split_name, results_dir)

    # Feature importance
    importance_df = model.get_feature_importance()
    log.info("Top 10 features:\n%s", importance_df.head(10).to_string(index=False))
    importance_df.to_csv(
        Path(cfg.paths.results_dir) / "tabular_feature_importance.csv", index=False
    )

    log.info("Tabular branch training complete ✓")


if __name__ == "__main__":
    main()
