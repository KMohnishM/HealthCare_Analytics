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

from src.tabular.features import build_feature_matrix, extract_lab_features, extract_demographic_features, augment_engineered_features
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
    log.info("Loading cohort splits from %s ...", cohort_dir)
    train_df = pd.read_parquet(cohort_dir / "train.parquet")
    val_df   = pd.read_parquet(cohort_dir / "val.parquet")
    test_df  = pd.read_parquet(cohort_dir / "test.parquet")

    log.info(
        "Splits: train=%d | val=%d | test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Check for pre-computed features (bypasses raw CSV load on cloud/Kaggle) ──
    X_train_path = cohort_dir / "X_train.parquet"
    if X_train_path.exists():
        log.info("Found pre-computed feature parquets. Loading directly...")
        X_train = pd.read_parquet(cohort_dir / "X_train.parquet")
        X_val   = pd.read_parquet(cohort_dir / "X_val.parquet")
        X_test  = pd.read_parquet(cohort_dir / "X_test.parquet")
        y_train = pd.read_parquet(cohort_dir / "y_train.parquet").iloc[:, 0].values
        y_val   = pd.read_parquet(cohort_dir / "y_val.parquet").iloc[:, 0].values
        y_test  = pd.read_parquet(cohort_dir / "y_test.parquet").iloc[:, 0].values

        # ── Fix 1: Re-derive race from cohort splits (parquet stores race as all-NaN) ──
        for split_name, X_split, cohort_split in [
            ("train", X_train, train_df),
            ("val",   X_val,   val_df),
            ("test",  X_test,  test_df),
        ]:
            demo = extract_demographic_features(cohort_split)
            race_cols = [c for c in demo.columns if c.startswith("race_")]
            if race_cols:
                # X_split may have RangeIndex; use hadm_id from cohort to align
                hadm_ids = cohort_split["hadm_id"].values
                X_split.index = hadm_ids
                X_split.index.name = "hadm_id"
                X_split.drop(columns=[c for c in race_cols if c in X_split.columns], inplace=True, errors="ignore")
                X_split[race_cols] = demo[race_cols].reindex(X_split.index)
            log.info("Race re-derived for %s split (%d race cols, %d non-null)",
                     split_name, len(race_cols),
                     int(X_split[race_cols[0]].notna().sum()) if race_cols else 0)

        # ── Fix 2: Compute lab delta features if labevents is available ─────────────
        # Search common paths: local MIMIC-IV, Kaggle input, Colab Drive
        lab_search_paths = [
            Path(cfg.paths.mimic_iv_dir) / "hosp" / "labevents.csv.gz",
            Path(cfg.paths.mimic_iv_dir) / "hosp" / "labevents.csv",
            Path("/kaggle/input/mimic-iv/hosp/labevents.csv.gz"),
            Path("/kaggle/input/mimic-iv-clinical-database/hosp/labevents.csv.gz"),
            Path("/content/drive/MyDrive/mimic-iv/hosp/labevents.csv.gz"),
        ]
        lab_path_found = next((p for p in lab_search_paths if p.exists()), None)

        DELTA_SENTINEL = "lab_delta_hemoglobin"  # check if deltas already present
        if lab_path_found and DELTA_SENTINEL not in X_train.columns:
            log.info("Found labevents at %s — computing lab trajectory delta features ...", lab_path_found)
            try:
                lab_itemids = {k: list(v) for k, v in cfg.tabular.lab_itemids.items()}
                all_hadm = set(train_df["hadm_id"]) | set(val_df["hadm_id"]) | set(test_df["hadm_id"])
                chunks = []
                for chunk in pd.read_csv(lab_path_found, chunksize=1_000_000, low_memory=False):
                    mask = chunk["hadm_id"].isin(all_hadm)
                    if mask.any():
                        chunks.append(chunk[mask])
                labevents_filtered = pd.concat(chunks, ignore_index=True)

                for split_name, X_split, cohort_split in [
                    ("train", X_train, train_df),
                    ("val",   X_val,   val_df),
                    ("test",  X_test,  test_df),
                ]:
                    lab_feats = extract_lab_features(cohort_split, labevents_filtered, lab_itemids)
                    delta_cols = [c for c in lab_feats.columns if "delta" in c or "ratio" in c]
                    if delta_cols:
                        X_split[delta_cols] = lab_feats[delta_cols].reindex(X_split.index)
                log.info("Lab trajectory delta features added: %d new columns", len(delta_cols))
            except Exception as exc:
                log.warning("Lab delta extraction failed (%s) — proceeding without delta features.", exc)
        elif DELTA_SENTINEL in X_train.columns:
            log.info("Lab delta features already present in cached parquets — skipping extraction.")
        else:
            log.warning(
                "labevents not found in any standard path — lab delta features unavailable. "
                "Add MIMIC-IV hosp data to Kaggle input to enable delta features."
            )

        X_train = augment_engineered_features(X_train)
        X_val   = augment_engineered_features(X_val)
        X_test  = augment_engineered_features(X_test)

        # Drop any feature columns that are entirely NaN (dead features)
        dead_cols = [c for c in X_train.columns if X_train[c].isna().all()]
        if dead_cols:
            log.warning("Dropping %d all-NaN feature columns: %s", len(dead_cols), dead_cols)
            X_train.drop(columns=dead_cols, inplace=True)
            X_val.drop(columns=dead_cols, inplace=True, errors="ignore")
            X_test.drop(columns=dead_cols, inplace=True, errors="ignore")

        # Save patched parquets back to disk so git tracks the correct, non-NaN features
        X_train.to_parquet(cohort_dir / "X_train.parquet")
        X_val.to_parquet(cohort_dir / "X_val.parquet")
        X_test.to_parquet(cohort_dir / "X_test.parquet")
        pd.DataFrame(y_train, columns=["readmitted_30d"]).to_parquet(cohort_dir / "y_train.parquet")
        pd.DataFrame(y_val, columns=["readmitted_30d"]).to_parquet(cohort_dir / "y_val.parquet")
        pd.DataFrame(y_test, columns=["readmitted_30d"]).to_parquet(cohort_dir / "y_test.parquet")
        log.info("Saved patched feature parquets on disk.")

        features = X_train.columns.tolist()
        log.info("Final feature matrix: %d features total", len(features))
    else:
        # ── Load MIMIC tables (loaded once, shared across splits) ─────────────────
        mimic_hosp = Path(cfg.paths.mimic_iv_dir) / "hosp"
        mimic_icu  = Path(cfg.paths.mimic_iv_dir) / "icu"

        log.info("Loading labevents ...")
        lab_path = mimic_hosp / "labevents.csv.gz"
        labevents = pd.read_csv(lab_path if lab_path.exists() else mimic_hosp / "labevents.csv",
                                low_memory=False)

        log.info("Loading chartevents (selected columns only) ...")
        ce_path = mimic_icu / "chartevents.csv.gz"
        chartevents = pd.read_csv(
            ce_path if ce_path.exists() else mimic_icu / "chartevents.csv",
            usecols=["hadm_id", "itemid", "charttime", "valuenum"],
            low_memory=False,
        )

        # ── Build feature matrices ────────────────────────────────────────────────
        log.info("Building feature matrices from raw CSVs...")
        X_train, y_train, features = build_feature_matrix(train_df, cfg, labevents, chartevents)
        X_val,   y_val,   _        = build_feature_matrix(val_df,   cfg, labevents, chartevents)
        X_test,  y_test,  _        = build_feature_matrix(test_df,  cfg, labevents, chartevents)

    # ── Log Missingness (XGBoost handles NaN natively) ─────────────────────────
    miss_report = missingness_report(X_train)
    log.info("Top 5 missing features:\n%s", miss_report.head(5).to_string(index=False))

    # ── Out-Of-Fold (OOF) Generation for Stacking (Fusion) ───────────────────
    log.info("Generating Out-Of-Fold predictions via 5-fold CV to prevent Stacking Leakage ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.cohort.random_seed)
    
    oof_scores = np.zeros(len(X_train))
    oof_confs = np.zeros(len(X_train))
    oof_stds = np.zeros(len(X_train))
    # Tabular "embeddings" are the raw features. OOF embeddings are just the features.
    oof_embeds = X_train.values.copy() 

    # We copy the config overrides to avoid modifying the global config
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        log.info(f"--- Fold {fold + 1}/5 ---")
        X_tr_fold, y_tr_fold = X_train.iloc[train_idx], y_train[train_idx]
        X_va_fold, y_va_fold = X_train.iloc[val_idx], y_train[val_idx]
        
        fold_model = TabularEnsemble(cfg)
        fold_model.fit(X_tr_fold, y_tr_fold, X_va_fold, y_va_fold)
        
        fold_pred = fold_model.predict(X_va_fold)
        oof_scores[val_idx] = fold_pred["score"]
        oof_confs[val_idx] = fold_pred["confidence"]
        oof_stds[val_idx] = fold_pred["std"]

    log.info("OOF prediction generation complete.")

    # ── Train final ensemble on full training set ────────────────────────────
    log.info("Training final tabular ensemble on full training set ...")
    model = TabularEnsemble(cfg)
    model.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("Evaluating final model on test set ...")
    test_result = model.predict(X_test)
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    metrics = evaluate_all(y_test_arr, test_result["score"])

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
            "label": y_train.values if hasattr(y_train, "values") else np.asarray(y_train),
            "embed": oof_embeds
        },
        "val": {
            "hadm_id": X_val.index,
            "score": val_pred["score"],
            "confidence": val_pred["confidence"],
            "std": val_pred["std"],
            "label": y_val.values if hasattr(y_val, "values") else np.asarray(y_val),
            "embed": X_val.values
        },
        "test": {
            "hadm_id": X_test.index,
            "score": test_pred["score"],
            "confidence": test_pred["confidence"],
            "std": test_pred["std"],
            "label": y_test.values if hasattr(y_test, "values") else np.asarray(y_test),
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
        log.info("Saved %s predictions (OOF for train) -> %s", split_name, results_dir)

    # Feature importance
    importance_df = model.get_feature_importance()
    log.info("Top 10 features:\n%s", importance_df.head(10).to_string(index=False))
    importance_df.to_csv(
        Path(cfg.paths.results_dir) / "tabular_feature_importance.csv", index=False
    )

    log.info("Tabular branch training complete [OK]")


if __name__ == "__main__":
    main()
