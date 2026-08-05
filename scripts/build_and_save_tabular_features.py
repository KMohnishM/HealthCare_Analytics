"""
scripts/build_and_save_tabular_features.py
---------------------------------------------------------------------------------
Pre-computes and saves tabular feature matrices from raw MIMIC-IV clinical CSVs.
Run this on the machine containing the raw hosp/ and icu/ tables to package features.
Usage:
    python scripts/build_and_save_tabular_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.config import load_config
from src.tabular.features import build_feature_matrix

def main():
    cfg = load_config()
    cohort_dir = Path(cfg.paths.cohort_dir)
    mimic_hosp = Path(cfg.paths.mimic_iv_dir) / "hosp"
    mimic_icu  = Path(cfg.paths.mimic_iv_dir) / "icu"
    
    print("Loading splits...")
    train_df = pd.read_parquet(cohort_dir / "train.parquet")
    val_df   = pd.read_parquet(cohort_dir / "val.parquet")
    test_df  = pd.read_parquet(cohort_dir / "test.parquet")
    
    print("Loading clinical tables (labevents & chartevents)...")
    lab_path = mimic_hosp / "labevents.csv.gz"
    labevents = pd.read_csv(lab_path if lab_path.exists() else mimic_hosp / "labevents.csv", low_memory=False)
    
    ce_path = mimic_icu / "chartevents.csv.gz"
    chartevents = pd.read_csv(
        ce_path if ce_path.exists() else mimic_icu / "chartevents.csv", 
        usecols=["hadm_id", "itemid", "charttime", "valuenum"], 
        low_memory=False
    )
    
    print("Building feature matrices...")
    X_train, y_train, _ = build_feature_matrix(train_df, cfg, labevents, chartevents)
    X_val,   y_val,   _ = build_feature_matrix(val_df,   cfg, labevents, chartevents)
    X_test,  y_test,  _ = build_feature_matrix(test_df,  cfg, labevents, chartevents)
    
    print("Saving pre-computed features to data/ ...")
    X_train.to_parquet(cohort_dir / "X_train.parquet", index=False)
    X_val.to_parquet(cohort_dir / "X_val.parquet", index=False)
    X_test.to_parquet(cohort_dir / "X_test.parquet", index=False)
    
    pd.DataFrame(y_train).to_parquet(cohort_dir / "y_train.parquet", index=False)
    pd.DataFrame(y_val).to_parquet(cohort_dir / "y_val.parquet", index=False)
    pd.DataFrame(y_test).to_parquet(cohort_dir / "y_test.parquet", index=False)
    print("Tabular features extracted and saved successfully!")

if __name__ == "__main__":
    main()
