"""
src/tabular/features.py
------------------------
Extract tabular feature matrix from MIMIC-IV for the HF cohort.

Features extracted:
  - Demographics : age, gender (binary), race dummies
  - Vitals       : last-48h mean/min/max for HR, SBP, DBP, SpO2, RR, Temp
  - Labs         : last-72h closest-to-discharge value for key HF labs
  - Administrative: LOS, via_ed, prior_admits_12m, ed_visits_6m
  - HF-specific  : NT-proBNP/BNP, creatinine, sodium, hemoglobin

Usage:
    from src.tabular.features import build_feature_matrix
    X_train, feature_names = build_feature_matrix(train_df, cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Lab feature extraction ────────────────────────────────────────────────────

def extract_lab_features(
    cohort: pd.DataFrame,
    labevents: pd.DataFrame,
    lab_itemids: dict[str, list[int]],
    window_hours: int = 72,
) -> pd.DataFrame:
    """
    For each admission, extract the last in-window lab value per lab type.

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort with ``hadm_id`` and ``dischtime``.
    labevents : pd.DataFrame
        MIMIC-IV labevents table.
    lab_itemids : dict
        Map from lab name → list of itemids.
    window_hours : int
        Look-back window in hours before discharge.

    Returns
    -------
    pd.DataFrame
        Index = hadm_id, columns = lab names (one value per lab).
    """
    log.info("Extracting lab features (window=%dh) …", window_hours)

    # Flatten itemid → lab name mapping
    itemid_to_name: dict[int, str] = {}
    for lab_name, ids in lab_itemids.items():
        for iid in ids:
            itemid_to_name[iid] = lab_name

    all_itemids = list(itemid_to_name.keys())

    # Filter labevents to relevant items
    labs = labevents[labevents["itemid"].isin(all_itemids)][
        ["hadm_id", "itemid", "charttime", "valuenum"]
    ].copy()
    labs = labs.dropna(subset=["valuenum", "hadm_id"])
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    labs["lab_name"] = labs["itemid"].map(itemid_to_name)

    # Join with dischtime
    cohort_time = cohort[["hadm_id", "dischtime"]].copy()
    cohort_time["dischtime"] = pd.to_datetime(cohort_time["dischtime"])
    labs = labs.merge(cohort_time, on="hadm_id", how="inner")

    # Filter to window
    labs = labs[
        (labs["charttime"] >= labs["dischtime"] - pd.Timedelta(hours=window_hours))
        & (labs["charttime"] <= labs["dischtime"])
    ]

    # For each hadm_id × lab_name take the value closest to discharge
    labs["time_to_discharge"] = (labs["dischtime"] - labs["charttime"]).dt.total_seconds()
    labs = labs.sort_values("time_to_discharge")
    last_labs = (
        labs.groupby(["hadm_id", "lab_name"])["valuenum"]
        .first()
        .unstack(fill_value=np.nan)
    )
    last_labs.columns = [f"lab_{c}" for c in last_labs.columns]
    log.info("  Lab feature matrix: %d rows × %d cols", *last_labs.shape)
    return last_labs


def extract_vital_features(
    cohort: pd.DataFrame,
    chartevents: pd.DataFrame,
    vital_itemids: dict[str, list[int]],
    window_hours: int = 48,
) -> pd.DataFrame:
    """
    For each admission, compute mean/min/max of key vitals in the last window.

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort with ``hadm_id``, ``dischtime``.
    chartevents : pd.DataFrame
        MIMIC-IV chartevents table (large — consider chunked reading).
    vital_itemids : dict
        Map from vital name → list of itemids.
    window_hours : int
        Look-back window in hours before discharge.

    Returns
    -------
    pd.DataFrame
        Index = hadm_id, columns = vital_{name}_{stat} (mean/min/max).
    """
    log.info("Extracting vital features (window=%dh) …", window_hours)

    itemid_to_name: dict[int, str] = {}
    for vname, ids in vital_itemids.items():
        for iid in ids:
            itemid_to_name[iid] = vname
    all_ids = list(itemid_to_name.keys())

    vitals = chartevents[chartevents["itemid"].isin(all_ids)][
        ["hadm_id", "itemid", "charttime", "valuenum"]
    ].copy()
    vitals = vitals.dropna(subset=["valuenum", "hadm_id"])
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    vitals["vital_name"] = vitals["itemid"].map(itemid_to_name)

    cohort_time = cohort[["hadm_id", "dischtime"]].copy()
    cohort_time["dischtime"] = pd.to_datetime(cohort_time["dischtime"])
    vitals = vitals.merge(cohort_time, on="hadm_id", how="inner")

    vitals = vitals[
        (vitals["charttime"] >= vitals["dischtime"] - pd.Timedelta(hours=window_hours))
        & (vitals["charttime"] <= vitals["dischtime"])
    ]

    agg = (
        vitals.groupby(["hadm_id", "vital_name"])["valuenum"]
        .agg(["mean", "min", "max"])
        .unstack()
    )
    agg.columns = [f"vital_{stat}_{name}" for stat, name in agg.columns]
    log.info("  Vital feature matrix: %d rows × %d cols", *agg.shape)
    return agg


# ── Demographic features ──────────────────────────────────────────────────────

def extract_demographic_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extract age, gender, and race dummy features from the cohort.

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort DataFrame with ``hadm_id``, ``age``, ``gender``, ``race`` columns.

    Returns
    -------
    pd.DataFrame
        Index = hadm_id, demographic feature columns.
    """
    demo = cohort[["hadm_id", "age", "gender"]].copy().set_index("hadm_id")
    demo["is_male"] = (demo["gender"].str.upper() == "M").astype(int)
    demo = demo.drop(columns=["gender"])

    # Broad race grouping to limit cardinality
    if "race" in cohort.columns:
        race = cohort.set_index("hadm_id")["race"].str.upper().fillna("UNKNOWN")
        simplified = race.map(lambda r: (
            "WHITE"    if "WHITE" in r else
            "BLACK"    if "BLACK" in r else
            "HISPANIC" if "HISPANIC" in r or "LATINO" in r else
            "ASIAN"    if "ASIAN" in r else
            "OTHER"
        ))
        race_dummies = pd.get_dummies(simplified, prefix="race").astype(int)
        demo = demo.join(race_dummies)

    return demo


# ── Admin features ────────────────────────────────────────────────────────────

def extract_admin_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extract administrative features: LOS, ED flag, prior admission counts.

    Parameters
    ----------
    cohort : pd.DataFrame
        Full cohort with computed admin columns.

    Returns
    -------
    pd.DataFrame
        Index = hadm_id, admin feature columns.
    """
    cols = ["hadm_id", "los_days", "via_ed", "prior_admits_12m", "ed_visits_6m"]
    admin = cohort[cols].copy().set_index("hadm_id")
    admin["via_ed"] = admin["via_ed"].astype(int)
    return admin


# ── Full feature matrix ───────────────────────────────────────────────────────

def build_feature_matrix(
    cohort: pd.DataFrame,
    cfg: DictConfig,
    labevents: pd.DataFrame | None = None,
    chartevents: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Assemble the full tabular feature matrix for a cohort split.

    Parameters
    ----------
    cohort : pd.DataFrame
        One of train/val/test cohort DataFrames.
    cfg : DictConfig
        Loaded project config.
    labevents : pd.DataFrame, optional
        Full labevents table. Loaded from disk if not provided.
    chartevents : pd.DataFrame, optional
        Full chartevents table. Loaded from disk if not provided.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix indexed by hadm_id.
    y : pd.Series
        Binary readmission labels indexed by hadm_id.
    feature_names : list of str
        Ordered list of feature column names.
    """
    mimic_hosp = Path(cfg.paths.mimic_iv_dir) / "hosp"
    mimic_icu  = Path(cfg.paths.mimic_iv_dir) / "icu"

    cohort_hadm_ids = set(cohort["hadm_id"].unique())

    # Load tables if not supplied (allows passing pre-loaded for efficiency)
    if labevents is None:
        log.info("Loading and filtering labevents in chunks …")
        lab_path = mimic_hosp / "labevents.csv.gz"
        if not lab_path.exists():
            lab_path = mimic_hosp / "labevents.csv"
        
        chunks = []
        for chunk in pd.read_csv(lab_path, chunksize=1000000, low_memory=False):
            # Keep only the rows for our cohort
            filtered_chunk = chunk[chunk["hadm_id"].isin(cohort_hadm_ids)]
            chunks.append(filtered_chunk)
        labevents = pd.concat(chunks, ignore_index=True)
        log.info("  Loaded %d labevents records", len(labevents))

    if chartevents is None:
        log.info("Loading and filtering chartevents in chunks …")
        ce_path = mimic_icu / "chartevents.csv.gz"
        if not ce_path.exists():
            ce_path = mimic_icu / "chartevents.csv"
        
        chunks = []
        # chartevents is very large — read only necessary columns
        for chunk in pd.read_csv(
            ce_path, usecols=["hadm_id", "itemid", "charttime", "valuenum"],
            chunksize=1000000, low_memory=False
        ):
            filtered_chunk = chunk[chunk["hadm_id"].isin(cohort_hadm_ids)]
            chunks.append(filtered_chunk)
        chartevents = pd.concat(chunks, ignore_index=True)
        log.info("  Loaded %d chartevents records", len(chartevents))

    lab_itemids   = {k: list(v) for k, v in cfg.tabular.lab_itemids.items()}
    vital_itemids = {k: list(v) for k, v in cfg.tabular.vital_itemids.items()}

    lab_feats    = extract_lab_features(cohort, labevents, lab_itemids)
    vital_feats  = extract_vital_features(cohort, chartevents, vital_itemids)
    demo_feats   = extract_demographic_features(cohort).set_index(
        cohort.set_index("hadm_id").index if "hadm_id" in cohort.columns else cohort.index
    )
    admin_feats  = extract_admin_features(cohort)

    # Align all on hadm_id index
    hadm_ids = cohort["hadm_id"].values
    X = (
        pd.DataFrame(index=hadm_ids)
        .join(demo_feats,  how="left")
        .join(admin_feats, how="left")
        .join(lab_feats,   how="left")
        .join(vital_feats, how="left")
    )
    X.index.name = "hadm_id"

    y = cohort.set_index("hadm_id")["readmitted_30d"]

    feature_names = X.columns.tolist()
    log.info(
        "Feature matrix built: %d patients × %d features | "
        "missing %.1f%%",
        len(X), len(feature_names),
        100 * X.isna().mean().mean(),
    )
    return X, y, feature_names
