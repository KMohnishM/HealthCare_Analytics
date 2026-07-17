"""
src/cohort/extract_cohort.py
-----------------------------
Extracts the heart failure cohort from MIMIC-IV and labels
30-day unplanned readmission outcomes.

Inputs  : MIMIC-IV hosp module CSVs (admissions, diagnoses_icd, patients)
Outputs : data/cohort/cohort.parquet

Usage:
    python scripts/run_cohort.py
    # or directly:
    from src.cohort.extract_cohort import build_cohort
    df = build_cohort(cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── ICD helpers ──────────────────────────────────────────────────────────────

def _is_hf_icd(row: pd.Series, icd10_prefix: str, icd9_codes: List[str]) -> bool:
    """Return True if the ICD code is a heart-failure diagnosis."""
    code = str(row["icd_code"]).strip()
    version = int(row["icd_version"])
    if version == 10:
        return code.startswith(icd10_prefix)
    else:
        return code in icd9_codes


# ── Cohort extraction ─────────────────────────────────────────────────────────

def load_mimic_tables(mimic_iv_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load required MIMIC-IV tables from the hosp module.

    Parameters
    ----------
    mimic_iv_dir : str
        Root directory of the MIMIC-IV release
        (e.g. ``/content/drive/MyDrive/MIMIC/mimic-iv-2.2``).

    Returns
    -------
    dict
        Keys: ``admissions``, ``patients``, ``diagnoses_icd``, ``procedures_icd``
    """
    hosp = Path(mimic_iv_dir) / "hosp"
    log.info("Loading MIMIC-IV tables from %s", hosp)

    tables = {}
    for name in ["admissions", "patients", "diagnoses_icd", "procedures_icd"]:
        # MIMIC-IV ships as both .csv.gz and plain .csv
        gz_path  = hosp / f"{name}.csv.gz"
        csv_path = hosp / f"{name}.csv"
        path = gz_path if gz_path.exists() else csv_path
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {name} at {gz_path} or {csv_path}. "
                "Check your MIMIC-IV path in config/config.yaml."
            )
        log.info("  Loading %s ...", path.name)
        tables[name] = pd.read_csv(path, low_memory=False)

    return tables


def extract_hf_admissions(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses_icd: pd.DataFrame,
    icd10_prefix: str,
    icd9_codes: List[str],
) -> pd.DataFrame:
    """
    Filter MIMIC-IV admissions to primary-diagnosis heart failure admissions
    and join basic demographic data.

    Primary diagnosis = ``seq_num == 1`` in diagnoses_icd.

    Parameters
    ----------
    admissions : pd.DataFrame
        MIMIC-IV admissions table.
    patients : pd.DataFrame
        MIMIC-IV patients table.
    diagnoses_icd : pd.DataFrame
        MIMIC-IV diagnoses_icd table.
    icd10_prefix : str
        ICD-10 code prefix for HF (default "I50").
    icd9_codes : list of str
        ICD-9 codes for HF.

    Returns
    -------
    pd.DataFrame
        One row per qualifying admission with demographic info.
    """
    log.info("Filtering primary-diagnosis HF admissions ...")

    # Primary diagnosis only (seq_num == 1 in MIMIC-IV, position may differ)
    primary_dx = diagnoses_icd[diagnoses_icd["seq_num"] == 1].copy()
    hf_mask = primary_dx.apply(
        _is_hf_icd, axis=1,
        icd10_prefix=icd10_prefix, icd9_codes=icd9_codes,
    )
    hf_hadm_ids = set(primary_dx.loc[hf_mask, "hadm_id"].unique())
    log.info("  Found %d HF admissions (primary dx)", len(hf_hadm_ids))

    # Filter admissions table
    adm = admissions[admissions["hadm_id"].isin(hf_hadm_ids)].copy()

    # Parse datetimes
    adm["admittime"]  = pd.to_datetime(adm["admittime"])
    adm["dischtime"]  = pd.to_datetime(adm["dischtime"])
    adm["edregtime"]  = pd.to_datetime(adm.get("edregtime", pd.NaT))

    # Compute length of stay in days
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400

    # Exclude in-hospital deaths (these patients cannot be readmitted)
    n_before = len(adm)
    adm = adm[adm["hospital_expire_flag"] == 0].copy()
    log.info(
        "  Excluded %d in-hospital deaths -> %d admissions remain",
        n_before - len(adm), len(adm),
    )

    # Derive ED flag (acute admission via emergency department)
    adm["via_ed"] = adm["admission_type"].str.upper().isin(
        {"EMERGENCY", "URGENT", "EW EMER."}
    )

    # Join patients table for demographics
    patients_clean = patients[["subject_id", "gender", "anchor_age", "anchor_year", "dod"]].copy()
    patients_clean["dod"] = pd.to_datetime(patients_clean["dod"], errors="coerce")

    adm = adm.merge(patients_clean, on="subject_id", how="left")

    # Compute approximate age at admission
    adm["age"] = (
        adm["anchor_age"]
        + (adm["admittime"].dt.year - adm["anchor_year"])
    ).clip(lower=0)

    # Age group for fairness analysis
    adm["age_group"] = pd.cut(
        adm["age"],
        bins=[0, 65, 80, 200],
        labels=["<65", "65-79", ">=80"],
        right=False,
    )

    log.info("  Final cohort before readmission labeling: %d rows", len(adm))
    return adm


def label_readmission(
    cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    readmission_days: int = 30,
    planned_types: tuple[str, ...] = ("ELECTIVE", "SCHEDULED"),
) -> pd.DataFrame:
    """
    Label each admission with 30-day unplanned readmission outcome using
    the raw unfiltered admissions table to prevent missing non-HF readmissions.
    """
    log.info("Labeling 30-day unplanned readmission using raw admissions ...")

    # Clean raw admissions for datetime parsing
    raw_adm = admissions.copy()
    raw_adm["admittime"] = pd.to_datetime(raw_adm["admittime"])
    raw_adm["dischtime"] = pd.to_datetime(raw_adm["dischtime"])

    # Map admission types that count as unplanned
    unplanned_mask = ~raw_adm["admission_type"].str.upper().isin(
        [t.upper() for t in planned_types]
    )
    unplanned_hadm_ids = set(raw_adm.loc[unplanned_mask, "hadm_id"])

    labels: dict[int, dict] = {}
    
    # Pre-index raw admissions by subject_id for fast lookup
    raw_adm_grouped = {
        sub_id: grp.sort_values("admittime").reset_index(drop=True)
        for sub_id, grp in raw_adm.groupby("subject_id")
    }

    for _, row in cohort.iterrows():
        hadm_id = row["hadm_id"]
        subject_id = row["subject_id"]
        dischtime = row["dischtime"]
        dod = row["dod"]           # may be NaT

        patient_visits = raw_adm_grouped.get(subject_id, pd.DataFrame())

        if len(patient_visits) > 0:
            # Look for subsequent unplanned admissions within 30 days
            future = patient_visits[
                (patient_visits["admittime"] > dischtime)
                & (patient_visits["admittime"] <= dischtime + pd.Timedelta(days=readmission_days))
                & (patient_visits["hadm_id"].isin(unplanned_hadm_ids))
            ].sort_values("admittime")

            if len(future) > 0:
                next_adm = future.iloc[0]
                days_delta = (next_adm["admittime"] - dischtime).total_seconds() / 86400
                labels[hadm_id] = {
                    "readmitted_30d": 1,
                    "competing_event": 0,
                    "days_to_readmit": days_delta,
                }
            else:
                # Check competing event: death within 30d without readmission
                died_within_30d = (
                    pd.notna(dod)
                    and dod <= dischtime + pd.Timedelta(days=readmission_days)
                )
                labels[hadm_id] = {
                    "readmitted_30d": 0,
                    "competing_event": int(died_within_30d),
                    "days_to_readmit": float("nan"),
                }
        else:
            labels[hadm_id] = {
                "readmitted_30d": 0,
                "competing_event": 0,
                "days_to_readmit": float("nan"),
            }

    label_df = pd.DataFrame.from_dict(labels, orient="index")
    label_df.index.name = "hadm_id"
    label_df = label_df.reset_index()

    cohort = cohort.merge(label_df, on="hadm_id", how="left")

    pos = cohort["readmitted_30d"].sum()
    comp = cohort["competing_event"].sum()
    log.info(
        "  Readmitted 30d: %d (%.1f%%) | Competing events: %d (%.1f%%)",
        pos, 100 * pos / len(cohort),
        comp, 100 * comp / len(cohort),
    )
    return cohort


def attach_prior_visit_counts(cohort: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Attach prior 12-month admission count and prior 6-month ED visit count
    for each index admission using the raw unfiltered admissions table.
    """
    log.info("Computing prior visit counts using raw admissions ...")

    raw_adm = admissions.copy()
    raw_adm["admittime"] = pd.to_datetime(raw_adm["admittime"])
    raw_adm["via_ed"] = raw_adm["admission_type"].str.upper().isin(
        {"EMERGENCY", "URGENT", "EW EMER."}
    )

    raw_adm_grouped = {
        sub_id: grp.sort_values("admittime").reset_index(drop=True)
        for sub_id, grp in raw_adm.groupby("subject_id")
    }

    prior_admits_12m = []
    ed_visits_6m = []

    for _, row in cohort.iterrows():
        subject_id = row["subject_id"]
        admit_t = row["admittime"]
        
        patient_visits = raw_adm_grouped.get(subject_id, pd.DataFrame())
        
        if len(patient_visits) > 0:
            cutoff_12m = admit_t - pd.Timedelta(days=365)
            cutoff_6m  = admit_t - pd.Timedelta(days=180)

            # Prior admissions (excluding current index admission)
            prior_12m = patient_visits[
                (patient_visits["admittime"] >= cutoff_12m) & (patient_visits["admittime"] < admit_t)
            ]
            prior_admits_12m.append(len(prior_12m))

            # Prior ED visits within 6 months
            ed_6m = patient_visits[
                (patient_visits["admittime"] >= cutoff_6m)
                & (patient_visits["admittime"] < admit_t)
                & (patient_visits["via_ed"] == True)
            ]
            ed_visits_6m.append(len(ed_6m))
        else:
            prior_admits_12m.append(0)
            ed_visits_6m.append(0)

    cohort["prior_admits_12m"] = prior_admits_12m
    cohort["ed_visits_6m"]     = ed_visits_6m

    log.info("  Done. Mean prior_admits_12m=%.2f, mean ed_visits_6m=%.2f",
             cohort["prior_admits_12m"].mean(), cohort["ed_visits_6m"].mean())
    return cohort


def build_cohort(cfg: DictConfig) -> pd.DataFrame:
    """
    Full cohort pipeline: load -> extract HF -> label readmission -> save.
    """
    tables = load_mimic_tables(cfg.paths.mimic_iv_dir)

    cohort = extract_hf_admissions(
        admissions    = tables["admissions"],
        patients      = tables["patients"],
        diagnoses_icd = tables["diagnoses_icd"],
        icd10_prefix  = cfg.cohort.icd10_hf_prefix,
        icd9_codes    = list(cfg.cohort.icd9_hf_codes),
    )

    # Attach procedure flag (needed for HOSPITAL score)
    procs = tables["procedures_icd"]
    has_proc = procs.groupby("hadm_id").size().rename("has_procedure")
    cohort["has_procedure"] = cohort["hadm_id"].map(has_proc).fillna(0).astype(bool)

    # Compute labels and prior visits using the unfiltered raw admissions table
    cohort = label_readmission(
        cohort,
        admissions=tables["admissions"],
        readmission_days=cfg.cohort.readmission_days,
    )

    cohort = attach_prior_visit_counts(cohort, admissions=tables["admissions"])

    # Save
    out_dir = Path(cfg.paths.cohort_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cohort.parquet"
    cohort.to_parquet(out_path, index=False)
    log.info("Saved cohort -> %s  (%d rows, %d cols)", out_path, *cohort.shape)

    return cohort

