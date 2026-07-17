"""
src/baselines/lace.py
----------------------
LACE score computation for the HF cohort.

LACE = Length of stay + Acuity + Comorbidity (Charlson) + Emergency visits

Score range: 0–19. High risk threshold: ≥ 10.
Published reference: van Walraven et al., CMAJ 2010.

Usage:
    from src.baselines.lace import compute_lace_scores
    lace_df = compute_lace_scores(cohort, diagnoses_icd, cfg)
    # lace_df has columns: hadm_id, lace_score, lace_prob
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Component functions ───────────────────────────────────────────────────────

def lace_l(los_days: float) -> int:
    """Length of stay component (L). Range: 1–7."""
    d = int(np.floor(los_days))
    if d < 1:  return 1
    if d == 1: return 1
    if d == 2: return 2
    if d == 3: return 3
    if d <= 6: return 4
    if d <= 13: return 5
    return 7


def lace_a(via_ed: bool) -> int:
    """Acuity component (A): 3 if admitted via ED, else 0."""
    return 3 if via_ed else 0


def lace_c(charlson_index: int) -> int:
    """Comorbidity component (C) from Charlson Index. Range: 0–5."""
    if charlson_index <= 0: return 0
    if charlson_index == 1: return 1
    if charlson_index == 2: return 2
    if charlson_index == 3: return 3
    return 5  # ≥4


def lace_e(ed_visits_6m: int) -> int:
    """ED visits in prior 6 months component (E). Range: 0–4."""
    return min(int(ed_visits_6m), 4)


def compute_charlson_index(
    hadm_ids: pd.Series,
    diagnoses_icd: pd.DataFrame,
) -> pd.Series:
    """
    Compute Charlson Comorbidity Index (CCI) for each admission.

    Uses the Quan et al. (2005) ICD-10 coding scheme via the
    ``comorbidipy`` package if available, otherwise applies a
    simplified ICD-10 prefix mapping.

    Parameters
    ----------
    hadm_ids : pd.Series
        Admission IDs to compute CCI for.
    diagnoses_icd : pd.DataFrame
        MIMIC-IV diagnoses_icd table.

    Returns
    -------
    pd.Series
        Index = hadm_id, values = CCI integer score.
    """
    try:
        from comorbidipy import comorbidity

        # comorbidipy expects a DataFrame with 'id' and 'code' columns
        dx = diagnoses_icd[diagnoses_icd["hadm_id"].isin(hadm_ids)].copy()
        dx = dx[dx["icd_version"] == 10][["hadm_id", "icd_code"]].rename(
            columns={"hadm_id": "id", "icd_code": "code"}
        )
        dx["code"] = dx["code"].str.replace(".", "", regex=False)

        cci_df = comorbidity(dx, id="id", code="code", mapping="charlson", assign0=True)
        cci_scores = cci_df["charlson_index"]
        cci_scores.index.name = "hadm_id"
        return cci_scores.reindex(hadm_ids, fill_value=0)

    except ImportError:
        log.warning(
            "comorbidipy not installed — using simplified CCI mapping. "
            "Install with: pip install comorbidipy"
        )
        return _simplified_charlson(hadm_ids, diagnoses_icd)


def _simplified_charlson(
    hadm_ids: pd.Series,
    diagnoses_icd: pd.DataFrame,
) -> pd.Series:
    """
    Simplified CCI using ICD-10 prefixes only.
    Weights are from Quan et al. 2005 (reduced set).
    """
    # (prefix, weight) pairs — 1-point conditions
    WEIGHT_1 = [
        "I21", "I22",           # AMI
        "I50",                  # CHF (not counted for our cohort but included for completeness)
        "I73",                  # PVD
        "I6",                   # CVD
        "F0",                   # Dementia
        "J4",                   # COPD
        "M0", "M3", "M33", "M34",  # Rheumatic
        "K25", "K26", "K27",    # PUD
        "B18", "K70",           # Mild liver
        "E10", "E11",           # Diabetes (no complication)
    ]
    WEIGHT_2 = [
        "E102", "E112",         # DM with complications
        "G81", "G82",           # Hemiplegia
        "N18", "N19",           # CKD
        "C0", "C1", "C2", "C3", "C4", "C5", "C6",  # Solid tumour
        "C91", "C92", "C93",    # Leukemia
        "C81", "C82", "C83",    # Lymphoma
    ]
    WEIGHT_3 = ["K72", "K76"]   # Moderate/severe liver
    WEIGHT_6 = ["C77", "C78", "C79", "C80", "B20", "B21", "B22"]  # Metastatic + AIDS

    dx_10 = diagnoses_icd[
        (diagnoses_icd["hadm_id"].isin(hadm_ids))
        & (diagnoses_icd["icd_version"] == 10)
    ][["hadm_id", "icd_code"]].copy()
    dx_10["icd_code"] = dx_10["icd_code"].fillna("").str.upper().str.replace(".", "", regex=False)

    cci_map = dict.fromkeys(hadm_ids, 0)

    for hadm_id, grp in dx_10.groupby("hadm_id"):
        codes = set(grp["icd_code"].tolist())
        score = 0
        for code in codes:
            for prefix in WEIGHT_1:
                if code.startswith(prefix):
                    score += 1
                    break
            for prefix in WEIGHT_2:
                if code.startswith(prefix):
                    score += 2
                    break
            for prefix in WEIGHT_3:
                if code.startswith(prefix):
                    score += 3
                    break
            for prefix in WEIGHT_6:
                if code.startswith(prefix):
                    score += 6
                    break
        cci_map[hadm_id] = score

    return pd.Series(cci_map).reindex(hadm_ids, fill_value=0)


# ── Full LACE pipeline ────────────────────────────────────────────────────────

def compute_lace_scores(
    cohort: pd.DataFrame,
    diagnoses_icd: pd.DataFrame,
    y_train: pd.Series | None = None,
    lace_train: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute LACE scores for all admissions and calibrate to probability.

    Calibration: fit a logistic regression mapping LACE score -> probability
    on the training set. Apply to full cohort for a fair same-population
    probability estimate.

    Parameters
    ----------
    cohort : pd.DataFrame
        Full cohort (or a split) with required LACE columns.
    diagnoses_icd : pd.DataFrame
        MIMIC-IV diagnoses_icd table (full).
    y_train : pd.Series, optional
        Training labels indexed by hadm_id. Required for calibration.
    lace_train : pd.Series, optional
        Pre-computed LACE scores for training split (to fit calibration).

    Returns
    -------
    pd.DataFrame
        Columns: hadm_id, lace_l, lace_a, lace_c, lace_e, lace_score, lace_prob.
    """
    log.info("Computing Charlson CCI for %d admissions ...", len(cohort))
    cci = compute_charlson_index(cohort["hadm_id"], diagnoses_icd)

    results = []
    for _, row in cohort.iterrows():
        hadm_id = row["hadm_id"]
        L = lace_l(row.get("los_days", 0))
        A = lace_a(bool(row.get("via_ed", False)))
        C = lace_c(int(cci.get(hadm_id, 0)))
        E = lace_e(int(row.get("ed_visits_6m", 0)))
        total = L + A + C + E
        results.append({
            "hadm_id":   hadm_id,
            "lace_l":    L,
            "lace_a":    A,
            "lace_c":    C,
            "lace_e":    E,
            "lace_score": total,
        })

    df = pd.DataFrame(results)

    # Calibrate score -> probability if training data provided
    if y_train is not None and lace_train is not None:
        log.info("Calibrating LACE score to probability ...")
        X_cal = lace_train.values.reshape(-1, 1)
        y_cal = y_train.values
        cal_model = LogisticRegression(max_iter=500)
        cal_model.fit(X_cal, y_cal)
        df["lace_prob"] = cal_model.predict_proba(
            df["lace_score"].values.reshape(-1, 1)
        )[:, 1]
    else:
        # Simple sigmoid normalisation as fallback (not calibrated)
        df["lace_prob"] = 1 / (1 + np.exp(-(df["lace_score"] - 10) / 3))

    log.info(
        "LACE scores: mean=%.2f, high-risk (>=10): %.1f%%",
        df["lace_score"].mean(),
        100 * (df["lace_score"] >= 10).mean(),
    )
    return df
