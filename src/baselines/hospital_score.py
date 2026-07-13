"""
src/baselines/hospital_score.py
---------------------------------
HOSPITAL score computation for the HF cohort.

HOSPITAL score components (0–13):
  H — Hemoglobin at discharge < 12 g/dL         → 1 point
  O — Oncology service                           → 2 points
  S — Sodium at discharge < 135 mmol/L           → 1 point
  P — Any ICD procedure during stay              → 1 point
  I — Index admission (urgent/emergent)          → 1 point
  T — Total admissions in prior 12 months        → 0/2/5 points
  A — Admission length of stay ≥ 5 days          → 2 points

Risk tiers: Low 0-4 | Intermediate 5-6 | High ≥7
Reference: Donzé et al., Circulation 2014.

Usage:
    from src.baselines.hospital_score import compute_hospital_scores
    hosp_df = compute_hospital_scores(cohort, labevents)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

log = get_logger(__name__)

# Lab itemids for hemoglobin and sodium in MIMIC-IV
HGB_ITEMIDS  = [51222]           # Hemoglobin
SOD_ITEMIDS  = [50983]           # Sodium

# Oncology-related DRG/service codes (simplified)
ONCOLOGY_SERVICES = {"oncology", "hematology", "hem/onc", "medical oncology"}


def _get_last_lab_value(
    hadm_id: int,
    itemids: list[int],
    labevents: pd.DataFrame,
    dischtime: pd.Timestamp,
    window_h: int = 72,
) -> float:
    """Get the most recent lab value within window_h of discharge."""
    subset = labevents[
        (labevents["hadm_id"] == hadm_id)
        & (labevents["itemid"].isin(itemids))
        & (labevents["charttime"] >= dischtime - pd.Timedelta(hours=window_h))
        & (labevents["charttime"] <= dischtime)
    ]
    if subset.empty:
        return float("nan")
    return subset.sort_values("charttime").iloc[-1]["valuenum"]


def hospital_h(hgb: float) -> int:
    """H component: hemoglobin < 12 → 1 point."""
    if np.isnan(hgb):
        return 0   # treat missing as normal (conservative)
    return 1 if hgb < 12.0 else 0


def hospital_o(current_service: str) -> int:
    """O component: oncology service → 2 points."""
    if pd.isna(current_service):
        return 0
    return 2 if str(current_service).lower().strip() in ONCOLOGY_SERVICES else 0


def hospital_s(sodium: float) -> int:
    """S component: sodium < 135 → 1 point."""
    if np.isnan(sodium):
        return 0
    return 1 if sodium < 135.0 else 0


def hospital_p(has_procedure: bool) -> int:
    """P component: any ICD-coded procedure during stay → 1 point."""
    return 1 if has_procedure else 0


def hospital_i(via_ed: bool) -> int:
    """I component: urgent/emergent admission → 1 point."""
    return 1 if via_ed else 0


def hospital_t(prior_admits_12m: int) -> int:
    """T component: total prior admissions in 12 months → 0/2/5 points."""
    if prior_admits_12m <= 1:
        return 0
    elif prior_admits_12m <= 5:
        return 2
    else:
        return 5


def hospital_a(los_days: float) -> int:
    """A component: LOS ≥ 5 days → 2 points."""
    return 2 if los_days >= 5.0 else 0


def compute_hospital_scores(
    cohort: pd.DataFrame,
    labevents: pd.DataFrame,
    y_train: pd.Series | None = None,
    hosp_train: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute HOSPITAL scores for all admissions and calibrate to probability.

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort with required columns: hadm_id, dischtime, los_days, via_ed,
        has_procedure, prior_admits_12m, and optionally current_service.
    labevents : pd.DataFrame
        MIMIC-IV labevents table (filtered to HGB and SOD itemids is fine).
    y_train : pd.Series, optional
        Training labels indexed by hadm_id (for calibration).
    hosp_train : pd.Series, optional
        Pre-computed HOSPITAL scores for training split.

    Returns
    -------
    pd.DataFrame
        Columns: hadm_id, H, O, S, P, I, T, A,
                 hospital_score, hospital_risk_tier, hospital_prob.
    """
    log.info("Computing HOSPITAL scores for %d admissions …", len(cohort))

    # Filter labevents to only needed itemids for performance
    labs = labevents[
        labevents["itemid"].isin(HGB_ITEMIDS + SOD_ITEMIDS)
    ].copy()
    labs["charttime"] = pd.to_datetime(labs["charttime"])

    dischtime_map = dict(zip(
        cohort["hadm_id"],
        pd.to_datetime(cohort["dischtime"])
    ))

    results = []
    for _, row in cohort.iterrows():
        hadm_id   = row["hadm_id"]
        dischtime = dischtime_map[hadm_id]

        hgb    = _get_last_lab_value(hadm_id, HGB_ITEMIDS, labs, dischtime)
        sodium = _get_last_lab_value(hadm_id, SOD_ITEMIDS, labs, dischtime)

        service = row.get("current_service", "")

        H = hospital_h(hgb)
        O = hospital_o(service)
        S = hospital_s(sodium)
        P = hospital_p(bool(row.get("has_procedure", False)))
        I = hospital_i(bool(row.get("via_ed", False)))
        T = hospital_t(int(row.get("prior_admits_12m", 0)))
        A = hospital_a(float(row.get("los_days", 0)))

        total = H + O + S + P + I + T + A

        if total <= 4:
            tier = "low"
        elif total <= 6:
            tier = "intermediate"
        else:
            tier = "high"

        results.append({
            "hadm_id":           hadm_id,
            "H":                 H, "O": O, "S": S, "P": P,
            "I":                 I, "T": T, "A": A,
            "hospital_score":    total,
            "hospital_risk_tier": tier,
        })

    df = pd.DataFrame(results)

    # Calibrate to probability
    if y_train is not None and hosp_train is not None:
        log.info("Calibrating HOSPITAL score to probability …")
        X_cal = hosp_train.values.reshape(-1, 1)
        y_cal = y_train.values
        cal_model = LogisticRegression(max_iter=500)
        cal_model.fit(X_cal, y_cal)
        df["hospital_prob"] = cal_model.predict_proba(
            df["hospital_score"].values.reshape(-1, 1)
        )[:, 1]
    else:
        # Fallback: sigmoid centred at threshold 7
        df["hospital_prob"] = 1 / (1 + np.exp(-(df["hospital_score"] - 7) / 2))

    log.info(
        "HOSPITAL scores: mean=%.2f | low=%.1f%% | mid=%.1f%% | high=%.1f%%",
        df["hospital_score"].mean(),
        100 * (df["hospital_risk_tier"] == "low").mean(),
        100 * (df["hospital_risk_tier"] == "intermediate").mean(),
        100 * (df["hospital_risk_tier"] == "high").mean(),
    )
    return df
