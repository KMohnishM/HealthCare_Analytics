"""
src/demo/case_studies.py
-------------------------
Representative case study data for the demo app and viva walkthroughs.

4 cases covering:
  1. Full data, high risk — all modalities, learned gate vs fixed-weight agree
  2. Missing ECG, moderate risk — shows graceful gate degradation
  3. Missing CXR, low risk — calibration maintained
  4. ECG + CXR only (no tabular) — edge case, gate uncertainty
"""

from __future__ import annotations

from typing import Dict, List

# ── Case study definitions ─────────────────────────────────────────────────────

CASE_STUDIES: List[Dict] = [
    {
        "label":           "Full Data — High Risk Patient",
        "description":     (
            "74-year-old male with systolic HF (EF 25%), BNP 1850 pg/mL, "
            "creatinine 2.1 mg/dL, sodium 130 mEq/L. "
            "Prior 3 admissions in 12 months. Admitted via ED. LOS 9 days. "
            "ECG shows LBBB, CXR shows pulmonary edema."
        ),
        # Demographics
        "age": 74, "gender": "M", "los_days": 9.0,
        "via_ed": True, "prior_admits_12m": 3, "ed_visits_6m": 2,
        # Labs
        "creatinine": 2.1, "sodium": 130.0, "bnp": 1850.0,
        "hemoglobin": 10.2, "egfr": 35.0,
        # Availability
        "ecg_avail": True, "cxr_avail": True,
        # Branch scores (simulated from trained model)
        "tab_score": 0.71, "tab_conf": 0.88,
        "ecg_score": 0.68, "ecg_conf": 0.75,
        "cxr_score": 0.65, "cxr_conf": 0.72,
        # LACE / HOSPITAL
        "lace_score":     16, "lace_prob":    0.73,
        "hospital_score":  9, "hospital_prob": 0.71,
        "fused_gate":   0.70,
        "fused_fixed":  0.70,
        "key_insight": (
            "All modalities available and consistent. Learned gate and "
            "fixed-weight agree (minimal missingness penalty). High risk "
            "correctly identified — patient should receive same-day TCM referral."
        ),
    },
    {
        "label":           "Missing ECG — Moderate Risk",
        "description":     (
            "67-year-old female with diastolic HF. "
            "BNP 420 pg/mL, creatinine 1.4 mg/dL, sodium 138 mEq/L. "
            "No ECG available within 72h of discharge (ECG performed 5 days prior). "
            "CXR shows mild cardiomegaly only."
        ),
        "age": 67, "gender": "F", "los_days": 4.5,
        "via_ed": True, "prior_admits_12m": 1, "ed_visits_6m": 0,
        "creatinine": 1.4, "sodium": 138.0, "bnp": 420.0,
        "hemoglobin": 11.8, "egfr": 58.0,
        "ecg_avail": False, "cxr_avail": True,
        "tab_score": 0.38, "tab_conf": 0.82,
        "ecg_score": 0.0,  "ecg_conf": 0.0,   # unavailable
        "cxr_score": 0.31, "cxr_conf": 0.55,
        "lace_score": 9,   "lace_prob":    0.48,
        "hospital_score": 4, "hospital_prob": 0.38,
        "fused_gate":   0.35,
        "fused_fixed":  0.35,
        "key_insight": (
            "ECG unavailable. Learned gate automatically upweights tabular and CXR. "
            "Fixed-weight does the same via confidence masking. "
            "Score is moderate — both methods agree here because the tabular signal "
            "is clear. Gate calibration advantage becomes apparent when branch "
            "scores disagree (see Case 4)."
        ),
    },
    {
        "label":           "Missing CXR — Low Risk",
        "description":     (
            "58-year-old male with new-onset HF. "
            "BNP 185 pg/mL, creatinine 0.9 mg/dL, sodium 141 mEq/L. "
            "No chest X-ray within 72h of discharge (CXR from 4 days prior). "
            "ECG: sinus rhythm, normal QRS."
        ),
        "age": 58, "gender": "M", "los_days": 3.0,
        "via_ed": False, "prior_admits_12m": 0, "ed_visits_6m": 0,
        "creatinine": 0.9, "sodium": 141.0, "bnp": 185.0,
        "hemoglobin": 13.5, "egfr": 82.0,
        "ecg_avail": True, "cxr_avail": False,
        "tab_score": 0.18, "tab_conf": 0.91,
        "ecg_score": 0.15, "ecg_conf": 0.80,
        "cxr_score": 0.0,  "cxr_conf": 0.0,   # unavailable
        "lace_score": 7,   "lace_prob":    0.37,
        "hospital_score": 1, "hospital_prob": 0.22,
        "fused_gate":   0.17,
        "fused_fixed":  0.17,
        "key_insight": (
            "CXR unavailable. Both tabular and ECG agree: low risk. "
            "Gate downweights absent CXR correctly. Fusion score (0.17) "
            "is well-calibrated in the low-risk range. "
            "Standard discharge with routine follow-up is appropriate."
        ),
    },
    {
        "label":           "Conflicting Modalities — Gate vs Fixed-Weight",
        "description":     (
            "81-year-old female with advanced HF and COPD. "
            "Tabular: low risk (stable labs, short LOS). "
            "ECG: complex arrhythmia pattern -> high risk signal. "
            "CXR: bilateral pleural effusions -> high risk signal. "
            "Shows where gate learns to override tabular when ECG+CXR disagree."
        ),
        "age": 81, "gender": "F", "los_days": 4.0,
        "via_ed": True, "prior_admits_12m": 2, "ed_visits_6m": 1,
        "creatinine": 1.1, "sodium": 136.0, "bnp": 780.0,
        "hemoglobin": 11.0, "egfr": 55.0,
        "ecg_avail": True, "cxr_avail": True,
        "tab_score": 0.28, "tab_conf": 0.75,   # Tabular says low-moderate
        "ecg_score": 0.72, "ecg_conf": 0.65,   # ECG says high
        "cxr_score": 0.69, "cxr_conf": 0.60,   # CXR says high
        "lace_score": 11,  "lace_prob":    0.55,
        "hospital_score": 6, "hospital_prob": 0.52,
        "fused_gate":   0.60,  # Gate correctly upweights ECG+CXR
        "fused_fixed":  0.51,  # Fixed-weight also upweights (via confidence) but less decisively
        "key_insight": (
            "KEY CASE: Tabular data underestimates risk (stable labs despite advanced disease). "
            "ECG and CXR both signal high risk. "
            "The learned gate upweights ECG+CXR over tabular (score=0.60 vs fixed=0.51). "
            "LACE/HOSPITAL also flag moderate-high risk. "
            "This case demonstrates why multimodal fusion adds value beyond EHR alone."
        ),
    },
]


def get_case_study_display(case: dict) -> dict:
    """
    Format a case study dict for Streamlit display.

    Parameters
    ----------
    case : dict
        One entry from CASE_STUDIES.

    Returns
    -------
    dict with keys: 'profile', 'scores', 'interpretation'.
    """
    profile = (
        f"- **Age/Gender**: {case['age']}y {case['gender']}\n"
        f"- **LOS**: {case['los_days']} days | **Via ED**: {'Yes' if case['via_ed'] else 'No'}\n"
        f"- **Prior admits (12m)**: {case['prior_admits_12m']}\n"
        f"- **BNP**: {case['bnp']} pg/mL\n"
        f"- **Creatinine**: {case['creatinine']} mg/dL | **Sodium**: {case['sodium']} mEq/L\n"
        f"- **ECG available**: {'✅' if case['ecg_avail'] else '❌'} | "
        f"**CXR available**: {'✅' if case['cxr_avail'] else '❌'}\n"
    )

    def tier_emoji(p: float) -> str:
        return "🔴" if p >= 0.5 else "🟡" if p >= 0.25 else "🟢"

    scores = [
        f"| Model | Score | Risk |",
        f"|-------|-------|------|",
        f"| 📋 Tabular only | {case['tab_score']:.3f} | {tier_emoji(case['tab_score'])} |",
        f"| ⚙️ Fixed-Weight Fusion | {case['fused_fixed']:.3f} | {tier_emoji(case['fused_fixed'])} |",
        f"| 🧠 Learned Gate Fusion | {case['fused_gate']:.3f} | {tier_emoji(case['fused_gate'])} |",
        f"| 📊 LACE ({case['lace_score']}/19) | {case['lace_prob']:.3f} | {tier_emoji(case['lace_prob'])} |",
        f"| 📊 HOSPITAL ({case['hospital_score']}/13) | {case['hospital_prob']:.3f} | {tier_emoji(case['hospital_prob'])} |",
    ]

    return {
        "profile":        profile,
        "scores":         scores,
        "interpretation": case["key_insight"],
    }
