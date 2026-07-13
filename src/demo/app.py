"""
src/demo/app.py
----------------
Streamlit demo: Multimodal HF Readmission Risk Dashboard.

Run:
    streamlit run src/demo/app.py
    # or on Colab:
    !streamlit run src/demo/app.py &
    from google.colab.output import eval_js
    print(eval_js("google.colab.kernel.proxyPort(8501)"))

Features:
  - Patient data input (tabular fields, upload ECG/CXR)
  - Risk score display with risk tier badge
  - Branch-level score breakdown with confidence bars
  - Live modality toggle → score updates as modalities removed
  - SHAP waterfall tab (pre-loaded for demo patients)
  - Grad-CAM overlay tab
  - Case study walkthroughs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import streamlit as st

from src.demo.case_studies import CASE_STUDIES, get_case_study_display

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HF Readmission Risk | Multimodal AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .risk-badge-high {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white; padding: 0.6rem 1.5rem;
        border-radius: 50px; font-size: 1.3rem;
        font-weight: 700; display: inline-block;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
    }
    .risk-badge-medium {
        background: linear-gradient(135deg, #d35400, #e67e22);
        color: white; padding: 0.6rem 1.5rem;
        border-radius: 50px; font-size: 1.3rem;
        font-weight: 700; display: inline-block;
        box-shadow: 0 4px 15px rgba(230, 126, 34, 0.4);
    }
    .risk-badge-low {
        background: linear-gradient(135deg, #1e8449, #27ae60);
        color: white; padding: 0.6rem 1.5rem;
        border-radius: 50px; font-size: 1.3rem;
        font-weight: 700; display: inline-block;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4);
    }

    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.5rem;
    }

    .branch-score {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.8rem 1rem;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid;
    }

    .disclaimer {
        background: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #e74c3c;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Utility functions ─────────────────────────────────────────────────────────

def compute_demo_fusion(
    tab_score: float, tab_conf: float, tab_avail: bool,
    ecg_score: float, ecg_conf: float, ecg_avail: bool,
    cxr_score: float, cxr_conf: float, cxr_avail: bool,
) -> float:
    """Simple confidence-weighted fusion for real-time demo display."""
    scores = [(tab_score, tab_conf, tab_avail),
              (ecg_score, ecg_conf, ecg_avail),
              (cxr_score, cxr_conf, cxr_avail)]
    num, den = 0.0, 0.0
    for s, c, a in scores:
        if a:
            num += c * s
            den += c
    return num / den if den > 0 else 0.5


def risk_tier(prob: float) -> tuple[str, str]:
    """Return (tier_name, tier_class) based on probability."""
    if prob >= 0.5:
        return "HIGH RISK", "high"
    elif prob >= 0.25:
        return "MODERATE RISK", "medium"
    else:
        return "LOW RISK", "low"


def render_progress_bar(label: str, value: float, color: str, disabled: bool = False) -> str:
    opacity = "0.3" if disabled else "1.0"
    return f"""
    <div style="margin: 0.3rem 0; opacity: {opacity}">
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:3px">
            <span><b>{label}</b></span>
            <span style="color:{color}"><b>{value:.3f}</b></span>
        </div>
        <div style="background:#2d2d3d; border-radius:6px; height:10px; overflow:hidden">
            <div style="width:{value*100:.1f}%; background:{color};
                        height:100%; border-radius:6px;
                        box-shadow: 0 0 8px {color}60;
                        transition: width 0.4s ease;"></div>
        </div>
    </div>
    """


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🫀 Multimodal Heart Failure Readmission Risk</h1>
    <p style="color:#aaa; margin:0.3rem 0 0; font-size:0.95rem;">
        EHR · 12-Lead ECG · Chest Radiograph Fusion — 30-Day Unplanned Readmission Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Modality Availability ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Modality Availability")
    st.markdown("Toggle to simulate missing data at inference:")

    tab_avail = st.toggle("📋 EHR / Tabular", value=True, disabled=True,
                           help="Tabular data always required")
    ecg_avail = st.toggle("📈 ECG (12-Lead)",  value=True)
    cxr_avail = st.toggle("🩻 Chest X-Ray",    value=True)

    st.markdown("---")
    st.markdown("### 🎯 Select Demo Patient")
    case_names = ["Enter Manual Data"] + [f"Case {i+1}: {c['label']}" for i, c in enumerate(CASE_STUDIES)]
    selected_case = st.selectbox("", case_names)

    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.info(
        "**Tabular**: XGBoost ensemble (N=20)\n\n"
        "**ECG**: 1D ResNet + MC-Dropout\n\n"
        "**CXR**: DenseNet-121 + MC-Dropout\n\n"
        "**Fusion**: Learned gate with modality-dropout training"
    )


# ── Main content ──────────────────────────────────────────────────────────────

tab_input, tab_results, tab_missingness, tab_explain, tab_cases = st.tabs([
    "📥 Patient Input",
    "📊 Risk Output",
    "🔍 Missingness Impact",
    "🔬 Explainability",
    "📂 Case Studies",
])


# ── Tab 1: Patient Input ──────────────────────────────────────────────────────
with tab_input:
    st.subheader("Patient Data Entry")

    # Populate from case study if selected
    case_data = None
    if selected_case != "Enter Manual Data":
        case_idx  = case_names.index(selected_case) - 1
        case_data = CASE_STUDIES[case_idx]
        st.success(f"Loaded case: {case_data['label']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏥 Demographics & Admin**")
        age       = st.number_input("Age (years)", 18, 110,
                                     value=int(case_data["age"]) if case_data else 72)
        gender    = st.selectbox("Gender", ["Male", "Female"],
                                  index=0 if (case_data and case_data.get("gender","M")=="M") else 1)
        los_days  = st.number_input("Length of Stay (days)", 0.0, 60.0,
                                     value=float(case_data["los_days"]) if case_data else 5.0,
                                     step=0.5)
        via_ed    = st.checkbox("Admitted via Emergency Dept",
                                 value=bool(case_data.get("via_ed", True)) if case_data else True)
        prior_adm = st.number_input("Prior Admissions (12 months)", 0, 20,
                                     value=int(case_data.get("prior_admits_12m", 2)) if case_data else 2)
        ed_6m     = st.number_input("ED Visits (prior 6 months)", 0, 10,
                                     value=int(case_data.get("ed_visits_6m", 1)) if case_data else 1)

    with col2:
        st.markdown("**🧪 Key Lab Values**")
        creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 20.0,
                                      value=float(case_data.get("creatinine", 1.8)) if case_data else 1.8,
                                      step=0.1)
        sodium     = st.number_input("Sodium (mEq/L)", 110.0, 160.0,
                                      value=float(case_data.get("sodium", 136.0)) if case_data else 136.0,
                                      step=0.5)
        bnp        = st.number_input("BNP (pg/mL)", 0.0, 5000.0,
                                      value=float(case_data.get("bnp", 850.0)) if case_data else 850.0,
                                      step=10.0)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 4.0, 20.0,
                                      value=float(case_data.get("hemoglobin", 10.5)) if case_data else 10.5,
                                      step=0.1)
        egfr       = st.number_input("eGFR (mL/min/1.73m²)", 5.0, 130.0,
                                      value=float(case_data.get("egfr", 45.0)) if case_data else 45.0,
                                      step=1.0)

    with col3:
        st.markdown("**📈 Branch Score Overrides**")
        st.caption("Pre-computed branch risk scores (from trained models):")
        tab_score = st.slider("📋 Tabular Risk Score", 0.0, 1.0,
                               value=float(case_data.get("tab_score", 0.42)) if case_data else 0.42)
        tab_conf  = st.slider("📋 Tabular Confidence",  0.0, 1.0,
                               value=float(case_data.get("tab_conf",  0.85)) if case_data else 0.85)

        ecg_score = st.slider("📈 ECG Risk Score",       0.0, 1.0,
                               value=float(case_data.get("ecg_score", 0.55)) if case_data else 0.55,
                               disabled=not ecg_avail)
        ecg_conf  = st.slider("📈 ECG Confidence",        0.0, 1.0,
                               value=float(case_data.get("ecg_conf",  0.70)) if case_data else 0.70,
                               disabled=not ecg_avail)

        cxr_score = st.slider("🩻 CXR Risk Score",       0.0, 1.0,
                               value=float(case_data.get("cxr_score", 0.38)) if case_data else 0.38,
                               disabled=not cxr_avail)
        cxr_conf  = st.slider("🩻 CXR Confidence",        0.0, 1.0,
                               value=float(case_data.get("cxr_conf",  0.60)) if case_data else 0.60,
                               disabled=not cxr_avail)

    # Store in session state
    st.session_state["inputs"] = dict(
        tab_score=tab_score, tab_conf=tab_conf, tab_avail=True,
        ecg_score=ecg_score, ecg_conf=ecg_conf, ecg_avail=ecg_avail,
        cxr_score=cxr_score, cxr_conf=cxr_conf, cxr_avail=cxr_avail,
        age=age, gender=gender, los_days=los_days, via_ed=via_ed,
        bnp=bnp, creatinine=creatinine, sodium=sodium,
        hemoglobin=hemoglobin, egfr=egfr,
    )


# ── Tab 2: Risk Output ────────────────────────────────────────────────────────
with tab_results:
    inp = st.session_state.get("inputs", {
        "tab_score": 0.42, "tab_conf": 0.85, "tab_avail": True,
        "ecg_score": 0.55, "ecg_conf": 0.70, "ecg_avail": True,
        "cxr_score": 0.38, "cxr_conf": 0.60, "cxr_avail": True,
    })

    fused_prob = compute_demo_fusion(
        inp["tab_score"], inp["tab_conf"], inp["tab_avail"],
        inp["ecg_score"], inp["ecg_conf"], inp["ecg_avail"],
        inp["cxr_score"], inp["cxr_conf"], inp["cxr_avail"],
    )
    tier_name, tier_class = risk_tier(fused_prob)

    # ── Main risk display ─────────────────────────────────────────────────────
    st.markdown("### 🎯 Fused Risk Score")
    col_main, col_branch = st.columns([1, 1])

    with col_main:
        st.metric(
            label="30-Day Readmission Probability",
            value=f"{fused_prob:.1%}",
            delta="Above 50% threshold" if fused_prob >= 0.5 else "Below 50% threshold",
            delta_color="inverse",
        )
        st.markdown(
            f'<div class="risk-badge-{tier_class}">{tier_name}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # LACE / HOSPITAL reference scores
        from src.baselines.lace import lace_l, lace_a, lace_c, lace_e
        from src.baselines.hospital_score import (
            hospital_h, hospital_s, hospital_t, hospital_a
        )
        lace_val = (
            lace_l(inp.get("los_days", 5))
            + lace_a(inp.get("via_ed", True))
            + lace_c(1)
            + lace_e(inp.get("ed_visits_6m", 1) if "ed_visits_6m" in inp else 1)
        )
        lace_prob = 1 / (1 + np.exp(-(lace_val - 10) / 3))

        hosp_val = (
            hospital_h(inp.get("hemoglobin", 10.5))
            + hospital_s(inp.get("sodium", 136.0))
            + hospital_t(inp.get("prior_admits_12m", 2) if "prior_admits_12m" in inp else 2)
            + hospital_a(inp.get("los_days", 5))
        )
        hosp_prob = 1 / (1 + np.exp(-(hosp_val - 7) / 2))

        st.markdown("**Clinical Baselines for Reference**")
        st.caption(f"LACE Score: {lace_val}/19 → Prob ≈ {lace_prob:.1%}")
        st.caption(f"HOSPITAL Score: {hosp_val}/13 → Prob ≈ {hosp_prob:.1%}")

    with col_branch:
        st.markdown("**Branch Score Breakdown**")
        bars_html = ""
        bars_html += render_progress_bar(
            "📋 Tabular (EHR)", inp["tab_score"], "#3498db", disabled=False)
        bars_html += render_progress_bar(
            "📈 ECG (12-Lead)", inp["ecg_score"], "#9b59b6",
            disabled=not inp["ecg_avail"])
        bars_html += render_progress_bar(
            "🩻 Chest X-Ray",  inp["cxr_score"], "#1abc9c",
            disabled=not inp["cxr_avail"])

        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown("<br>**Fusion Gate Weights**", unsafe_allow_html=True)
        total_w = (inp["tab_conf"] + inp["ecg_conf"] * inp["ecg_avail"]
                   + inp["cxr_conf"] * inp["cxr_avail"])
        if total_w > 0:
            w_tab = inp["tab_conf"] / total_w
            w_ecg = inp["ecg_conf"] * inp["ecg_avail"] / total_w
            w_cxr = inp["cxr_conf"] * inp["cxr_avail"] / total_w
        else:
            w_tab = w_ecg = w_cxr = 1/3

        weight_html = (
            render_progress_bar("📋 Tab weight", w_tab, "#3498db")
            + render_progress_bar("📈 ECG weight", w_ecg, "#9b59b6",
                                   disabled=not inp["ecg_avail"])
            + render_progress_bar("🩻 CXR weight", w_cxr, "#1abc9c",
                                   disabled=not inp["cxr_avail"])
        )
        st.markdown(weight_html, unsafe_allow_html=True)

    # ── Discharge planning triage ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏥 Discharge Planning Guidance")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**📞 TCM Referral**")
        if fused_prob >= 0.5:
            st.error("✅ HIGH PRIORITY — Same-day TCM contact")
        elif fused_prob >= 0.25:
            st.warning("⚠️ MODERATE — Schedule within 48h")
        else:
            st.success("✓ Standard — Routine follow-up")

    with col_b:
        st.markdown("**📋 Care Plan**")
        if fused_prob >= 0.5:
            st.markdown("- Intensify HF education\n- Remote monitoring\n- Early cardiology follow-up")
        elif fused_prob >= 0.25:
            st.markdown("- Standard HF education\n- Weight monitoring\n- PCP follow-up in 7d")
        else:
            st.markdown("- Routine discharge\n- Standard follow-up")

    with col_c:
        st.markdown("**⚠️ Limitations**")
        st.markdown("""
        - Single-center data (MIMIC-IV, BIDMC)
        - Retrospective validation only
        - No prospective clinical testing
        - Model may underperform in populations unlike MIMIC cohort
        """)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Research Tool Only</b>: This system is intended for academic research and demonstration
    purposes. It has NOT been validated for clinical deployment and should NOT be used to make
    clinical decisions without physician oversight.
    </div>
    """, unsafe_allow_html=True)


# ── Tab 3: Missingness Impact ─────────────────────────────────────────────────
with tab_missingness:
    st.subheader("🔍 How Score Changes With Missing Modalities")

    inp = st.session_state.get("inputs", {
        "tab_score": 0.42, "tab_conf": 0.85,
        "ecg_score": 0.55, "ecg_conf": 0.70,
        "cxr_score": 0.38, "cxr_conf": 0.60,
    })

    combos = [
        ("Tab only",         True,  False, False),
        ("ECG only",         False, True,  False),
        ("CXR only",         False, False, True),
        ("Tab + ECG",        True,  True,  False),
        ("Tab + CXR",        True,  False, True),
        ("ECG + CXR",        False, True,  True),
        ("Tab + ECG + CXR",  True,  True,  True),
    ]

    import matplotlib.pyplot as plt

    combo_names   = [c[0] for c in combos]
    combo_scores  = [
        compute_demo_fusion(
            inp.get("tab_score", 0.42), inp.get("tab_conf", 0.85), t_a,
            inp.get("ecg_score", 0.55), inp.get("ecg_conf", 0.70), e_a,
            inp.get("cxr_score", 0.38), inp.get("cxr_conf", 0.60), c_a,
        )
        for _, t_a, e_a, c_a in combos
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#e74c3c" if s >= 0.5 else "#e67e22" if s >= 0.25 else "#27ae60"
              for s in combo_scores]
    bars = ax.barh(combo_names, combo_scores, color=colors, alpha=0.85, height=0.6)
    ax.axvline(0.5,  color="red",    linestyle="--", alpha=0.7, lw=1.5, label="High risk threshold (0.5)")
    ax.axvline(0.25, color="orange", linestyle="--", alpha=0.7, lw=1.5, label="Moderate risk threshold (0.25)")
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Predicted 30-Day Readmission Probability")
    ax.set_title("Risk Score Under Different Modality Availability Scenarios", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        "The fixed-weight baseline (shown here) becomes overconfident as modalities are removed. "
        "The learned gate model maintains better calibration under missingness — "
        "this is the key finding of the missingness sweep experiment."
    )


# ── Tab 4: Explainability ─────────────────────────────────────────────────────
with tab_explain:
    st.subheader("🔬 Model Explainability")

    col_shap, col_gcam = st.columns(2)
    with col_shap:
        st.markdown("**SHAP — Tabular Feature Contributions**")
        st.info(
            "SHAP values are computed after training and show which EHR features "
            "drive the readmission risk prediction. Run `python scripts/evaluate_all.py` "
            "to generate SHAP plots for the test set.\n\n"
            "**Expected top drivers**: prior admission count, BNP, sodium, "
            "length of stay, creatinine."
        )
        shap_fig_path = Path("outputs/figures/shap_beeswarm.png")
        if shap_fig_path.exists():
            st.image(str(shap_fig_path), caption="Global SHAP Beeswarm Plot", use_container_width=True)
        else:
            st.warning("SHAP plots not yet generated. Train models and run `scripts/evaluate_all.py`.")

    with col_gcam:
        st.markdown("**Grad-CAM — CXR Region Attribution**")
        st.info(
            "Grad-CAM highlights which regions of the chest X-ray most influence "
            "the risk prediction. Expected focus: bilateral lung fields, "
            "cardiomegaly regions, pleural effusion areas."
        )
        gcam_path = Path("outputs/figures")
        gcam_files = list(gcam_path.glob("gradcam_*.png")) if gcam_path.exists() else []
        if gcam_files:
            for f in gcam_files[:2]:
                st.image(str(f), use_container_width=True)
        else:
            st.warning("Grad-CAM images not yet generated. Train CXR model first.")


# ── Tab 5: Case Studies ───────────────────────────────────────────────────────
with tab_cases:
    st.subheader("📂 Representative Case Walkthroughs")
    st.markdown("These 4 cases illustrate how the model behaves across different data scenarios.")

    for i, case in enumerate(CASE_STUDIES):
        with st.expander(f"Case {i+1}: {case['label']}", expanded=(i == 0)):
            display = get_case_study_display(case)
            col_info, col_scores = st.columns([1, 1])
            with col_info:
                st.markdown(f"**Patient Profile**")
                st.markdown(display["profile"])
            with col_scores:
                st.markdown("**Risk Score Comparison**")
                for row in display["scores"]:
                    st.markdown(row)
            st.markdown(f"**Clinical Interpretation**: {display['interpretation']}")
