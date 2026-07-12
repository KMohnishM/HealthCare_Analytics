# Multimodal Heart Failure 30-Day Readmission Prediction

> **Multimodal Fusion of Electronic Health Records, ECG Waveforms, and Chest Radiographs for 30-Day Heart Failure Readmission Risk Prediction Under Data Missingness**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![Datasets](https://img.shields.io/badge/Data-MIMIC--IV%20%7C%20MIMIC--IV--ECG%20%7C%20MIMIC--CXR-green.svg)](https://physionet.org)

---

## Overview

This repository implements a three-branch multimodal ML system for predicting 30-day unplanned heart failure hospital readmissions. The core contribution is a **learned gating fusion layer trained with modality-dropout** that maintains reliable risk scores even when one or more data modalities are unavailable at inference time — the condition most likely to occur in real clinical deployments.

### Architecture at a Glance

```
Patient Data
├── EHR / Tabular    → XGBoost Bootstrap Ensemble (N=20)  → score + confidence
├── 12-Lead ECG      → 1D ResNet + MC-Dropout             → score + confidence + embed
└── Chest X-Ray      → DenseNet-121 (frozen) + MC-Dropout → score + confidence + embed
                                        ↓
                          ┌─────────────────────────┐
                          │  Learned Gating Fusion   │
                          │  (trained with p=0.3     │
                          │   modality dropout)      │
                          └─────────────────────────┘
                                        ↓
                           30-Day Readmission Risk Score
```

**Baselines**: LACE score, HOSPITAL score (recomputed on same cohort)

**Core Experiment**: Missingness sweep — 7 modality combinations × 2 fusion strategies × 4 metrics

---

## Repository Structure

```
HealthCare_Analytics/
├── config/
│   └── config.yaml              ← All paths, hyperparameters (edit this first)
├── src/
│   ├── cohort/                  ← Cohort extraction, labeling, splitting
│   ├── tabular/                 ← XGBoost features, imputation, ensemble
│   ├── ecg/                     ← ECG preprocessing, 1D ResNet, dataset
│   ├── cxr/                     ← CXR preprocessing, DenseNet-121, dataset
│   ├── baselines/               ← LACE, HOSPITAL scores
│   ├── fusion/                  ← Fixed-weight & learned gate fusion
│   ├── evaluation/              ← Metrics, missingness sweep, DCA, fairness
│   ├── explainability/          ← SHAP (tabular), Grad-CAM (CXR)
│   ├── utils/                   ← Config loader, logger, seed
│   └── demo/                    ← Streamlit app, case studies
├── scripts/
│   ├── run_cohort.py            ← Step 1: Extract cohort
│   ├── train_tabular.py         ← Step 2a: Train XGBoost branch
│   ├── train_ecg.py             ← Step 2b: Train ECG branch
│   ├── train_cxr.py             ← Step 2c: Train CXR branch
│   ├── train_fusion.py          ← Step 3: Train fusion layer
│   └── evaluate_all.py          ← Step 4: Full evaluation + figures
└── requirements.txt
```

---

## Quick Start (Google Colab)

### 1. Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo
!git clone https://github.com/KMohnishM/HealthCare_Analytics.git
%cd HealthCare_Analytics

# Install dependencies
!pip install -r requirements.txt
```

### 2. Configure Paths

Edit `config/config.yaml` to point to your MIMIC data:

```yaml
paths:
  mimic_iv_dir:  "/content/drive/MyDrive/MIMIC/mimic-iv-2.2"
  mimic_ecg_dir: "/content/drive/MyDrive/MIMIC/mimic-iv-ecg-1.0"
  mimic_cxr_dir: "/content/drive/MyDrive/MIMIC/mimic-cxr-jpg-2.0.0"
  base_dir:      "/content/drive/MyDrive/HealthCare_Analytics"
```

### 3. Run Pipeline (in order)

```bash
# Step 1: Extract cohort and split by subject_id
python scripts/run_cohort.py

# Step 2: Train branch models (can run in parallel on separate Colab notebooks)
python scripts/train_tabular.py
python scripts/train_ecg.py
python scripts/train_cxr.py

# Step 3: Train fusion layer (requires all branch outputs)
python scripts/train_fusion.py

# Step 4: Full evaluation (missingness sweep + DCA + fairness)
python scripts/evaluate_all.py
```

### 4. Launch Demo App

```bash
streamlit run src/demo/app.py

# On Colab:
!streamlit run src/demo/app.py &
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8501)"))
```

---

## Data Requirements

> **PhysioNet credentialed access required**: Apply at https://physionet.org/settings/credentialing/

| Dataset | Version | Size | Purpose |
|---------|---------|------|---------|
| MIMIC-IV | 2.2 | ~6 GB | EHR (admissions, labs, vitals, diagnoses) |
| MIMIC-IV-ECG | 1.0 | ~90 GB | 12-lead ECG waveforms (WFDB format) |
| MIMIC-CXR-JPG | 2.0.0 | ~40 GB | Chest radiograph images (JPEG, 224×224) |

All three datasets share `subject_id` for patient-level linkage.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tabular model | XGBoost bootstrap ensemble (N=20) | Confidence via prediction std; handles mixed feature types |
| ECG model | 1D ResNet + MC-Dropout | Better than plain CNN for long sequences; uncertainty from MC-Dropout |
| CXR model | DenseNet-121 (frozen) + head | Transfer learning avoids overfitting on small positive class |
| Fusion | Learned gate + modality dropout | Gate learns to reweight under missingness — not just handle noise |
| Confidence | 1 − normalized prediction std (tabular) / 1 − normalized MC-Dropout std (ECG/CXR) | Modality-agnostic confidence scale for gate input |
| Split | Subject-level stratified 70/10/20 | No patient data leakage between train and test |
| Imputation | Median (training set only) | Documented as limitation; consistent with clinical baselines |

---

## Metrics Reported

| Metric | Why |
|--------|-----|
| AUROC | Discrimination |
| AUPRC | Better for imbalanced outcome (~12% readmission rate) |
| Brier Score | Probabilistic accuracy |
| ECE (10 bins) | Calibration — key for missingness comparison |

**Core expected finding**: Learned gate maintains lower ECE than fixed-weight fusion as modalities are removed from the 7-combination sweep.

---

## Limitations

- Single-center data (MIMIC-IV, Beth Israel Deaconess Medical Center)
- Retrospective validation only — no prospective clinical testing
- Median imputation for missing labs is a simplification
- Not validated across hospital systems outside BIDMC
- For research/academic use only — not for clinical deployment

---

## Team

3-person team, 6-week project.

| Role | Scope |
|------|-------|
| Person A | Cohort extraction, tabular branch, baselines |
| Person B | ECG preprocessing and branch model |
| Person C | CXR preprocessing and branch model |
| Shared | Fusion layer, evaluation, explainability, demo |

---

## References

1. van Walraven C et al. (2010). LACE index. *CMAJ*.
2. Donzé J et al. (2014). HOSPITAL score. *Circulation*.
3. Johnson AEW et al. (2023). MIMIC-IV. *Scientific Data*.
4. Goldberger AL et al. (2000). PhysioBank. *Circulation*.
5. Huang G et al. (2017). DenseNet. *CVPR*.
6. Selvaraju RR et al. (2017). Grad-CAM. *ICCV*.
7. Lundberg SM & Lee SI (2017). SHAP. *NeurIPS*.
