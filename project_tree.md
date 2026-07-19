# Project Tree — HealthCare_Analytics

> Auto-generated project tree. Update when files are added/removed/renamed.

## Top-Level Structure

```
HealthCare_Analytics/
├── .git/                          # Git repository (excluded from listing below)
├── .gitignore                     # Git ignore rules: __pycache__, data/, outputs/, .venv/, etc.
├── README.md                      # Project overview & usage
├── repository_overview.md         # Architectural deep-dive
├── code_review_and_flaws.md       # Self-identified bugs & architectural flaws
├── project_tree.md                # THIS FILE — project directory tree
├── dependency_graph.md            # Module & pipeline dependency graph
├── file_metadata.md               # Per-file metadata (purpose, imports, exports)
├── requirements.txt               # Python package dependencies
│
├── config/                        # YAML configuration files
│   ├── config.yaml                #   Main config (Google Drive paths, full hyperparams)
│   └── config_dummy.yaml          #   Smoke-test config (local paths, reduced params)
│
├── scripts/                       # Executable pipeline scripts (run in order)
│   ├── run_cohort.py              #   1. Cohort extraction + split
│   ├── train_tabular.py           #   2a. XGBoost tabular branch
│   ├── train_ecg.py               #   2b. 1D ResNet ECG branch
│   ├── train_cxr.py               #   2c. DenseNet-121 CXR branch
│   ├── train_fusion.py            #   3. Learned gating fusion
│   ├── evaluate_all.py            #   4. Full evaluation (missingness sweep, DCA, fairness)
│   ├── generate_dummy_data.py     #   Synthetic MIMIC data generator
│   └── smoke_test.py              #   End-to-end pipeline smoke test
│
└── src/                           # Main Python package
    ├── __init__.py                #   Package marker
    │
    ├── cohort/                    #   Cohort extraction & splitting
    │   ├── __init__.py
    │   ├── extract_cohort.py      #     MIMIC-IV -> HF cohort with readmission labels
    │   └── split.py               #     Subject-level stratified train/val/test split
    │
    ├── tabular/                   #   Tabular EHR branch
    │   ├── __init__.py
    │   ├── features.py            #     Feature extraction (demographics, labs, vitals)
    │   ├── impute.py              #     Median imputation pipeline
    │   └── model.py               #     XGBoost bootstrap ensemble
    │
    ├── ecg/                       #   12-Lead ECG branch
    │   ├── __init__.py
    │   ├── preprocess.py          #     WFDB loading, filtering, resampling
    │   ├── dataset.py             #     PyTorch Dataset + DataLoader
    │   └── model.py               #     1D ResNet + MC-Dropout
    │
    ├── cxr/                       #   Chest X-Ray branch
    │   ├── __init__.py
    │   ├── preprocess.py          #     JPEG loading, augmentation, normalization
    │   ├── dataset.py             #     PyTorch Dataset + DataLoader
    │   └── model.py               #     DenseNet-121 + MC-Dropout
    │
    ├── fusion/                    #   Modality fusion
    │   ├── __init__.py
    │   ├── fixed_weight.py        #     Confidence-weighted average baseline
    │   └── learned_gate.py        #     Gated fusion + modality-dropout training
    │
    ├── baselines/                 #   Clinical baselines
    │   ├── __init__.py
    │   ├── lace.py                #     LACE index (LOS + Acuity + Comorbidity + ED)
    │   └── hospital_score.py      #     HOSPITAL score (7-component)
    │
    ├── evaluation/                #   Evaluation suite
    │   ├── __init__.py
    │   ├── metrics.py             #     AUROC, AUPRC, Brier, ECE
    │   ├── decision_curve.py      #     DCA net benefit curves
    │   ├── fairness.py            #     Subgroup fairness analysis
    │   └── missingness_sweep.py   #     7-modality-combination sweep
    │
    ├── explainability/            #   Model explainability
    │   ├── __init__.py
    │   ├── shap_tabular.py        #     SHAP TreeExplainer for XGBoost
    │   └── gradcam_cxr.py         #     Grad-CAM for DenseNet-121
    │
    ├── demo/                      #   Streamlit demo dashboard
    │   ├── __init__.py
    │   ├── app.py                 #     Main Streamlit app
    │   └── case_studies.py        #     4 pre-loaded case studies
    │
    └── utils/                     #   Shared utilities
        ├── __init__.py
        ├── config.py              #     OmegaConf loader
        ├── logger.py              #     Logging + MLflow init
        └── seed.py                #     Global reproducibility seed
```

## Git-Excluded / Generated Directories

These directories are listed in `.gitignore` and are **not tracked** by Git but will appear at runtime:

| Directory | Purpose | Created By |
|-----------|---------|------------|
| `data/` | Raw MIMIC data & generated dummies | User / `generate_dummy_data.py` |
| `data/processed_dummy/` | Smoke test intermediate outputs | `smoke_test.py` |
| `outputs/` | Models, results, figures, logs, MLflow | Training scripts |
| `__pycache__/` | Python bytecode (auto-generated) | Python interpreter |
| `.venv/`, `venv/`, `env/`, `ENV/` | Virtual environments | User |
| `.streamlit/` | Streamlit config cache | Streamlit |

## File Count Summary

| Category | Count | Details |
|----------|-------|---------|
| Python source files | 44 | All `.py` files in `src/` and `scripts/` |
| Config files | 2 | YAML |
| Documentation | 5 | Markdown (including these 3 tracking files) |
| Requirements | 1 | `requirements.txt` |
| Git config | 1 | `.gitignore` |
| **Total tracked** | **53** | Excluding `.git/` internals |
