# Dependency Graph — HealthCare_Analytics

> Module dependency graph and pipeline data flow. Update when imports change or pipeline steps are modified.

---

## 1. Pipeline Data Flow (Execution Order)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. run_cohort.py                                                           │
│     ├── src.cohort.extract_cohort.build_cohort()       -> cohort.parquet   │
│     ├── src.cohort.split.split_cohort()                -> train/val/test   │
│     └── src.utils.{config,logger,seed}                                     │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │
                ├──────────────────────┬──────────────────────┐
                ▼                      ▼                      ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│ 2a. train_tabular.py  │ │ 2b. train_ecg.py      │ │ 2c. train_cxr.py      │
│                       │ │                       │ │                       │
│ XGBoost Ensemble      │ │ 1D ResNet + MC-Dropout│ │ DenseNet-121 + MC-Drop│
│ (N=20 bootstrap)      │ │                       │ │                       │
│                       │ │                       │ │                       │
│ Outputs:              │ │ Outputs:              │ │ Outputs:              │
│ - tabular_preds_*.csv  │ │ - ecg_preds_*.csv     │ │ - cxr_preds_*.csv     │
│ - tabular_embed_*.npy  │ │ - ecg_embed_*.npy     │ │ - cxr_embed_*.npy     │
│ - tabular_ensemble.pkl │ │ - ecg_resnet.pt       │ │ - cxr_densenet.pt     │
└───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ 3. train_fusion.py             │
                      │   - Loads branch embeddings    │
                      │   - Trains GatedFusionModel    │
                      │   - Evaluates fixed_weight vs  │
                      │     learned_gate               │
                      │   - Output: fusion_gate.pt     │
                      │   - Output: fusion_test_preds  │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ 4. evaluate_all.py             │
                      │   - Missingness sweep (7 combos│
                      │   - Decision Curve Analysis    │
                      │   - Fairness subgroup report   │
                      │   - Baseline comparisons       │
                      │   - Heatmaps, DCA plots, etc.  │
                      └───────────────────────────────┘
```

---

## 2. Internal Module Dependency Map

```
src.utils
├── config.py         (omegaconf, pathlib)
├── logger.py         (logging, mlflow)
└── seed.py           (random, numpy, torch)

src.cohort
├── extract_cohort.py -> src.utils.logger
└── split.py          -> src.utils.logger

src.tabular
├── features.py       -> src.utils.logger
├── impute.py         -> src.utils.logger
└── model.py          -> src.utils.logger, src.utils.seed

src.ecg
├── preprocess.py     -> src.utils.logger
├── dataset.py        -> src.ecg.preprocess, src.utils.logger
└── model.py          -> (standalone)

src.cxr
├── preprocess.py     -> src.utils.logger
├── dataset.py        -> src.cxr.preprocess, src.utils.logger
└── model.py          -> (standalone)

src.baselines
├── lace.py           -> src.utils.logger
└── hospital_score.py -> src.utils.logger

src.fusion
├── fixed_weight.py   -> (standalone, numpy only)
└── learned_gate.py   -> src.utils.logger, src.utils.seed

src.evaluation
├── metrics.py        -> sklearn.metrics
├── decision_curve.py -> (standalone matplotlib inside plot fn)
├── fairness.py       -> src.evaluation.metrics, src.utils.logger
└── missingness_sweep.py -> src.evaluation.metrics, src.fusion.fixed_weight, src.utils.logger

src.explainability
├── shap_tabular.py   -> src.utils.logger
└── gradcam_cxr.py    -> src.utils.logger

src.demo
├── case_studies.py   -> (standalone, pure data)
└── app.py            -> src.demo.case_studies, src.baselines.lace, src.baselines.hospital_score

scripts (entry points)
├── run_cohort.py     -> src.cohort.{extract_cohort,split}, src.utils.{config,logger,seed}
├── train_tabular.py  -> src.tabular.{features,impute,model}, src.evaluation.metrics, src.utils.*
├── train_ecg.py      -> src.ecg.{dataset,model,preprocess}, src.evaluation.metrics, src.utils.*
├── train_cxr.py      -> src.cxr.{dataset,model,preprocess}, src.evaluation.metrics, src.utils.*
├── train_fusion.py   -> src.fusion.{learned_gate,fixed_weight}, src.evaluation.metrics, src.utils.*
├── evaluate_all.py   -> src.evaluation.{metrics,missingness_sweep,decision_curve,fairness}, src.fusion.learned_gate, src.utils.*
├── generate_dummy_data.py -> (standalone, uses wfdb, PIL directly)
└── smoke_test.py     -> (subprocess calls to other scripts, shutil)
```

---

## 3. External Library Dependency Graph

```
                            ┌─────────────────────────────────┐
                            │        requirements.txt          │
                            └─────────────────────────────────┘
                                        │
         ┌───────────────────────┬───────┴───────┬───────────────────────┐
         ▼                       ▼               ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌────────────┐   ┌─────────────────┐
│  ML / DL Core   │   │ Data Processing  │   │ Clinical  │   │  Viz & App      │
│                 │   │                  │   │           │   │                 │
│ xgboost         │   │ pandas           │   │ comorbidipy│  │ matplotlib      │
│ scikit-learn    │   │ numpy            │   │ dcurves   │   │ seaborn         │
│ torch           │   │ scipy            │   │           │   │ streamlit       │
│ torchvision     │   │ pyarrow          │   │           │   │                 │
│ timm            │   │ wfdb             │   │           │   │                 │
│                 │   │ pydicom          │   │           │   │                 │
│                 │   │ Pillow           │   │           │   │                 │
└─────────────────┘   └─────────────────┘   └────────────┘   └─────────────────┘
        │                      │                                      │
        └──────────────────────┼──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Infrastructure     │
                    │                     │
                    │  mlflow             │
                    │  omegaconf / pyyaml │
                    │  joblib             │
                    │  tqdm               │
                    │  shap               │
                    │  grad-cam           │
                    └─────────────────────┘
```

## 4. Dependency by Module (External Packages)

| Package / Module | External Dependencies |
|---|---|
| `src.utils.config` | `omegaconf` |
| `src.utils.logger` | `mlflow` (optional), `omegaconf` |
| `src.utils.seed` | `numpy`, `torch` (optional) |
| `src.cohort.extract_cohort` | `pandas`, `omegaconf` |
| `src.cohort.split` | `numpy`, `pandas`, `sklearn`, `omegaconf` |
| `src.tabular.features` | `numpy`, `pandas`, `omegaconf` |
| `src.tabular.impute` | `joblib`, `numpy`, `pandas`, `sklearn` |
| `src.tabular.model` | `numpy`, `pandas`, `xgboost`, `sklearn`, `omegaconf` |
| `src.ecg.preprocess` | `numpy`, `pandas`, `scipy`, `wfdb`, `omegaconf` |
| `src.ecg.dataset` | `numpy`, `pandas`, `torch`, `omegaconf` |
| `src.ecg.model` | `numpy`, `torch`, `omegaconf` |
| `src.cxr.preprocess` | `numpy`, `pandas`, `torch`, `torchvision`, `omegaconf`, `PIL` |
| `src.cxr.dataset` | `numpy`, `pandas`, `torch`, `omegaconf` |
| `src.cxr.model` | `numpy`, `torch`, `omegaconf`, `timm` |
| `src.baselines.lace` | `numpy`, `pandas`, `sklearn`, `omegaconf`, `comorbidipy` (optional) |
| `src.baselines.hospital_score` | `numpy`, `pandas`, `sklearn` |
| `src.fusion.fixed_weight` | `numpy` |
| `src.fusion.learned_gate` | `numpy`, `torch`, `sklearn`, `omegaconf` |
| `src.evaluation.metrics` | `numpy`, `sklearn` |
| `src.evaluation.decision_curve` | `numpy`, `pandas`, `matplotlib` |
| `src.evaluation.fairness` | `numpy`, `pandas`, `matplotlib` |
| `src.evaluation.missingness_sweep` | `numpy`, `pandas`, `torch`, `matplotlib`, `seaborn`, `omegaconf` |
| `src.explainability.shap_tabular` | `shap`, `matplotlib`, `numpy`, `pandas`, `omegaconf` |
| `src.explainability.gradcam_cxr` | `pytorch_grad_cam`, `torch`, `torchvision`, `PIL`, `matplotlib`, `numpy`, `omegaconf` |
| `src.demo.app` | `streamlit`, `numpy`, `matplotlib` |
| `src.demo.case_studies` | (none — pure data structure) |
| `scripts/generate_dummy_data.py` | `numpy`, `pandas`, `wfdb`, `PIL` |
| `scripts/smoke_test.py` | `shutil`, `subprocess` (stdlib only) |

## 5. Key Dependency Rules

- **No circular dependencies** between `src/` sub-packages.
- **All `scripts/` depend on `src/`**, never the reverse.
- **`src/utils/`** is the foundational layer with no internal dependencies other than stdlib + omegaconf.
- **Branch packages** (`tabular`, `ecg`, `cxr`) are independent of each other — they only depend on `utils`.
- **`fusion`** consumes outputs from all 3 branches (at the file level, not import level — it reads `.npy`/`.csv` files).
- **`evaluation`** depends on `fusion` and `metrics` (but not on branch models directly).
- **`explainability`** is standalone, consuming trained models for post-hoc analysis.
- **`demo`** depends on `baselines` for live LACE/HOSPITAL computation in the Streamlit app.
