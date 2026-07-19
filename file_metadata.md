# File Metadata — HealthCare_Analytics

> Per-file metadata: purpose, imports, exports, and key contents. Update when files are added/modified.

---

## Root Files

### `.gitignore`
- **Path**: `./.gitignore`
- **Type**: Git ignore rules
- **Size**: 22 lines
- **Purpose**: Excludes `__pycache__/`, `*.pyc`, `data/`, `outputs/`, `.venv/`, `venv/`, `env/`, `ENV/`, `.streamlit/`, `.DS_Store`, `Thumbs.db`

### `README.md`
- **Path**: `./README.md`
- **Type**: Markdown documentation
- **Purpose**: Project overview — multimodal HF readmission prediction, 3-branch architecture, data sources (MIMIC-IV, MIMIC-IV-ECG, MIMIC-CXR), pipeline description

### `repository_overview.md`
- **Path**: `./repository_overview.md`
- **Type**: Markdown documentation
- **Size**: 137 lines
- **Purpose**: Architectural deep-dive with system diagram, module breakdown, pipeline workflow (Mermaid graph)

### `code_review_and_flaws.md`
- **Path**: `./code_review_and_flaws.md`
- **Type**: Markdown documentation
- **Purpose**: Self-identified code review documenting 4 critical bugs (all fixed), architectural recommendations

### `project_tree.md`
- **Path**: `./project_tree.md`
- **Type**: Markdown tracking file
- **Purpose**: Auto-generated directory tree with annotations and runtime directory notes

### `dependency_graph.md`
- **Path**: `./dependency_graph.md`
- **Type**: Markdown tracking file
- **Purpose**: Pipeline data flow diagram, internal module dependencies, external library graph, per-module dependency table

### `file_metadata.md`
- **Path**: `./file_metadata.md`
- **Type**: Markdown tracking file
- **Purpose**: THIS FILE — per-file metadata: purpose, imports, exports, classes, functions

### `requirements.txt`
- **Path**: `./requirements.txt`
- **Type**: pip requirements
- **Size**: 43 lines, 23 packages
- **Key packages**: `xgboost`, `scikit-learn`, `torch`, `torchvision`, `timm`, `pandas`, `numpy`, `scipy`, `wfdb`, `shap`, `grad-cam`, `comorbidipy`, `dcurves`, `streamlit`, `mlflow`, `omegaconf`

---

## Config Files

### `config/config.yaml`
- **Path**: `config/config.yaml`
- **Type**: YAML (OmegaConf)
- **Size**: 145 lines
- **Purpose**: Full configuration — Google Drive paths, MIMIC locations, all hyperparameters for all 4 models + evaluation + MLflow

### `config/config_dummy.yaml`
- **Path**: `config/config_dummy.yaml`
- **Type**: YAML (OmegaConf)
- **Size**: 126 lines
- **Purpose**: Smoke-test configuration — local paths, reduced epochs/batch sizes, `pretrained=false` for CXR

---

## Scripts (`scripts/`)

### `scripts/run_cohort.py`
- **Path**: `scripts/run_cohort.py`
- **Size**: 53 lines
- **Purpose**: Pipeline step 1 — end-to-end cohort extraction and splitting
- **Imports (internal)**: `src.cohort.extract_cohort.build_cohort`, `src.cohort.split.split_cohort`, `src.utils.config.{load_config,ensure_dirs}`, `src.utils.logger.get_logger`, `src.utils.seed.set_seed`
- **Entry Point**: `main()` — loads config, builds cohort, splits, saves

### `scripts/train_tabular.py`
- **Path**: `scripts/train_tabular.py`
- **Size**: 204 lines
- **Purpose**: Pipeline step 2a — train XGBoost bootstrap ensemble + OOF predictions
- **Imports (internal)**: `src.tabular.features.build_feature_matrix`, `src.tabular.impute.*`, `src.tabular.model.TabularEnsemble`, `src.evaluation.metrics.evaluate_all`, `src.utils.*`
- **Key logic**: 5-fold CV for OOF embeddings; final model on full train set; saves `tabular_preds_*.csv` + `tabular_embed_*.npy`

### `scripts/train_ecg.py`
- **Path**: `scripts/train_ecg.py`
- **Size**: 302 lines
- **Purpose**: Pipeline step 2b — train 1D ResNet ECG branch + MC-Dropout inference
- **Imports (internal)**: `src.ecg.dataset.{ECGDataset,make_ecg_loader}`, `src.ecg.model.{ECGResNet,mc_predict_ecg}`, `src.ecg.preprocess.build_ecg_index`, `src.evaluation.metrics.evaluate_all`, `src.utils.*`
- **Key logic**: 5-fold CV for OOF; final model with pos_weight BCE; saves `ecg_preds_*.csv` + `ecg_embed_*.npy`

### `scripts/train_cxr.py`
- **Path**: `scripts/train_cxr.py`
- **Size**: 301 lines
- **Purpose**: Pipeline step 2c — train DenseNet-121 CXR branch + MC-Dropout inference
- **Imports (internal)**: `src.cxr.dataset.{CXRDataset,make_cxr_loader}`, `src.cxr.model.{CXREncoder,mc_predict_cxr}`, `src.cxr.preprocess.build_cxr_index`, `src.evaluation.metrics.evaluate_all`, `src.utils.*`
- **Key logic**: 5-fold CV for OOF; frozen backbone; saves `cxr_preds_*.csv` + `cxr_embed_*.npy`

### `scripts/train_fusion.py`
- **Path**: `scripts/train_fusion.py`
- **Size**: 187 lines
- **Purpose**: Pipeline step 3 — load branch embeddings, train gated fusion, compare with fixed-weight
- **Imports (internal)**: `src.fusion.learned_gate.{GatedFusionModel,make_fusion_dataset,train_fusion}`, `src.fusion.fixed_weight.fixed_fusion_predict`, `src.evaluation.metrics.evaluate_all`, `src.utils.*`
- **Key logic**: Builds availability masks; trains with modality-dropout; evaluates both methods on test set; saves `fusion_test_preds.csv`

### `scripts/evaluate_all.py`
- **Path**: `scripts/evaluate_all.py`
- **Size**: 169 lines
- **Purpose**: Pipeline step 4 — full evaluation suite
- **Imports (internal)**: `src.evaluation.metrics.*`, `src.evaluation.missingness_sweep.*`, `src.evaluation.decision_curve.*`, `src.evaluation.fairness.*`, `src.fusion.learned_gate.GatedFusionModel`, `src.utils.*`
- **Key outputs**: `missingness_sweep.csv`, `model_comparison.csv`, `dca_results.csv`, `fairness_report.csv` + PNG figures

### `scripts/generate_dummy_data.py`
- **Path**: `scripts/generate_dummy_data.py`
- **Size**: 183 lines
- **Purpose**: Generate synthetic MIMIC data for smoke testing
- **Imports**: `numpy`, `pandas`, `wfdb`, `PIL.Image`
- **Key outputs**: MIMIC-IV CSV tables (10 patients), WFDB ECG records, JPEG CXR images

### `scripts/smoke_test.py`
- **Path**: `scripts/smoke_test.py`
- **Size**: 90 lines
- **Purpose**: End-to-end pipeline smoke test on CPU (<60s)
- **Imports**: stdlib only (`os`, `shutil`, `subprocess`, `sys`, `pathlib`)
- **Key logic**: Swaps config, runs all 8 scripts sequentially, restores config

---

## Source Package (`src/`)

### `src/__init__.py`
- **Path**: `src/__init__.py`
- **Size**: 1 line
- **Purpose**: Package marker
- **Content**: `# Multimodal HF Readmission Prediction — src package`

---

### `src/utils/` — Utilities

#### `src/utils/__init__.py`
- **Size**: 1 line — `# utils package`

#### `src/utils/config.py`
- **Size**: 76 lines
- **Purpose**: OmegaConf-based YAML config loader
- **Exports**:
  - `load_config(config_path=None, overrides=None) -> DictConfig`
  - `ensure_dirs(cfg) -> None`
- **Key imports**: `omegaconf.{DictConfig,OmegaConf}`, `pathlib.Path`

#### `src/utils/logger.py`
- **Size**: 77 lines
- **Purpose**: Logging setup with optional MLflow integration
- **Exports**:
  - `get_logger(name, level=INFO) -> logging.Logger`
  - `init_mlflow(cfg, run_name=None) -> str` (run_id)
- **Key imports**: `logging`, `omegaconf.DictConfig`, `mlflow` (optional)

#### `src/utils/seed.py`
- **Size**: 47 lines
- **Purpose**: Global reproducibility seed setter
- **Exports**:
  - `set_seed(seed=42) -> None`
- **Key imports**: `os`, `random`, `numpy`, `torch` (optional)

---

### `src/cohort/` — Cohort Extraction

#### `src/cohort/__init__.py`
- **Size**: 1 line — `# cohort package`

#### `src/cohort/extract_cohort.py`
- **Size**: 350 lines
- **Purpose**: Extract HF cohort from MIMIC-IV, label 30-day readmission, compute prior visits
- **Exports**:
  - `build_cohort(cfg) -> pd.DataFrame`
  - `load_mimic_tables(mimic_iv_dir) -> dict[str, pd.DataFrame]`
  - `extract_hf_admissions(admissions, patients, diagnoses_icd, ...) -> pd.DataFrame`
  - `label_readmission(cohort, admissions, readmission_days=30, ...) -> pd.DataFrame`
  - `attach_prior_visit_counts(cohort, admissions) -> pd.DataFrame`
- **Key imports**: `pandas`, `omegaconf.DictConfig`, `src.utils.logger`
- **Key logic**: ICD-10 "I50" + ICD-9 HF code filtering; excludes in-hospital deaths; computes readmission + competing event labels; uses raw admissions for prior visit counts

#### `src/cohort/split.py`
- **Size**: 136 lines
- **Purpose**: Subject-level stratified train/val/test split
- **Exports**:
  - `split_cohort(cohort, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
- **Key imports**: `numpy`, `pandas`, `sklearn.model_selection.train_test_split`, `omegaconf.DictConfig`, `src.utils.logger`
- **Key logic**: Groups by `subject_id`; stratified split using patient-level label; safeguards against class imbalance

---

### `src/tabular/` — Tabular EHR Branch

#### `src/tabular/__init__.py`
- **Size**: 1 line — `# tabular package`

#### `src/tabular/features.py`
- **Size**: 316 lines
- **Purpose**: Extract tabular feature matrix (demographics, labs, vitals, admin)
- **Exports**:
  - `build_feature_matrix(cohort, cfg, labevents=None, chartevents=None) -> Tuple[pd.DataFrame, pd.Series, List[str]]`
  - `extract_lab_features(cohort, labevents, lab_itemids, window_hours=72) -> pd.DataFrame`
  - `extract_vital_features(cohort, chartevents, vital_itemids, window_hours=48) -> pd.DataFrame`
  - `extract_demographic_features(cohort) -> pd.DataFrame`
  - `extract_admin_features(cohort) -> pd.DataFrame`
- **Key imports**: `numpy`, `pandas`, `omegaconf.DictConfig`, `src.utils.logger`
- **Key logic**: Chunked CSV reading for large tables; last-72h labs closest to discharge; last-48h vitals mean/min/max

#### `src/tabular/impute.py`
- **Size**: 114 lines
- **Purpose**: Median imputation pipeline
- **Exports**:
  - `fit_imputer(X_train) -> SimpleImputer`
  - `apply_imputer(imputer, X) -> pd.DataFrame`
  - `save_imputer(imputer, path) -> None`
  - `load_imputer(path) -> SimpleImputer`
  - `missingness_report(X) -> pd.DataFrame`
- **Key imports**: `joblib`, `numpy`, `pandas`, `sklearn.impute.SimpleImputer`, `src.utils.logger`

#### `src/tabular/model.py`
- **Size**: 217 lines
- **Purpose**: XGBoost bootstrap ensemble
- **Exports**:
  - `TabularEnsemble` (class)
    - `__init__(cfg)`
    - `fit(X_train, y_train, X_val, y_val) -> TabularEnsemble`
    - `predict(X) -> Dict[str, np.ndarray]` (score, std, confidence, all_preds)
    - `save(path)`, `load(path)` (classmethod)
    - `get_feature_importance() -> pd.DataFrame`
- **Key imports**: `pickle`, `numpy`, `pandas`, `xgboost`, `sklearn.utils.resample`, `omegaconf.DictConfig`, `src.utils.{logger,seed}`
- **Key logic**: N=20 bootstrap; early stopping; static `global_max_std` from validation set for confidence normalization

---

### `src/ecg/` — 12-Lead ECG Branch

#### `src/ecg/__init__.py`
- **Size**: 1 line — `# ecg package`

#### `src/ecg/preprocess.py`
- **Size**: 187 lines
- **Purpose**: ECG preprocessing pipeline (WFDB loading, filtering, resampling)
- **Exports**:
  - `build_ecg_index(cohort, cfg) -> pd.DataFrame`
  - `load_and_preprocess_ecg(record_path, cfg) -> Optional[np.ndarray]` (12, 5000)
- **Key imports**: `numpy`, `pandas`, `scipy.signal.{butter,filtfilt}`, `wfdb` (lazy), `omegaconf.DictConfig`, `src.utils.logger`
- **Key logic**: 0.5–40 Hz bandpass filter; resample to 500 Hz; pad/trim to 5000 samples; per-lead Z-score normalization

#### `src/ecg/dataset.py`
- **Size**: 134 lines
- **Purpose**: PyTorch Dataset for ECG waveforms
- **Exports**:
  - `ECGDataset` (class — Dataset)
    - `__getitem__(idx) -> Dict` (waveform, available, label, hadm_id)
  - `make_ecg_loader(dataset, batch_size=64, shuffle=False, num_workers=0) -> DataLoader`
- **Key imports**: `numpy`, `pandas`, `torch`, `omegaconf.DictConfig`, `torch.utils.data.*`, `src.ecg.preprocess.load_and_preprocess_ecg`, `src.utils.logger`
- **Key logic**: Missing ECGs return zero tensor with `available=0`; optional caching

#### `src/ecg/model.py`
- **Size**: 208 lines
- **Purpose**: 1D ResNet for ECG classification + MC-Dropout
- **Exports**:
  - `ECGResNet` (class — nn.Module)
    - `forward(x) -> Tuple[logit, embed]`
  - `mc_predict_ecg(model, waveform, n_passes=50, ...) -> Dict` (prob, std, confidence, embed)
- **Key imports**: `numpy`, `torch`, `torch.nn`, `omegaconf.DictConfig`
- **Key architecture**: Stem conv → 4× ResBlock1D (64→128→256→embed_dim) → pool → dropout → head

---

### `src/cxr/` — Chest X-Ray Branch

#### `src/cxr/__init__.py`
- **Size**: 1 line — `# cxr package`

#### `src/cxr/preprocess.py`
- **Size**: 196 lines
- **Purpose**: CXR preprocessing (JPEG loading, indexing, augmentation)
- **Exports**:
  - `get_transform(is_train, image_size=224) -> transforms.Compose`
  - `build_cxr_index(cohort, cfg) -> pd.DataFrame`
  - `load_and_preprocess_cxr(image_path, is_train=False, image_size=224) -> Optional[torch.Tensor]`
- **Key imports**: `numpy`, `pandas`, `torch`, `torchvision.transforms`, `PIL.Image`, `omegaconf.DictConfig`, `src.utils.logger`
- **Key logic**: Frontal-only (PA/AP); closest to discharge within 72h window; ImageNet normalization; random augmentation for training

#### `src/cxr/dataset.py`
- **Size**: 121 lines
- **Purpose**: PyTorch Dataset for CXR images
- **Exports**:
  - `CXRDataset` (class — Dataset)
    - `__getitem__(idx) -> Dict` (image, available, label, hadm_id)
  - `make_cxr_loader(dataset, batch_size=32, shuffle=False, num_workers=2) -> DataLoader`
- **Key imports**: `numpy`, `pandas`, `torch`, `omegaconf.DictConfig`, `torch.utils.data.*`, `src.cxr.preprocess.load_and_preprocess_cxr`, `src.utils.logger`

#### `src/cxr/model.py`
- **Size**: 169 lines
- **Purpose**: DenseNet-121 transfer learning encoder + MC-Dropout
- **Exports**:
  - `CXREncoder` (class — nn.Module)
    - `forward(x) -> Tuple[logit, embed]`
    - `unfreeze_backbone() -> None`
  - `mc_predict_cxr(model, images, n_passes=50, ...) -> Dict` (prob, std, confidence, embed)
- **Key imports**: `numpy`, `torch`, `torch.nn`, `timm` (lazy), `omegaconf.DictConfig`
- **Key architecture**: DenseNet-121 backbone (frozen) → linear proj (1024→256) → ReLU → Dropout → head

---

### `src/baselines/` — Clinical Baselines

#### `src/baselines/__init__.py`
- **Size**: 1 line — `# baselines package`

#### `src/baselines/lace.py`
- **Size**: 247 lines
- **Purpose**: LACE index computation + probability calibration
- **Exports**:
  - `compute_lace_scores(cohort, diagnoses_icd, y_train=None, lace_train=None) -> pd.DataFrame`
  - `compute_charlson_index(hadm_ids, diagnoses_icd) -> pd.Series`
  - `lace_l(los_days) -> int`
  - `lace_a(via_ed) -> int`
  - `lace_c(charlson_index) -> int`
  - `lace_e(ed_visits_6m) -> int`
- **Key imports**: `numpy`, `pandas`, `sklearn.linear_model.LogisticRegression`, `comorbidipy` (optional), `omegaconf.DictConfig`, `src.utils.logger`
- **Key logic**: Charlson via `comorbidipy` or simplified ICD-10 prefix mapping; LR calibration

#### `src/baselines/hospital_score.py`
- **Size**: 204 lines
- **Purpose**: HOSPITAL score computation + probability calibration
- **Exports**:
  - `compute_hospital_scores(cohort, labevents, y_train=None, hosp_train=None) -> pd.DataFrame`
  - `hospital_h(hgb)`, `hospital_o(service)`, `hospital_s(sodium)`, `hospital_p(has_procedure)`, `hospital_i(via_ed)`, `hospital_t(prior_admits_12m)`, `hospital_a(los_days)`
- **Key imports**: `numpy`, `pandas`, `sklearn.linear_model.LogisticRegression`, `src.utils.logger`

---

### `src/fusion/` — Modality Fusion

#### `src/fusion/__init__.py`
- **Size**: 1 line — `# fusion package`

#### `src/fusion/fixed_weight.py`
- **Size**: 105 lines
- **Purpose**: Confidence-weighted average fusion baseline
- **Exports**:
  - `confidence_weighted_fusion(scores, confidences, available) -> np.ndarray`
  - `fixed_fusion_predict(branch_results, avail_flags) -> Dict` (score, method)
- **Key imports**: `numpy`

#### `src/fusion/learned_gate.py`
- **Size**: 374 lines
- **Purpose**: Learned gating fusion with modality-dropout training
- **Exports**:
  - `GatedFusionModel` (class — nn.Module)
    - `forward(tab_embed, ecg_embed, cxr_embed, tab_conf, ecg_conf, cxr_conf, availability) -> Tuple[logit, weights]`
  - `apply_modality_dropout(...) -> augmented tensors`
  - `train_fusion(model, train_loader, val_loader, cfg, save_path=None) -> Dict` (history)
  - `make_fusion_dataset(...) -> TensorDataset`
- **Key imports**: `numpy`, `torch`, `torch.nn`, `torch.optim`, `sklearn.metrics.roc_auc_score`, `omegaconf.DictConfig`, `src.utils.{logger,seed}`
- **Key architecture**: Per-modality projectors → Gate MLP (d_fuse×3+6 → 64 → 3) → softmax mask → weighted sum → classifier → logit
- **Key logic**: Modality-dropout randomly drops ECG/CXR (p=0.3) during training; availability masking to -inf before softmax

---

### `src/evaluation/` — Evaluation Suite

#### `src/evaluation/__init__.py`
- **Size**: 1 line — `# evaluation package`

#### `src/evaluation/metrics.py`
- **Size**: 145 lines
- **Purpose**: Core classification and calibration metrics
- **Exports**:
  - `auroc(y_true, y_prob) -> float`
  - `auprc(y_true, y_prob) -> float`
  - `brier(y_true, y_prob) -> float`
  - `expected_calibration_error(y_true, y_prob, n_bins=10) -> float`
  - `reliability_diagram_data(y_true, y_prob, n_bins=10) -> Dict`
  - `evaluate_all(y_true, y_prob, n_bins=10, prefix="") -> Dict`
- **Key imports**: `numpy`, `sklearn.metrics.{roc_auc_score,average_precision_score,brier_score_loss}`

#### `src/evaluation/decision_curve.py`
- **Size**: 172 lines
- **Purpose**: Decision Curve Analysis (net benefit vs threshold)
- **Exports**:
  - `net_benefit(y_true, y_prob, threshold) -> float`
  - `treat_all_net_benefit(y_true, threshold) -> float`
  - `run_dca(y_true, predictions, thresh_min=0.0, thresh_max=0.5, thresh_step=0.01) -> pd.DataFrame`
  - `plot_dca(dca_df, title, save_path, y_limits) -> None`
- **Key imports**: `numpy`, `pandas`, `matplotlib.pyplot` (lazy)

#### `src/evaluation/fairness.py`
- **Size**: 175 lines
- **Purpose**: Subgroup fairness analysis (AUROC gap, ECE)
- **Exports**:
  - `fairness_report(cohort_test, y_prob, groups=None, min_group_size=30) -> pd.DataFrame`
  - `plot_fairness_report(report_df, save_path) -> None`
- **Key imports**: `numpy`, `pandas`, `matplotlib.pyplot`, `src.evaluation.metrics.{auroc,expected_calibration_error}`, `src.utils.logger`

#### `src/evaluation/missingness_sweep.py`
- **Size**: 266 lines
- **Purpose**: Core experiment — 7 modality combinations × 2 fusion methods × 4 metrics
- **Exports**:
  - `run_missingness_sweep(branch_outputs, y_true, gate_model=None, cfg=None) -> pd.DataFrame`
  - `plot_missingness_sweep(df, save_path) -> None`
  - `_all_subsets(modalities) -> list` (private helper)
  - `_make_availability_mask(subset, n) -> dict` (private helper)
  - `_gated_fusion_predict(gate_model, branch_outputs, availability_mask) -> np.ndarray` (private helper)
- **Key imports**: `itertools`, `numpy`, `pandas`, `torch`, `matplotlib`, `seaborn`, `omegaconf.DictConfig`, `src.evaluation.metrics.evaluate_all`, `src.fusion.fixed_weight.confidence_weighted_fusion`, `src.utils.logger`

---

### `src/explainability/` — Model Explainability

#### `src/explainability/__init__.py`
- **Size**: 1 line — `# explainability package`

#### `src/explainability/shap_tabular.py`
- **Size**: 160 lines
- **Purpose**: SHAP explanation for XGBoost tabular branch
- **Exports**:
  - `explain_tabular(ensemble, X_test, y_test, cfg, n_samples_global=500, patient_hadm_ids=None, save_dir=None) -> Dict`
- **Key imports**: `shap`, `matplotlib.pyplot`, `numpy`, `pandas`, `omegaconf.DictConfig`, `src.utils.logger`
- **Key outputs**: Beeswarm, bar, waterfall plots; SHAP values CSV

#### `src/explainability/gradcam_cxr.py`
- **Size**: 196 lines
- **Purpose**: Grad-CAM heatmaps for CXR branch
- **Exports**:
  - `generate_gradcam(cxr_model, image_paths, hadm_ids, cfg, save_dir=None, top_k=4) -> Dict[int, np.ndarray]`
- **Key imports**: `pytorch_grad_cam.{GradCAM,show_cam_on_image,ClassifierOutputTarget}`, `PIL.Image`, `matplotlib.pyplot`, `torchvision.transforms`, `torch`, `numpy`, `omegaconf.DictConfig`, `src.utils.logger`
- **Key logic**: Targets last conv of DenseNet-121 denseblock4; wraps model for GradCAM compatibility

---

### `src/demo/` — Streamlit Demo

#### `src/demo/__init__.py`
- **Size**: 1 line — `# demo package`

#### `src/demo/case_studies.py`
- **Size**: 178 lines
- **Purpose**: 4 pre-loaded case studies for demo walkthroughs
- **Exports**:
  - `CASE_STUDIES` (list of dict — 4 cases)
  - `get_case_study_display(case) -> Dict` (profile, scores, interpretation)
- **Key imports**: none (pure data structure)
- **Data**: 4 cases — full high risk, missing ECG moderate, missing CXR low, conflicting modalities

#### `src/demo/app.py`
- **Size**: 539 lines
- **Purpose**: Main Streamlit dashboard
- **Key imports**: `streamlit`, `numpy`, `matplotlib.pyplot`, `src.demo.case_studies.{CASE_STUDIES,get_case_study_display}`, `src.baselines.lace.*`, `src.baselines.hospital_score.*`
- **Features**: Patient input form, fused risk score display, branch breakdown bars, modality toggle, missingness impact bar chart, SHAP/Grad-CAM tabs, case study walkthrough

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Python files | 44 |
| Total lines of Python | ~7,196 |
| Config files | 2 |
| Markdown files | 5 (including these 3) |
| Requirements entries | 23 packages |
| Classes defined | 7 (`TabularEnsemble`, `ECGResNet`, `ResBlock1D`, `CXREncoder`, `GatedFusionModel`, `ECGDataset`, `CXRDataset`) |
| Public functions | ~55 |
| Pipeline scripts | 6 (numbered) |
| Supporting scripts | 2 (data generation, smoke test) |
