# Code Issues, Hardcoded Values & Partially Implemented Logic

> Tracking file documenting hardcoded values, incorrect/partial implementations, dead code, and design smells. Update as issues are resolved.

---

## 1. Hardcoded Values (Should Be Config-Driven)

### Filter Parameters in ECG Preprocessing
- **File**: `src/ecg/preprocess.py:162`
- **Issue**: Butterworth filter order (3) and cutoff frequencies (0.5, 40.0 Hz) are hardcoded
- **Impact**: Cannot be tuned without code changes
- **Fix**: Move to `config.yaml` under `ecg.filter_order`, `ecg.filter_low`, `ecg.filter_high`

### Dropout Rate Inside Gate MLP
- **File**: `src/fusion/learned_gate.py:88`
- **Issue**: `nn.Dropout(0.2)` is hardcoded inside the gate, while the classifier uses `cfg.fusion.dropout` (0.4) and modality-dropout uses `cfg.fusion.modality_drop_p` (0.3)
- **Impact**: Three different dropout rates for the same fusion module — confusing and non-configurable
- **Fix**: Read from config as `cfg.fusion.gate_dropout`

### ICD Prefix Mapping for Simplified Charlson
- **File**: `src/baselines/lace.py:119-140`
- **Issue**: ICD-10 prefix lists and their weights are hardcoded
- **Impact**: Cannot be extended without code changes; duplicates `comorbidipy` with lower accuracy
- **Fix**: Consider marking as deprecated or moving to config

### Chest X-Ray Window in CXR Preprocessing
- **File**: `src/cxr/preprocess.py:100`
- **Issue**: `window_h = cfg.cohort.cxr_window_hours` — the window is read from config but the logic assumes a single window for all patients
- **Impact**: Reasonable but worth noting

### Admission Type Hardcoding for ED Flag
- **File**: `src/cohort/extract_cohort.py:141-142`
- **Issue**: `{"EMERGENCY", "URGENT", "EW EMER."}` hardcoded as ED admission types
- **Impact**: MIMIC-specific; would fail on different EHR systems
- **Fix**: Move to config

### ICU Chartevents Item IDs
- **File**: `src/tabular/features.py` (via config but default itemids in `config.yaml` are MIMIC-IV specific)
- **Issue**: While these are in config, they are MIMIC-IV hardcoded defaults

### XGBoost Device Hardcoded
- **File**: `src/tabular/model.py:76`
- **Issue**: `device = "cpu"` hardcoded in `_make_base_model`
- **Impact**: Cannot use GPU even when available
- **Fix**: Read from config `tabular.xgb.device`

### CXR Num Workers Hardcoded
- **File**: `src/cxr/dataset.py:99` (default parameter `num_workers=2`), but `scripts/train_cxr.py:133-135` always passes `num_workers=0`
- **Impact**: Parameter is effectively dead — always 0 in practice
- **Fix**: Read from config

### ECG Num Workers Hardcoded
- **File**: `src/ecg/dataset.py:110` (default parameter `num_workers=0`)
- **Impact**: Intentionally 0 for Colab compatibility but should be configurable

### Cohort Directory Keys Hardcoded
- **File**: `src/utils/config.py:66-76`
- **Issue**: List of path keys (`cohort_dir`, `processed_dir`, etc.) hardcoded in `ensure_dirs`
- **Impact**: Adding new output directories requires code change
- **Fix**: Derive from config keys dynamically

### Demo Fallback Risk Threshold (0.5)
- **File**: `src/demo/app.py:145`
- **Issue**: `return 0.5` when all modalities missing (division by zero guard)
- **Impact**: Silent fallback to mid-risk instead of warning the user

### LACE/HOSPITAL Simplified Fallback in Demo
- **File**: `src/demo/app.py:335-353`
- **Issue**: Demo reimplements simplified LACE/HOSPITAL computation duplicating `src/baselines/` logic
- **Impact**: Two versions of the same logic — maintenance burden, likely to drift
- **Fix**: Reuse the baseline module functions directly

### ECG CV Fold Epochs Hardcoded
- **File**: `scripts/train_ecg.py:167` and `scripts/train_cxr.py:168`
- **Issue**: `for epoch in range(1, 6):` — 5 epochs hardcoded for CV fold training
- **Impact**: Not configurable
- **Fix**: Read from config or infer from `cfg.ecg.epochs`

### MC-Passes in CV Hardcoded
- **File**: `scripts/train_ecg.py:176` and `scripts/train_cxr.py:178`
- **Issue**: `n_passes=10` hardcoded for CV (full inference uses `cfg.ecg.mc_passes` = 50)
- **Impact**: Config parameter not used consistently

---

## 2. Incorrect / Bug-Adjacent

### Dead Code in Fairness Report
- **File**: `src/evaluation/fairness.py:83-91`
- **Issue**: Lines 83-86 attempt to index using `get_indexer` but the result is never used. Lines 89-91 recompute the mask correctly. The entire `if-else` block (lines 82-87) is unreachable dead code.
- **Fix**: Remove the dead block

### `global_max_std` Fragile Fallback
- **File**: `src/tabular/model.py:177`, `src/ecg/model.py:204`, `src/cxr/model.py:162`
- **Issue**: `global_max_std` falls back to 0.5 via `getattr(self, "global_max_std", 0.5)` — this is used for N=1 deployment but is a theoretical maximum, not dataset-specific
- **Impact**: Confidence calibration may be poor if the actual max std deviates from 0.5
- **Fix**: Ensure `global_max_std` is always computed during training; warn if using default

### ECG Preprocessing Load Order
- **File**: `src/ecg/preprocess.py:68-76`
- **Issue**: Tries `record_list.csv`, then `machine_measurements.csv` as fallback — but these have different schema
- **Impact**: If `machine_measurements.csv` is loaded, the `filename` column may not exist, causing `build_path` (line 108) to fail with `str(row.get("filename", ""))` returning empty strings
- **Fix**: Validate column presence after loading

### `roc_auc_score` Imported Inside Functions
- **File**: `scripts/train_ecg.py:100`, `scripts/train_cxr.py:96`, `src/fusion/learned_gate.py:305`
- **Issue**: `from sklearn.metrics import roc_auc_score` inside function bodies
- **Impact**: Violates PEP8; minor performance overhead on every eval call

### Charitevents/Labevents CSV Loading Assumes Gzip
- **File**: `src/tabular/features.py:257-259`, `scripts/train_tabular.py:67-68`
- **Issue**: Code checks for `.csv.gz` first, falls back to `.csv` — but if neither exists, the error is a generic `FileNotFoundError` after the fact
- **Impact**: User gets a confusing error traceback

### OOF Embeddings Impute NaN with 0.0
- **File**: `scripts/train_tabular.py:188`
- **Issue**: `np.nan_to_num(data["embed"], nan=0.0)` replaces missing values with 0 before passing to fusion
- **Impact**: Zero imputation may introduce bias; missingness is lost as a signal
- **Fix**: Consider passing a missingness mask alongside the imputed values

### WFDB Write Format Mismatch
- **File**: `scripts/generate_dummy_data.py:133`
- **Issue**: `fmt=["16"] * 12` writes 16-bit integers, but `p_signal` is `np.float32`
- **Impact**: WFDB will truncate/cast the float32 signal to 16-bit int, losing precision. This is fine for dummy data but worth noting.

### XGBoost Early Stopping on Fixed Validation Set
- **File**: `src/tabular/model.py:132`
- **Issue**: All bootstrap models use the same validation set for early stopping
- **Impact**: Early stopping may overfit to the same validation split across bootstraps

---

## 3. Partially Implemented

### Modality-Dropout Only Drops ECG and CXR
- **File**: `src/fusion/learned_gate.py:154-155`
- **Issue**: Tabular is never dropped. The comment says "Tabular is NEVER dropped (always available assumption)" — this is a design choice, but the fusion layer never learns to handle missing tabular data.
- **Impact**: If tabular data is ever missing in production, the gate has no training for that scenario

### Missingness Sweep Fixed-Weight Only When No Gate Model
- **File**: `src/evaluation/missingness_sweep.py:192`
- **Issue**: `if gate_model is not None:` — the sweep gracefully degrades to fixed-weight only, but the function signature requires `cfg` only for this case, suggesting partial implementation

### MLflow Integration is Optional and Silent
- **File**: `src/utils/logger.py:66-77`
- **Issue**: `import mlflow` is inside a try/except. If MLflow is not installed, it silently returns `"no_mlflow"` run ID
- **Impact**: User may think MLflow logging is active when it isn't
- **Fix**: Log a warning (already done on line 76) — this is actually fine but worth tracking

### Config Override via CLI Args Not Fully Implemented
- **File**: `src/utils/config.py:50-52`
- **Issue**: `load_config` accepts an `overrides` dict parameter, but no CLI argument parsing exists in the scripts — overrides can only be passed programmatically
- **Impact**: Users cannot easily override config from command line

### Comorbidipy Is Optional but Incomplete Without It
- **File**: `src/baselines/lace.py:87-107`
- **Issue**: `comorbidipy` is optional — falls back to `_simplified_charlson` which only handles ICD-10 codes
- **Impact**: ICD-9 diagnoses (which exist in MIMIC-IV) are ignored in the fallback path

### Demo Uses Simulated Scores, Not Real Inference
- **File**: `src/demo/app.py` (entire file)
- **Issue**: The demo uses slider overrides for branch scores rather than running actual model inference
- **Impact**: Demo is a simulation/prototype, not a functional inference interface

### CXR Index File Saved but Never Used
- **File**: `scripts/train_cxr.py:125`
- **Issue**: `cxr_index.to_csv(...)` saves the index but no other script reads it

---

## 4. Design Smells & Questionable Patterns

### `sys.path.insert(0, ...)` in Every Script
- **Files**: All 6 pipeline scripts + `src/demo/app.py`
- **Issue**: Every entry point inserts the project root into `sys.path` to allow `import src.*`
- **Impact**: Fragile; breaks if script is run from a different working directory
- **Fix**: Install as a proper package (`pip install -e .`) or use `PYTHONPATH`

### Duplicate `_enable_dropout` Functions
- **Files**: `src/ecg/model.py:138-142`, `src/cxr/model.py:99-103`
- **Issue**: Identical function defined in two places
- **Fix**: Move to `src/utils/`

### Duplicate `train_one_epoch` / `eval_epoch` Patterns
- **Files**: `scripts/train_ecg.py:40-69`, `scripts/train_cxr.py:40-66`
- **Issue**: Nearly identical training loops duplicated across ECG and CXR scripts
- **Impact**: Maintenance burden
- **Fix**: Generalize into `src/utils/training.py`

### Config Path Construction Uses Raw String Concatenation
- **File**: `src/cohort/extract_cohort.py:59-64`, `src/tabular/features.py:249-250`
- **Issue**: Paths built by concatenating `cfg.paths.mimic_iv_dir` with `/hosp`, `/icu` etc.
- **Impact**: Cross-platform path separator issues on Windows
- **Fix**: Use `Path(cfg.paths.mimic_iv_dir) / "hosp"`

### Subject Prefix Logic Duplicated
- **File**: `src/cxr/preprocess.py:152-156` and `src/ecg/preprocess.py:105-109`
- **Issue**: MIMIC subject/study path construction logic duplicated
- **Impact**: If MIMIC directory structure changes, both must be updated
- **Fix**: Extract into `src/utils/mimic.py`
