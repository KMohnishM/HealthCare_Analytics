import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🫀 Multimodal Heart Failure Readmission Prediction Pipeline\n",
                "**Master Kaggle Execution Notebook (Robust Dataset & Pip Package Setup)**\n\n",
                "This notebook executes the entire pipeline sequentially:\n\n",
                "1. **Install Required Python Packages** (`timm`, `wfdb`, `xgboost`, `shap`, `dcurves`)\n",
                "2. **Setup & Clone Repository**\n",
                "3. **Copy All Datasets from Kaggle Input** (Parquets, ECGs, CXRs)\n",
                "4. **Fast Resume Download Check** (Skips automatically if uploaded dataset contains raw files)\n",
                "5. **Train Tabular Branch** (XGBoost Bootstrap Ensemble)\n",
                "6. **Train ECG Branch** (1D ResNet-34 with MC-Dropout)\n",
                "7. **Train CXR Branch** (DenseNet-121 with Transfer Learning)\n",
                "8. **Train Gated Fusion Layer** (Masked Softmax MLP Gating)\n",
                "9. **Run Comprehensive Evaluations** (DCA, Baselines, Confusion Matrices, Fairness)\n",
                "10. **Generate & Display Inline Dashboard**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 1: Install Required Python Packages ────────────────────────────\n",
                "!pip install -q timm wfdb xgboost shap dcurves pyarrow"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 2: Environment Setup & Clone ──────────────────────────────────\n",
                "import os, shutil\n\n",
                "if not os.path.exists(\"HealthCare_Analytics\"):\n",
                "    !git clone https://github.com/KMohnishM/HealthCare_Analytics.git\n\n",
                "%cd HealthCare_Analytics\n",
                "!git pull"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 3: Copy All Datasets from Kaggle Input ─────────────────────────\n",
                "import os\n",
                "os.makedirs(\"data\", exist_ok=True)\n",
                "print(\"Copying teammate's parquet splits, feature matrices, and uploaded raw ECG/CXR files...\")\n",
                "!find /kaggle/input/ -name \"*.parquet\" -exec cp {} data/ \\; 2>/dev/null || true\n",
                "!find /kaggle/input/ -name \"*.csv\" -exec cp {} data/ \\; 2>/dev/null || true\n",
                "!mkdir -p data/raw\n",
                "!cp -r /kaggle/input/**/raw/* data/raw/ 2>/dev/null || cp -r /kaggle/input/*/raw/* data/raw/ 2>/dev/null || true\n",
                "print(\"\\n--- Data Folder Contents ---\")\n",
                "!ls -la data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 4: Fast Resume Download Check ─────────────────────────────────\n",
                "# If raw files are already mounted from your uploaded Kaggle Dataset, this skips in 1 second!\n",
                "!python scripts/download_cohort_physionet.py --cohort data/cohort.parquet --username kmohnishm --password HereisMy2006Bye"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 5: Train Tabular Branch (XGBoost Ensemble) ───────────────────\n",
                "# Loads pre-computed X_train.parquet, X_val.parquet, X_test.parquet directly\n",
                "!python scripts/train_tabular.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 6: Train ECG Branch (1D ResNet-34) ───────────────────────────\n",
                "!python scripts/train_ecg.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 7: Train CXR Branch (DenseNet-121) ───────────────────────────\n",
                "!python scripts/train_cxr.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 8: Train Gated Fusion Model (MLP Gating Head) ─────────────────\n",
                "!python scripts/train_fusion.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 9: Comprehensive Evaluation (DCA, Baselines, Confusion Matrix) ──\n",
                "!python scripts/evaluate_all.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 10: Generate Interactive Dashboard Notebook ───────────────────\n",
                "!python scripts/generate_notebook.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 11: Display All Visual Results Inline in Kaggle ───────────────\n",
                "import os\n",
                "from IPython.display import Image, display, HTML\n\n",
                "figures = [\n",
                "    (\"confusion_matrix.png\", \"Side-by-Side Confusion Matrices (F1-Optimized Thresholds)\"),\n",
                "    (\"decision_curve.png\", \"Clinical Decision Curve Analysis (DCA vs. LACE / HOSPITAL)\"),\n",
                "    (\"missingness_sweep_heatmap.png\", \"Modality Missingness Sweep Heatmap\"),\n",
                "    (\"fairness_subgroups.png\", \"Algorithmic Fairness Subgroup Analysis\")\n",
                "]\n\n",
                "for filename, title in figures:\n",
                "    filepath = os.path.join(\"outputs\", \"figures\", filename)\n",
                "    if os.path.exists(filepath):\n",
                "        display(HTML(f\"\"\"<h3 style='color:#2c3e50; font-family:sans-serif; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px;'>\n",
                "                        {title} (<code>{filename}</code>)\n",
                "                      </h3>\"\"\"))\n",
                "        display(Image(filename=filepath, width=750))\n",
                "    else:\n",
                "        print(f\"Warning: Figure {filename} not found at {filepath}\")"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("master_kaggle_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Successfully created master_kaggle_pipeline.ipynb with pip install & dataset mounting!")
