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
                "1. **Install Required Python Packages** (`timm`, `wfdb`, `xgboost`, `torchxrayvision`, `shap`, `dcurves`)\n",
                "2. **Setup & Clone Repository**\n",
                "3. **Copy All Datasets from Kaggle Input** (Parquets, ECGs, CXRs)\n",
                "4. **Fast Resume Download Check** (Skips automatically if uploaded dataset contains raw files)\n",
                "5. **Train Tabular Branch** (XGBoost Ensemble with Trajectories & Ratios)\n",
                "6. **Train ECG Branch** (1D ResNet-34 with Lead Attention)\n",
                "7. **Train CXR Branch** (DenseNet-121 with Medical Pretraining)\n",
                "8. **Train Gated Fusion Layer** (Focal Loss & Gated MLP)\n",
                "9. **Run Comprehensive Evaluations** (DCA, Baselines, Confusion Matrices, Fairness)\n",
                "10. **Generate Interactive Dashboard Notebook**\n",
                "11. **Display Inline Visual Dashboard**\n",
                "12. **Auto-Push ALL Files (`git add .`) to New Versioned Branch on GitHub**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 1: Install Required Python Packages ────────────────────────────\n",
                "!pip install -q timm wfdb xgboost shap dcurves pyarrow torchxrayvision"
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
                "import os, glob, shutil\n",
                "os.makedirs(\"data\", exist_ok=True)\n",
                "os.makedirs(\"data/raw\", exist_ok=True)\n",
                "print(\"Copying missing parquet splits and feature matrices from Kaggle Input...\")\n",
                "for src in glob.glob(\"/kaggle/input/**/*.parquet\", recursive=True):\n",
                "    fname = os.path.basename(src)\n",
                "    dest = os.path.join(\"data\", fname)\n",
                "    if not os.path.exists(dest):\n",
                "        print(f\"  Copying {fname}...\")\n",
                "        shutil.copy(src, dest)\n",
                "!find /kaggle/input/ -type d -name \"mimic-iv-ecg-1.0\" -exec cp -r {} data/raw/ \\; 2>/dev/null || true\n",
                "!find /kaggle/input/ -type d -name \"mimic-cxr-jpg-2.1.0\" -exec cp -r {} data/raw/ \\; 2>/dev/null || true\n",
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
                "# ── Cell 4: Fast Resume Download Check & Zipping ───────────────────────\n",
                "# If raw files are already mounted from your uploaded Kaggle Dataset, this skips in 1 second!\n",
                "!python scripts/download_cohort_physionet.py --cohort data/cohort.parquet --username kmohnishm --password HereisMy2006Bye\n\n",
                "# Compress raw dataset into raw_dataset.zip for 1-click download\n",
                "import os\n",
                "if os.path.exists('data/raw') and len(os.listdir('data/raw')) > 0:\n",
                "    print('\\nCompressing data/raw into raw_dataset.zip for 1-click download from Kaggle Output...')\n",
                "    !zip -q -r raw_dataset.zip data/raw\n",
                "    print('raw_dataset.zip created successfully! Available in Kaggle Output pane.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 5: Train Tabular Branch ───────────────────────────────────────\n",
                "!python scripts/train_tabular.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 6: Train ECG Branch ───────────────────────────────────────────\n",
                "!python scripts/train_ecg.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 7: Train CXR Branch ───────────────────────────────────────────\n",
                "!python scripts/train_cxr.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 8: Train Gated Fusion Layer ───────────────────────────────────\n",
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
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 12: Auto-Push EVERYTHING (git add .) to a New Versioned Branch ─\n",
                "import datetime, os\n\n",
                "# 1. Fetch GitHub Token securely from Kaggle Secrets (Add-ons -> Secrets -> GITHUB_TOKEN)\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    user_secrets = UserSecretsClient()\n",
                "    GITHUB_TOKEN = user_secrets.get_secret(\"GITHUB_TOKEN\")\n",
                "except Exception:\n",
                "    GITHUB_TOKEN = os.environ.get(\"GITHUB_TOKEN\", \"\")\n\n",
                "# 2. Create a unique versioned branch name with timestamp\n",
                "version_tag = datetime.datetime.now().strftime('run-v%Y%m%d-%H%M%S')\n",
                "print(f'Creating and pushing to new version branch: {version_tag}')\n\n",
                "# 3. Configure Git Identity\n",
                "!git config user.name 'KMohnishM'\n",
                "!git config user.email 'kmohnishm@gmail.com'\n\n",
                "# 4. Create & Checkout New Branch\n",
                "!git checkout -b {version_tag}\n\n",
                "# 5. Stage EVERYTHING in working directory (including outputs and model weights)\n",
                "!git add -f outputs/ data/processed/ visualize_results.ipynb 2>/dev/null || true\n",
                "!git add .\n",
                "!git commit -m f'feat(kaggle-run): full automated output push for version {version_tag}'\n\n",
                "# 6. Push new version branch to GitHub\n",
                "if GITHUB_TOKEN:\n",
                "    !git push https://{GITHUB_TOKEN}@github.com/KMohnishM/HealthCare_Analytics.git {version_tag}\n",
                "    print(f'\\n🎉 Successfully pushed branch \"{version_tag}\" with all files to GitHub!')\n",
                "else:\n",
                "    print('\\n⚠️ GITHUB_TOKEN secret not found! Please add label GITHUB_TOKEN under Kaggle Add-ons -> Secrets.')"
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

print("Successfully created master_kaggle_pipeline.ipynb with refined Cell 3 paths!")
