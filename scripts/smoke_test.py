"""
scripts/smoke_test.py
---------------------
Runs the entire cohort extraction, training, and evaluation pipeline
end-to-end using synthetic data and dummy configuration.

Runs entirely on CPU in less than 60 seconds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Setup paths
root = Path(__file__).resolve().parent.parent
config_real  = root / "config" / "config.yaml"
config_dummy = root / "config" / "config_dummy.yaml"
config_temp  = root / "config" / "config_temp.yaml"

def run_cmd(cmd: list[str]) -> bool:
    print(f"\nExecuting: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(root))
    if res.returncode != 0:
        print(f"[FAIL] Command failed with return code {res.returncode}")
        return False
    return True

def main():
    print("=== STARTING MULTIMODAL PIPELINE SMOKE TEST ===")

    # 1. Swap real config out, swap dummy config in
    if not config_dummy.exists():
        print("Error: config_dummy.yaml does not exist.")
        sys.exit(1)

    print("Swapping config.yaml with config_dummy.yaml ...")
    shutil.move(str(config_real), str(config_temp))
    shutil.copy(str(config_dummy), str(config_real))

    success = False
    try:
        # 2. Generate dummy raw data
        if not run_cmd(["python", "scripts/generate_dummy_data.py"]):
            raise RuntimeError("Dummy data generation failed.")

        # 3. Run cohort extraction
        if not run_cmd(["python", "scripts/run_cohort.py"]):
            raise RuntimeError("Cohort extraction failed.")

        # 4. Train Tabular branch (XGBoost)
        if not run_cmd(["python", "scripts/train_tabular.py"]):
            raise RuntimeError("Tabular branch training failed.")

        # 5. Train ECG branch (1D ResNet)
        if not run_cmd(["python", "scripts/train_ecg.py"]):
            raise RuntimeError("ECG branch training failed.")

        # 6. Train CXR branch (DenseNet-121)
        if not run_cmd(["python", "scripts/train_cxr.py"]):
            raise RuntimeError("CXR branch training failed.")

        # 7. Train Fusion Gating Layer
        if not run_cmd(["python", "scripts/train_fusion.py"]):
            raise RuntimeError("Fusion training failed.")

        # 8. Run Evaluation (DCA, Missingness sweep, subgroup fairness)
        if not run_cmd(["python", "scripts/evaluate_all.py"]):
            raise RuntimeError("Evaluation pipeline failed.")

        print("\n[SUCCESS] SMOKE TEST SUCCEEDED! All modules compiled and executed without errors.")
        success = True

    except Exception as e:
        print(f"\n[FAIL] SMOKE TEST FAILED: {str(e)}")
    finally:
        # Restore real config
        print("Restoring original config.yaml ...")
        if config_real.exists():
            os.remove(config_real)
        shutil.move(str(config_temp), str(config_real))

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
