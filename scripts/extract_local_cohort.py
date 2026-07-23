"""
scripts/extract_local_cohort.py
---------------------------------------------------------------------------------
Filters and copies matching ECG waveforms and CXR images from the full raw local
MIMIC datasets to a compact folder, then zips it for easy sharing.
---------------------------------------------------------------------------------
Usage:
    python scripts/extract_local_cohort.py
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config

def main():
    cfg = load_config()
    
    cohort_dir = Path(cfg.paths.cohort_dir)
    cohort_path = cohort_dir / "cohort.parquet"
    
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort file not found at: {cohort_path}. "
            "Please run 'python scripts/run_cohort.py' first to generate it."
        )

    print("Loading cohort...")
    cohort_df = pd.read_parquet(cohort_path)
    subject_ids = set(cohort_df["subject_id"].unique())
    print(f"Cohort loaded: {len(cohort_df)} admissions for {len(subject_ids)} patients.")

    # Define paths
    raw_iv_dir = Path(cfg.paths.mimic_iv_dir)
    raw_ecg_dir = Path(cfg.paths.mimic_ecg_dir)
    raw_cxr_dir = Path(cfg.paths.mimic_cxr_dir)

    compact_root = Path("data/processed_compact")
    compact_root.mkdir(parents=True, exist_ok=True)

    # Copy cohort.parquet
    shutil.copy(cohort_path, compact_root / "cohort.parquet")
    # Copy split datasets
    for split in ["train", "val", "test"]:
        split_file = cohort_dir / f"{split}.parquet"
        if split_file.exists():
            shutil.copy(split_file, compact_root / f"{split}.parquet")

    # 1. ── Extract ECG Waveforms ──
    print("\n=== STEP 1: Copying Filtered ECGs ===")
    ecg_record_list_path = raw_ecg_dir / "record_list.csv"
    if not ecg_record_list_path.exists():
         # Check subfolder
         ecg_record_list_path = raw_ecg_dir / "hosp" / "record_list.csv"
         
    if not ecg_record_list_path.exists():
        print(f"Warning: ECG record_list.csv not found in {raw_ecg_dir}. Skipping ECG extraction.")
        matching_ecg = pd.DataFrame()
    else:
        ecg_records_df = pd.read_csv(ecg_record_list_path)
        matching_ecg = ecg_records_df[ecg_records_df["subject_id"].isin(subject_ids)]
        print(f"Found {len(matching_ecg)} matching ECG records.")

        dest_ecg_dir = compact_root / "raw" / "mimic-iv-ecg-1.0"
        dest_ecg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ecg_record_list_path, dest_ecg_dir / "record_list.csv")

        # Copy .hea and .dat files
        copy_tasks = []
        for _, row in matching_ecg.iterrows():
            path_str = row["path"]
            for ext in [".hea", ".dat"]:
                src_file = raw_ecg_dir / f"{path_str}{ext}"
                dest_file = dest_ecg_dir / f"{path_str}{ext}"
                if src_file.exists():
                    copy_tasks.append((src_file, dest_file))

        print("Copying ECG records...")
        for src, dest in tqdm(copy_tasks, desc="ECG"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    # 2. ── Extract CXR Images ──
    print("\n=== STEP 2: Copying Filtered CXRs ===")
    cxr_meta_files = list(raw_cxr_dir.glob("mimic-cxr-*-metadata.csv*"))
    if not cxr_meta_files:
        print(f"Warning: CXR metadata file not found in {raw_cxr_dir}. Skipping CXR extraction.")
    else:
        cxr_meta_path = cxr_meta_files[0]
        cxr_meta_df = pd.read_csv(cxr_meta_path)
        matching_cxr = cxr_meta_df[cxr_meta_df["subject_id"].isin(subject_ids)]
        print(f"Found {len(matching_cxr)} matching CXR records.")

        dest_cxr_dir = compact_root / "raw" / raw_cxr_dir.name
        dest_cxr_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cxr_meta_path, dest_cxr_dir / cxr_meta_path.name)

        cxr_copy_tasks = []
        for _, row in matching_cxr.iterrows():
            sub_str = str(row["subject_id"])
            prefix = f"p{sub_str[:2]}"
            rel_path = f"files/{prefix}/p{sub_str}/s{row['study_id']}/{row['dicom_id']}.jpg"
            src_file = raw_cxr_dir / rel_path
            dest_file = dest_cxr_dir / rel_path
            if src_file.exists():
                cxr_copy_tasks.append((src_file, dest_file))

        print("Copying CXR images...")
        for src, dest in tqdm(cxr_copy_tasks, desc="CXR"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    # 3. ── Zipping ──
    zip_filename = "mimic_cohort_dataset.zip"
    print(f"\n=== STEP 3: Zipping compact dataset to '{zip_filename}' ===")
    
    file_list = list(compact_root.rglob("*"))
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(file_list, desc="Zipping"):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(compact_root.parent))

    print(f"\nAll done! Send '{zip_filename}' to your teammate.")

if __name__ == "__main__":
    main()
