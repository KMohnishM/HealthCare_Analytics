"""
scripts/download_cohort_physionet.py
---------------------------------------------------------------------------------
Downloads matching ECG records and CXR images directly from PhysioNet web servers
using your PhysioNet login credentials. Bypasses Google Cloud completely.
---------------------------------------------------------------------------------
Usage:
    python scripts/download_cohort_physionet.py --cohort data/cohort/cohort.parquet
"""

from __future__ import annotations

import argparse
import getpass
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

def download_physionet_file(url: str, dest_path: Path, auth: tuple[str, str]):
    """Download a file from PhysioNet using HTTP Basic Auth."""
    if dest_path.exists():
        return  # Skip if already exists
        
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, auth=auth, stream=True, timeout=30)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        elif response.status_code == 401:
            raise PermissionError("Authentication failed. Check your PhysioNet credentials.")
        else:
            print(f"\nFailed to download {url} (Status: {response.status_code})")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download filtered MIMIC cohort directly from PhysioNet")
    parser.add_argument("--cohort", required=True, help="Path to cohort.parquet (or cohort.csv)")
    parser.add_argument("--out-dir", default="data", help="Output data directory")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of parallel downloads")
    args = parser.parse_args()

    cohort_path = Path(args.cohort)
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found at: {cohort_path}")

    # Read cohort
    print("Loading cohort...")
    if cohort_path.suffix == ".parquet":
        cohort_df = pd.read_parquet(cohort_path)
    else:
        cohort_df = pd.read_csv(cohort_path)
        
    subject_ids = set(cohort_df["subject_id"].unique())
    print(f"Cohort loaded: {len(cohort_df)} admissions for {len(subject_ids)} patients.")

    # Get PhysioNet Credentials
    print("\n--- PhysioNet Authentication ---")
    username = input("PhysioNet Username: ").strip()
    password = getpass.getpass("PhysioNet Password: ")
    auth = (username, password)

    # 1. ── Download ECGs ──
    print("\n=== STEP 1: Querying ECGs ===")
    raw_ecg_dir = Path(args.out_dir) / "raw" / "mimic-iv-ecg-1.0"
    ecg_list_dest = raw_ecg_dir / "record_list.csv"
    
    print("Fetching ECG record list index...")
    ecg_list_url = "https://physionet.org/files/mimic-iv-ecg/1.0/record_list.csv"
    download_physionet_file(ecg_list_url, ecg_list_dest, auth)
    
    ecg_records_df = pd.read_csv(ecg_list_dest)
    matching_ecg = ecg_records_df[ecg_records_df["subject_id"].isin(subject_ids)]
    print(f"Found {len(matching_ecg)} matching ECG records.")

    ecg_tasks = []
    for _, row in matching_ecg.iterrows():
        path_str = row["path"]
        for ext in [".hea", ".dat"]:
            file_path = f"{path_str}{ext}"
            url = f"https://physionet.org/files/mimic-iv-ecg/1.0/{file_path}"
            dest = raw_ecg_dir / file_path
            ecg_tasks.append((url, dest))

    # 2. ── Download CXRs ──
    print("\n=== STEP 2: Querying CXRs ===")
    raw_cxr_dir = Path(args.out_dir) / "raw" / "mimic-cxr-jpg-2.1.0"
    cxr_meta_dest = raw_cxr_dir / "mimic-cxr-2.1.0-metadata.csv.gz"
    
    print("Fetching CXR metadata index...")
    cxr_meta_url = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.1.0-metadata.csv.gz"
    download_physionet_file(cxr_meta_url, cxr_meta_dest, auth)
    
    cxr_meta_df = pd.read_csv(cxr_meta_dest, compression="gzip")
    matching_cxr = cxr_meta_df[cxr_meta_df["subject_id"].isin(subject_ids)]
    print(f"Found {len(matching_cxr)} matching CXR records.")

    cxr_tasks = []
    for _, row in matching_cxr.iterrows():
        sub_str = str(row["subject_id"])
        prefix = f"p{sub_str[:2]}"
        rel_path = f"files/{prefix}/p{sub_str}/s{row['study_id']}/{row['dicom_id']}.jpg"
        url = f"https://physionet.org/files/mimic-cxr-jpg/2.1.0/{rel_path}"
        dest = raw_cxr_dir / rel_path
        cxr_tasks.append((url, dest))

    # Download Queue
    all_tasks = ecg_tasks + cxr_tasks
    print(f"\nQueueing {len(all_tasks)} files for download...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_physionet_file, url, dest, auth): (url, dest)
            for url, dest in all_tasks
        }
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            pass

    print("\nExtraction and download complete! Output directory: data/")

if __name__ == "__main__":
    main()
