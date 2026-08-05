"""
scripts/download_cohort_physionet.py
---------------------------------------------------------------------------------
Downloads matching ECG records and CXR images directly from PhysioNet web servers
using cookie-based session login credentials. Bypasses Google Cloud completely.
---------------------------------------------------------------------------------
Usage:
    python scripts/download_cohort_physionet.py --cohort data/cohort.parquet
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

def get_physionet_session(username: str, password: str) -> requests.Session:
    """Establish a cookie-based session by logging into PhysioNet website."""
    session = requests.Session()
    login_url = "https://physionet.org/login/"
    
    # 1. Get the login page to extract Django CSRF token
    r_get = session.get(login_url, timeout=15)
    csrf_token = session.cookies.get("csrftoken")
    if not csrf_token:
        match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r_get.text)
        if match:
            csrf_token = match.group(1)
            
    if not csrf_token:
        raise PermissionError("Failed to extract CSRF token from PhysioNet login page.")

    # 2. POST the login credentials
    payload = {
        "username": username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token,
        "next": "/"
    }
    headers = {
        "Referer": login_url,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    r_post = session.post(login_url, data=payload, headers=headers, timeout=15)
    if r_post.status_code != 200 or "sessionid" not in session.cookies:
        raise PermissionError("PhysioNet authentication failed. Verify username and password.")
        
    return session

def download_physionet_file(session: requests.Session, url: str, dest_path: Path, force: bool = False):
    """Download a file from PhysioNet using a logged-in session."""
    if dest_path.exists() and not force:
        return  # Skip if already exists
        
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {
        "Referer": "https://physionet.org/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = session.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        elif response.status_code == 403:
            raise PermissionError(f"Access Denied (403) to {url}. Ensure you have signed the Data Use Agreement.")
        else:
            raise RuntimeError(f"Failed download (Status: {response.status_code})")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download filtered MIMIC cohort directly from PhysioNet")
    parser.add_argument("--cohort", required=True, help="Path to cohort.parquet")
    parser.add_argument("--out-dir", default="data", help="Output data directory")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of parallel downloads")
    parser.add_argument("--limit-subjects", type=int, default=None, help="Limit to a random subset of N subjects for fast local testing")
    parser.add_argument("--username", default=None, help="PhysioNet Username")
    parser.add_argument("--password", default=None, help="PhysioNet Password")
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
        
    if args.limit_subjects is not None:
        import numpy as np
        np.random.seed(42)
        all_subjects = cohort_df["subject_id"].unique()
        if len(all_subjects) > args.limit_subjects:
            selected_subjects = np.random.choice(all_subjects, size=args.limit_subjects, replace=False)
            subject_ids = set(selected_subjects)
            # Filter and overwrite cohort Parquets in data/ directory
            cohort_df = cohort_df[cohort_df["subject_id"].isin(subject_ids)].copy()
            cohort_df.to_parquet(cohort_path, index=False)
            
            # Filter train/val/test splits as well if they exist alongside cohort_path
            for name in ["train", "val", "test"]:
                split_file = cohort_path.parent / f"{name}.parquet"
                if split_file.exists():
                    split_df = pd.read_parquet(split_file)
                    split_df = split_df[split_df["subject_id"].isin(subject_ids)].copy()
                    split_df.to_parquet(split_file, index=False)
            print(f"Limiting to a random subset of {args.limit_subjects} subjects for local testing.")
        else:
            subject_ids = set(all_subjects)
    else:
        subject_ids = set(cohort_df["subject_id"].unique())
        
    print(f"Cohort loaded: {len(cohort_df)} admissions for {len(subject_ids)} patients.")

    # Get PhysioNet Credentials
    username = args.username
    password = args.password
    if username is None or password is None:
        print("\n--- PhysioNet Authentication ---")
        if username is None:
            username = input("PhysioNet Username: ").strip()
        if password is None:
            password = getpass.getpass("PhysioNet Password: ")
            
    print("Establishing authenticated session with PhysioNet...")
    session = get_physionet_session(username, password)
    print("Authentication successful!")

    # 1. ── Download ECGs ──
    print("\n=== STEP 1: Querying ECGs ===")
    raw_ecg_dir = Path(args.out_dir) / "raw" / "mimic-iv-ecg-1.0"
    ecg_list_dest = raw_ecg_dir / "record_list.csv"
    
    print("Fetching ECG record list index...")
    ecg_list_url = "https://physionet.org/files/mimic-iv-ecg/1.0/record_list.csv"
    download_physionet_file(session, ecg_list_url, ecg_list_dest, force=False)
    
    ecg_records_df = pd.read_csv(ecg_list_dest)
    
    # Smart filtering: keep ECGs within discharge window (72h) or closest per admission
    if "dischtime" in cohort_df.columns:
        cohort_sub = cohort_df[["subject_id", "hadm_id", "dischtime"]].copy()
        cohort_sub["dischtime"] = pd.to_datetime(cohort_sub["dischtime"])
        ecg_records_df["ecg_time"] = pd.to_datetime(ecg_records_df["ecg_time"], errors="coerce")
        ecg_merged = cohort_sub.merge(ecg_records_df, on="subject_id", how="inner")
        ecg_merged["hours_before_disch"] = (ecg_merged["dischtime"] - ecg_merged["ecg_time"]).dt.total_seconds() / 3600
        in_win = ecg_merged[(ecg_merged["hours_before_disch"] >= 0) & (ecg_merged["hours_before_disch"] <= 168)]  # within 7 days
        if len(in_win) > 0:
            matching_ecg = in_win.sort_values("hours_before_disch").groupby("hadm_id").first().reset_index()
        else:
            matching_ecg = ecg_records_df[ecg_records_df["subject_id"].isin(subject_ids)].groupby("subject_id").head(2)
    else:
        matching_ecg = ecg_records_df[ecg_records_df["subject_id"].isin(subject_ids)].groupby("subject_id").head(2)

    print(f"Found {len(matching_ecg)} matching ECG records for discharge windows.")

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
    cxr_meta_dest = raw_cxr_dir / "mimic-cxr-2.0.0-metadata.csv.gz"
    
    cxr_tasks = []
    print("Fetching CXR metadata index...")
    # Note: PhysioNet uses version 2.0.0 in the metadata file name under the 2.1.0 directory
    cxr_meta_url = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"
    
    try:
        download_physionet_file(session, cxr_meta_url, cxr_meta_dest, force=False)
        cxr_meta_df = pd.read_csv(cxr_meta_dest, compression="gzip")
        
        # Keep Frontal views closest to discharge
        frontal_cxr = cxr_meta_df[cxr_meta_df["ViewPosition"].isin(["PA", "AP"])].copy() if "ViewPosition" in cxr_meta_df.columns else cxr_meta_df.copy()
        
        if "dischtime" in cohort_df.columns and "StudyDate" in frontal_cxr.columns:
            cohort_sub = cohort_df[["subject_id", "hadm_id", "dischtime"]].copy()
            cohort_sub["dischtime"] = pd.to_datetime(cohort_sub["dischtime"])
            
            sdate = frontal_cxr["StudyDate"].astype(str)
            stime = frontal_cxr["StudyTime"].fillna(0).astype(int).astype(str).str.zfill(6)
            frontal_cxr["study_datetime"] = pd.to_datetime(sdate + " " + stime, format="%Y%m%d %H%M%S", errors="coerce")
            
            cxr_merged = cohort_sub.merge(frontal_cxr, on="subject_id", how="inner")
            cxr_merged["hours_before_disch"] = (cxr_merged["dischtime"] - cxr_merged["study_datetime"]).dt.total_seconds() / 3600
            in_win_cxr = cxr_merged[(cxr_merged["hours_before_disch"] >= 0) & (cxr_merged["hours_before_disch"] <= 168)]  # within 7 days
            if len(in_win_cxr) > 0:
                matching_cxr = in_win_cxr.sort_values("hours_before_disch").groupby("hadm_id").first().reset_index()
            else:
                matching_cxr = frontal_cxr[frontal_cxr["subject_id"].isin(subject_ids)].groupby("subject_id").head(2)
        else:
            matching_cxr = frontal_cxr[frontal_cxr["subject_id"].isin(subject_ids)].groupby("subject_id").head(2)

        print(f"Found {len(matching_cxr)} matching CXR records for discharge windows.")

        for _, row in matching_cxr.iterrows():
            sub_str = str(row["subject_id"])
            prefix = f"p{sub_str[:2]}"
            rel_path = f"files/{prefix}/p{sub_str}/s{row['study_id']}/{row['dicom_id']}.jpg"
            url = f"https://physionet.org/files/mimic-cxr-jpg/2.1.0/{rel_path}"
            dest = raw_cxr_dir / rel_path
            cxr_tasks.append((url, dest))
    except Exception as e:
        print(f"\n[WARNING] Could not access Chest X-Ray files: {e}")
        print("Continuing with Tabular and ECG files only...")

    # Download Queue
    all_tasks = ecg_tasks + cxr_tasks
    print(f"\nQueueing {len(all_tasks)} files for download...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_physionet_file, session, url, dest): (url, dest)
            for url, dest in all_tasks
        }
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            pass

    print("\nExtraction and download complete! Output directory: data/")

if __name__ == "__main__":
    main()
