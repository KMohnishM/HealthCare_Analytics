"""
scripts/download_cohort_multimodal.py
---------------------------------------------------------------------------------
Downloads matching ECG records and CXR images for the cohort extracted in Step 1.
Leverages the public MIMIC GCP buckets (Requester Pays) to filter and download.
---------------------------------------------------------------------------------
Usage:
    python scripts/download_cohort_multimodal.py \
        --cohort cohort_with_features.csv \
        --gcp-project your-gcp-project-id
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

def get_gcs_bucket(bucket_name: str, project_id: str, client: storage.Client) -> storage.Bucket:
    """Get GCS bucket with requester pays enabled."""
    bucket = client.bucket(bucket_name, user_project=project_id)
    return bucket

def download_blob(bucket: storage.Bucket, blob_path: str, dest_path: Path):
    """Download a single blob to a destination file path."""
    if dest_path.exists():
        return  # Skip if already downloaded
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob = bucket.blob(blob_path)
    try:
        blob.download_to_filename(str(dest_path))
    except Exception as e:
        print(f"Error downloading {blob_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download filtered MIMIC-IV multimodal cohort data")
    parser.add_argument("--cohort", required=True, help="Path to the cohort_with_features.csv file from BigQuery")
    parser.add_argument("--gcp-project", required=True, help="Your Google Cloud Project ID (required for billing/requester-pays)")
    parser.add_argument("--out-dir", default="data", help="Output data directory")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel download workers")
    args = parser.parse_args()

    cohort_path = Path(args.cohort)
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort CSV file not found at: {cohort_path}")

    # Load cohort subject IDs
    print("Loading cohort...")
    cohort_df = pd.read_csv(cohort_path)
    subject_ids = set(cohort_df["subject_id"].unique())
    print(f"Cohort loaded: {len(cohort_df)} admissions for {len(subject_ids)} patients.")

    # Initialize GCS client
    print(f"Initializing GCS client with billing project: {args.gcp-project}")
    client = storage.Client(project=args.gcp_project)

    # 1. ── Download ECG Waveforms ──
    print("\n=== STEP 1: Downloading Matching ECGs ===")
    ecg_bucket = get_gcs_bucket("mimic-iv-ecg-1.0", args.gcp_project, client)
    
    # Download ECG record list to find matching subjects
    print("Downloading ECG record list index...")
    record_list_dest = Path(args.out_dir) / "raw" / "mimic-iv-ecg-1.0" / "record_list.csv"
    download_blob(ecg_bucket, "record_list.csv", record_list_dest)
    
    ecg_records_df = pd.read_csv(record_list_dest)
    matching_ecg = ecg_records_df[ecg_records_df["subject_id"].isin(subject_ids)]
    print(f"Found {len(matching_ecg)} matching ECG records for cohort patients.")

    # Queue ECG files (.hea and .dat)
    ecg_download_tasks = []
    raw_ecg_dir = Path(args.out_dir) / "raw" / "mimic-iv-ecg-1.0"
    for _, row in matching_ecg.iterrows():
        path_str = row["path"]  # e.g. "files/p1000/p10000032/s10000032/10000032_0001"
        for ext in [".hea", ".dat"]:
            blob_path = f"{path_str}{ext}"
            dest_path = raw_ecg_dir / blob_path
            ecg_download_tasks.append((blob_path, dest_path))

    # 2. ── Download CXR Images ──
    print("\n=== STEP 2: Downloading Matching CXRs ===")
    cxr_bucket = get_gcs_bucket("mimic-cxr-jpg-2.1.0", args.gcp_project, client)
    
    # Download CXR metadata to find matching subjects
    print("Downloading CXR metadata index...")
    cxr_meta_dest = Path(args.out_dir) / "raw" / "mimic-cxr-jpg-2.1.0" / "mimic-cxr-2.1.0-metadata.csv.gz"
    download_blob(cxr_bucket, "mimic-cxr-2.1.0-metadata.csv.gz", cxr_meta_dest)
    
    cxr_meta_df = pd.read_csv(cxr_meta_dest, compression="gzip")
    matching_cxr = cxr_meta_df[cxr_meta_df["subject_id"].isin(subject_ids)]
    print(f"Found {len(matching_cxr)} matching CXR records for cohort patients.")

    # Queue CXR images (.jpg)
    cxr_download_tasks = []
    raw_cxr_dir = Path(args.out_dir) / "raw" / "mimic-cxr-jpg-2.1.0"
    for _, row in matching_cxr.iterrows():
        # Path details: dicom_id, study_id, subject_id
        # MIMIC-CXR folder format: p{first_2_digits_of_subject_id}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        sub_str = str(row["subject_id"])
        prefix = f"p{sub_str[:2]}"
        blob_path = f"files/{prefix}/p{sub_str}/s{row['study_id']}/{row['dicom_id']}.jpg"
        dest_path = raw_cxr_dir / blob_path
        cxr_download_tasks.append((blob_path, dest_path))

    # Run Parallel Downloads
    total_tasks = ecg_download_tasks + cxr_download_tasks
    print(f"\nQueueing {len(total_tasks)} total files for download (ECGs + CXRs)...")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                download_blob,
                ecg_bucket if "ecg" in str(dest) else cxr_bucket,
                blob,
                dest
            ): (blob, dest)
            for blob, dest in total_tasks
        }
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            pass

    print("\nDownload complete! All filtered files saved to the local 'data/' directory.")

if __name__ == "__main__":
    main()
