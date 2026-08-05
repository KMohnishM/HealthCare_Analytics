import os
import gzip
import shutil
import pandas as pd
import requests

cohort_path = "data/cohort.parquet"
cohort_df = pd.read_parquet(cohort_path)
subject_ids = set(cohort_df["subject_id"].unique())

print(f"Total Cohort Patients: {len(subject_ids)}")

auth = ("kmohnishm", "HereisMy2006Bye")

def download_file(url, local_path, use_auth=False):
    print(f"Downloading {url} to {local_path}...")
    headers = {}
    if use_auth:
        r = requests.get(url, auth=auth, stream=True, timeout=30)
    else:
        r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# ECG (Public index)
ecg_local = "scratch/temp_ecg_list.csv"
if not os.path.exists(ecg_local):
    download_file("https://physionet.org/files/mimic-iv-ecg/1.0/record_list.csv", ecg_local)
ecg_df = pd.read_csv(ecg_local)
matching_ecg = ecg_df[ecg_df["subject_id"].isin(subject_ids)]
print(f"Total matching ECG records on PhysioNet: {len(matching_ecg)}")

# CXR (Restricted index)
cxr_local = "scratch/temp_cxr_meta_210.csv.gz"
if not os.path.exists(cxr_local):
    try:
        download_file("https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.1.0-metadata.csv.gz", cxr_local, use_auth=True)
    except Exception as e:
        print("Could not download 2.1.0 metadata directly:", e)
        # Try 2.0.0 metadata
        cxr_local = "scratch/temp_cxr_meta_200.csv.gz"
        if not os.path.exists(cxr_local):
            download_file("https://physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz", cxr_local, use_auth=True)

cxr_df = pd.read_csv(cxr_local, compression="gzip")
matching_cxr = cxr_df[cxr_df["subject_id"].isin(subject_ids)]
print(f"Total matching CXR records on PhysioNet: {len(matching_cxr)}")
