"""
scripts/audit_download.py
---------------------------------------------------------------------------------
Scans the cohort and local folders to report on download completeness.
Usage:
    python scripts/audit_download.py --cohort data/cohort.parquet
"""

import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Audit download status of the cohort")
    parser.add_argument("--cohort", default="data/cohort.parquet", help="Path to cohort.parquet")
    parser.add_argument("--out-dir", default="data", help="Output data directory")
    args = parser.parse_args()

    cohort_path = Path(args.cohort)
    if not cohort_path.exists():
        print(f"Error: Cohort file not found at: {cohort_path}")
        return

    # Read cohort
    cohort_df = pd.read_parquet(cohort_path)
    subject_ids = set(cohort_df["subject_id"].unique())
    print(f"Cohort loaded: {len(cohort_df)} admissions for {len(subject_ids)} patients.")

    # 1. Audit ECGs
    print("\n--- Auditing ECG Waveforms ---")
    raw_ecg_dir = Path(args.out_dir) / "raw" / "mimic-iv-ecg-1.0"
    ecg_list_dest = raw_ecg_dir / "record_list.csv"
    
    if not ecg_list_dest.exists():
        print("Warning: ECG record list not found on disk. Cannot audit ECGs.")
    else:
        ecg_records_df = pd.read_csv(ecg_list_dest)
        
        # Filter to discharge window ECGs (matching download filtering logic)
        if "dischtime" in cohort_df.columns:
            cohort_sub = cohort_df[["subject_id", "hadm_id", "dischtime"]].copy()
            cohort_sub["dischtime"] = pd.to_datetime(cohort_sub["dischtime"])
            ecg_records_df["ecg_time"] = pd.to_datetime(ecg_records_df["ecg_time"], errors="coerce")
            ecg_merged = cohort_sub.merge(ecg_records_df, on="subject_id", how="inner")
            ecg_merged["hours_before_disch"] = (ecg_merged["dischtime"] - ecg_merged["ecg_time"]).dt.total_seconds() / 3600
            in_win = ecg_merged[(ecg_merged["hours_before_disch"] >= 0) & (ecg_merged["hours_before_disch"] <= 168)]
            if len(in_win) > 0:
                expected_ecg = in_win.sort_values("hours_before_disch").groupby("hadm_id").first().reset_index()
            else:
                expected_ecg = ecg_records_df[ecg_records_df["subject_id"].isin(subject_ids)].groupby("subject_id").head(2)
        else:
            expected_ecg = ecg_records_df[ecg_records_df["subject_id"].isin(subject_ids)].groupby("subject_id").head(2)

        ecg_found = 0
        ecg_missing = 0
        for _, row in expected_ecg.iterrows():
            path_str = row["path"]
            all_exist = True
            for ext in [".hea", ".dat"]:
                file_path = raw_ecg_dir / f"{path_str}{ext}"
                if not file_path.exists():
                    all_exist = False
            if all_exist:
                ecg_found += 1
            else:
                ecg_missing += 1

        print(f"Total ECG records expected for cohort: {len(expected_ecg)}")
        print(f"  -> Successfully Downloaded: {ecg_found} ({(ecg_found/max(1, len(expected_ecg))*100):.1f}%)")
        print(f"  -> Missing/Pending: {ecg_missing}")

    # 2. Audit CXRs
    print("\n--- Auditing Chest X-Rays ---")
    raw_cxr_dir = Path(args.out_dir) / "raw" / "mimic-cxr-jpg-2.1.0"
    cxr_meta_dest = raw_cxr_dir / "mimic-cxr-2.0.0-metadata.csv.gz"

    if not cxr_meta_dest.exists():
        print("Warning: CXR metadata index not found on disk. Cannot audit CXRs.")
    else:
        cxr_meta_df = pd.read_csv(cxr_meta_dest, compression="gzip")
        frontal_cxr = cxr_meta_df[cxr_meta_df["ViewPosition"].isin(["PA", "AP"])].copy() if "ViewPosition" in cxr_meta_df.columns else cxr_meta_df.copy()
        
        if "dischtime" in cohort_df.columns and "StudyDate" in frontal_cxr.columns:
            cohort_sub = cohort_df[["subject_id", "hadm_id", "dischtime"]].copy()
            cohort_sub["dischtime"] = pd.to_datetime(cohort_sub["dischtime"])
            sdate = frontal_cxr["StudyDate"].astype(str)
            stime = frontal_cxr["StudyTime"].fillna(0).astype(int).astype(str).str.zfill(6)
            frontal_cxr["study_datetime"] = pd.to_datetime(sdate + " " + stime, format="%Y%m%d %H%M%S", errors="coerce")
            
            cxr_merged = cohort_sub.merge(frontal_cxr, on="subject_id", how="inner")
            cxr_merged["hours_before_disch"] = (cxr_merged["dischtime"] - cxr_merged["study_datetime"]).dt.total_seconds() / 3600
            in_win_cxr = cxr_merged[(cxr_merged["hours_before_disch"] >= 0) & (cxr_merged["hours_before_disch"] <= 168)]
            if len(in_win_cxr) > 0:
                expected_cxr = in_win_cxr.sort_values("hours_before_disch").groupby("hadm_id").first().reset_index()
            else:
                expected_cxr = frontal_cxr[frontal_cxr["subject_id"].isin(subject_ids)].groupby("subject_id").head(2)
        else:
            expected_cxr = frontal_cxr[frontal_cxr["subject_id"].isin(subject_ids)].groupby("subject_id").head(2)

        cxr_found = 0
        cxr_missing = 0
        for _, row in expected_cxr.iterrows():
            sub_str = str(row["subject_id"])
            prefix = f"p{sub_str[:2]}"
            rel_path = f"files/{prefix}/p{sub_str}/s{row['study_id']}/{row['dicom_id']}.jpg"
            dest = raw_cxr_dir / rel_path
            if dest.exists():
                cxr_found += 1
            else:
                cxr_missing += 1

        print(f"Total CXR records expected for cohort: {len(expected_cxr)}")
        print(f"  -> Successfully Downloaded: {cxr_found} ({(cxr_found/max(1, len(expected_cxr))*100):.1f}%)")
        print(f"  -> Missing/Pending: {cxr_missing}")

if __name__ == "__main__":
    main()
