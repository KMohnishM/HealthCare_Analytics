"""
scripts/generate_dummy_data.py
-------------------------------
Generates a tiny, synthetically generated micro-MIMIC dataset
to smoke-test the entire pipeline locally on CPU or on Colab.

Creates:
  - MIMIC-IV CSVs (admissions, patients, diagnoses_icd, procedures_icd, labevents, chartevents)
  - MIMIC-IV-ECG WFDB records (.hea + .dat binary files)
  - MIMIC-CXR-JPG images (.jpg files)
  - Nested directory structures mirroring the actual databases.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
# We configure this to use a local 'data/raw' folder inside the project
BASE_DIR = Path("data/raw")
MIMIC_IV_DIR = BASE_DIR / "mimic-iv-2.2"
ECG_DIR = BASE_DIR / "mimic-iv-ecg-1.0"
CXR_DIR = BASE_DIR / "mimic-cxr-jpg-2.0.0"

def create_mimic_iv_tables():
    log_dir = MIMIC_IV_DIR / "hosp"
    icu_dir = MIMIC_IV_DIR / "icu"
    log_dir.mkdir(parents=True, exist_ok=True)
    icu_dir.mkdir(parents=True, exist_ok=True)

    # 10 synthetic patients
    subject_ids = list(range(10001, 10011))
    hadm_ids = list(range(20001, 20011))

    # Patients
    patients = pd.DataFrame({
        "subject_id": subject_ids,
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "anchor_age": [72, 65, 81, 58, 70, 77, 63, 79, 85, 68],
        "anchor_year": [2150] * 10,
        "dod": [None, "2150-12-15", None, None, None, None, None, None, None, None]
    })
    patients.to_csv(log_dir / "patients.csv", index=False)

    # Admissions
    admissions = pd.DataFrame({
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "admittime": ["2150-01-01 10:00:00"] * 10,
        "dischtime": ["2150-01-06 14:00:00"] * 10, # LOS = 5 days
        "admission_type": ["EMERGENCY"] * 10,
        "hospital_expire_flag": [0] * 10
    })
    # Add a mock subsequent readmission for subject 10001
    readmit = pd.DataFrame([{
        "subject_id": 10001,
        "hadm_id": 20099,
        "admittime": "2150-01-15 08:00:00", # within 30 days
        "dischtime": "2150-01-20 12:00:00",
        "admission_type": "EMERGENCY",
        "hospital_expire_flag": 0
    }])
    admissions = pd.concat([admissions, readmit], ignore_index=True)
    admissions.to_csv(log_dir / "admissions.csv", index=False)

    # Diagnoses ICD (All heart failure, ICD-10 = I509, ICD-9 = 4280)
    diagnoses = pd.DataFrame({
        "subject_id": subject_ids + [10001],
        "hadm_id": hadm_ids + [20099],
        "seq_num": [1] * 11,
        "icd_code": ["I509"] * 5 + ["4280"] * 5 + ["I509"],
        "icd_version": [10] * 5 + [9] * 5 + [10]
    })
    diagnoses.to_csv(log_dir / "diagnoses_icd.csv", index=False)

    # Procedures
    procedures = pd.DataFrame({
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "seq_num": [1] * 10,
        "icd_code": ["3E033VJ"] * 10,
        "icd_version": [10] * 10
    })
    procedures.to_csv(log_dir / "procedures_icd.csv", index=False)

    # Labevents (hemoglobin, sodium, creatinine, bnp)
    # Sodium = 50983, Creatinine = 50912, Hemoglobin = 51222, BNP = 51002
    lab_records = []
    for hadm in hadm_ids + [20099]:
        lab_records.append({"hadm_id": hadm, "itemid": 50983, "charttime": "2150-01-05 06:00:00", "valuenum": 134.0}) # sodium
        lab_records.append({"hadm_id": hadm, "itemid": 50912, "charttime": "2150-01-05 06:00:00", "valuenum": 1.7})   # creatinine
        lab_records.append({"hadm_id": hadm, "itemid": 51222, "charttime": "2150-01-05 06:00:00", "valuenum": 10.2})  # hgb
        lab_records.append({"hadm_id": hadm, "itemid": 51002, "charttime": "2150-01-05 06:00:00", "valuenum": 820.0}) # bnp
    pd.DataFrame(lab_records).to_csv(log_dir / "labevents.csv", index=False)

    # Chartevents (Heart rate = 220045, Systolic BP = 220179, Temp = 223761)
    chart_records = []
    for hadm in hadm_ids + [20099]:
        chart_records.append({"hadm_id": hadm, "itemid": 220045, "charttime": "2150-01-05 08:00:00", "valuenum": 88.0})
        chart_records.append({"hadm_id": hadm, "itemid": 220179, "charttime": "2150-01-05 08:00:00", "valuenum": 118.0})
        chart_records.append({"hadm_id": hadm, "itemid": 223761, "charttime": "2150-01-05 08:00:00", "valuenum": 98.4})
    pd.DataFrame(chart_records).to_csv(icu_dir / "chartevents.csv", index=False)

    print("MIMIC-IV hosp & icu dummy tables created.")

def create_mimic_ecg():
    files_root = ECG_DIR / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    records = []
    # 10 subjects
    for i, sub_id in enumerate(range(10001, 10011)):
        study_id = 30000 + i
        rec_name = f"rec_{sub_id}"
        
        # Nested MIMIC structure: files/p10/p10001/s30000/rec_10001
        folder = files_root / f"p{str(sub_id)[:2]}" / f"p{sub_id}" / f"s{study_id}"
        folder.mkdir(parents=True, exist_ok=True)
        
        # Generate random 12-channel signal
        sig_data = np.random.randn(5000, 12).astype(np.float32) * 0.2
        
        # Write WFDB file
        wfdb.wrsamp(
            record_name=rec_name,
            fs=500,
            units=["mV"] * 12,
            sig_name=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            p_signal=sig_data,
            fmt=["16"] * 12,
            write_dir=str(folder)
        )
        
        # Relative file name for metadata listing
        rel_path = f"p{str(sub_id)[:2]}/p{sub_id}/s{study_id}/{rec_name}.hea"
        records.append({
            "subject_id": sub_id,
            "study_id": study_id,
            "ecg_time": "2150-01-05 12:00:00",
            "filename": rel_path
        })
        
    pd.DataFrame(records).to_csv(ECG_DIR / "record_list.csv", index=False)
    print("MIMIC-IV-ECG dummy waveforms created.")

def create_mimic_cxr():
    files_root = CXR_DIR / "files"
    files_root.mkdir(parents=True, exist_ok=True)

    records = []
    for i, sub_id in enumerate(range(10001, 10011)):
        study_id = 40000 + i
        dicom_id = f"dicom_{sub_id}"
        
        folder = files_root / f"p{str(sub_id)[:2]}" / f"p{sub_id}" / f"s{study_id}"
        folder.mkdir(parents=True, exist_ok=True)
        
        # Create a simple grey image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.save(folder / f"{dicom_id}.jpg")
        
        records.append({
            "subject_id": sub_id,
            "study_id": study_id,
            "dicom_id": dicom_id,
            "ViewPosition": "PA",
            "StudyDate": "21500105",
            "StudyTime": "143000.000"
        })
        
    pd.DataFrame(records).to_csv(CXR_DIR / "mimic-cxr-2.0.0-metadata.csv", index=False)
    print("MIMIC-CXR-JPG dummy chest X-rays created.")

if __name__ == "__main__":
    create_mimic_iv_tables()
    create_mimic_ecg()
    create_mimic_cxr()
    print("\n--- Smoke test dataset generation complete. ---")
