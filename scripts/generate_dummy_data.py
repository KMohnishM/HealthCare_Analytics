from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from PIL import Image
import argparse

BASE_DIR = Path("data/raw")
MIMIC_IV_DIR = BASE_DIR / "mimic-iv-2.2"
ECG_DIR = BASE_DIR / "mimic-iv-ecg-1.0"
CXR_DIR = BASE_DIR / "mimic-cxr-jpg-2.0.0"

def create_mimic_iv_tables(n_patients, readmit_rate, seed):
    rng = np.random.default_rng(seed)
    subject_ids = list(range(10001, 10001 + n_patients))
    hadm_ids = list(range(20001, 20001 + n_patients))
    patients = pd.DataFrame({
        "subject_id": subject_ids,
        "gender": rng.choice(["M", "F"], n_patients, p=[0.55, 0.45]),
        "anchor_age": rng.normal(72, 12, n_patients).clip(18, 100).astype(int),
        "anchor_year": [2150] * n_patients,
        "dod": [None] * n_patients
    })
    patients.to_csv(MIMIC_IV_DIR / "hosp" / "patients.csv", index=False)
    admissions = pd.DataFrame({
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "admittime": ["2150-01-01 10:00:00"] * n_patients,
        "dischtime": ["2150-01-06 14:00:00"] * n_patients,
        "admission_type": ["EMERGENCY"] * n_patients,
        "hospital_expire_flag": [0] * n_patients
    })
    readmit_hadm = hadm_ids[int(n_patients * readmit_rate)]
    if readmit_hadm > 0:
        readmit = pd.DataFrame([{
            "subject_id": subject_ids[hadm_ids.tolist().index(readmit_hadm[0])],
            "hadm_id": readmit_hadm[0],
            "admittime": "2150-01-15 08:00:00",
            "dischtime": "2150-01-20 12:00:00",
            "admission_type": "EMERGENCY",
            "hospital_expire_flag": 0
        }])
        admissions = pd.concat([admissions, readmit], ignore_index=True)
    admissions.to_csv(MIMIC_IV_DIR / "hosp" / "admissions.csv", index=False)
    diagnoses = pd.DataFrame({
        "subject_id": subject_ids + ([readmit_hadm[0]] if readmit_hadm.size > 0 else []),
        "hadm_id": hadm_ids + ([readmit_hadm[0]] if readmit_hadm.size > 0 else []),
        "seq_num": [1] * (n_patients + (1 if readmit_hadm.size > 0 else 0)),
        "icd_code": ["I509"] * n_patients + (["4280"] * (n_patients//2) if readmit_hadm.size > 0 else []),
        "icd_version": [10] * n_patients + ([9] * (n_patients//2) if readmit_hadm.size > 0 else [])
    })
    diagnoses.to_csv(MIMIC_IV_DIR / "hosp" / "diagnoses_icd.csv", index=False)
    procedures = pd.DataFrame({
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "seq_num": [1] * n_patients,
        "icd_code": ["3E033VJ"] * n_patients,
        "icd_version": [10] * n_patients
    })
    procedures.to_csv(MIMIC_IV_DIR / "hosp" / "procedures_icd.csv", index=False)
    lab_records = []
    for hadm in hadm_ids:
        lab_records.append({"hadm_id": hadm, "itemid": 50983, "charttime": "2150-01-05 06:00:00", "valuenum": rng.integers(130, 145)})
        lab_records.append({"hadm_id": hadm, "itemid": 50912, "charttime": "2150-01-05 06:00:00", "valuenum": rng.uniform(0.5, 1.5)})
        lab_records.append({"hadm_id": hadm, "itemid": 51222, "charttime": "2150-01-05 06:00:00", "valuenum": rng.uniform(8, 15)})
        lab_records.append({"hadm_id": hadm, "itemid": 51002, "charttime": "2150-01-05 06:00:00", "valuenum": rng.uniform(500, 1500)})
    pd.DataFrame(lab_records).to_csv(MIMIC_IV_DIR / "hosp" / "labevents.csv", index=False)
    chart_records = []
    for hadm in hadm_ids:
        chart_records.append({"hadm_id": hadm, "itemid": 220045, "charttime": "2150-01-05 08:00:00", "valuenum": rng.integers(60, 120)})
        chart_records.append({"hadm_id": hadm, "itemid": 220179, "charttime": "2150-01-05 08:00:00", "valuenum": rng.integers(100, 140)})
        chart_records.append({"hadm_id": hadm, "itemid": 223761, "charttime": "2150-01-05 08:00:00", "valuenum": rng.integers(95, 105)})
    pd.DataFrame(chart_records).to_csv(ECG_DIR / "record_list.csv", index=False)
    print("MIMIC-IV hosp & icu dummy tables created.")

def create_mimic_ecg(n_patients, ecg_rate, seed):
    rng = np.random.default_rng(seed)
    files_root = ECG_DIR / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    records = []
    for i, sub_id in enumerate(range(10001, 10001 + n_patients)):
        if rng.random() < ecg_rate:
            study_id = 30000 + i
            folder = files_root / f"p{str(sub_id)[:2]}" / f"p{sub_id}" / f"s{study_id}"
            folder.mkdir(parents=True, exist_ok=True)
            sig_data = np.zeros((5000, 12), dtype=np.float32)
            hr = rng.integers(50, 120)
            rr = 60 / hr
            for lead in range(12):
                for beat in range(10):
                    beat_center = (beat + 1) * rr * 500
                    sig_data[lead][beat * 500:(beat + 1) * 500] += rng.normal(0, 0.1, 500) + np.sin(2 * np.pi * beat * rr * 500 / 500.0)
            wfdb.wrsamp(record_name=f"rec_{sub_id}", fs=500, units=["mV"] * 12, sig_name=[f"I", f"II", f"III", f"aVR", f"aVL", f"aVF", f"V1", f"V2", f"V3", f"V4", f"V6"], p_signal=sig_data, fmt=["16"] * 12, write_dir=str(folder))
            rel_path = f"p{str(sub_id)[:2]}/p{sub_id}/s{study_id}/rec_{sub_id}.hea"
            records.append({
                "subject_id": sub_id,
                "study_id": study_id,
                "ecg_time": "2150-01-05 12:00:00",
                "filename": rel_path
            })
    pd.DataFrame(records).to_csv(ECG_DIR / "record_list.csv", index=False)
    print("MIMIC-IV-ECG dummy waveforms created.")

def create_mimic_cxr(n_patients, cxr_rate, seed):
    rng = np.random.default_rng(seed)
    files_root = CXR_DIR / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    records = []
    for i, sub_id in enumerate(range(10001, 10001 + n_patients)):
        if rng.random() < cxr_rate:
            study_id = 40000 + i
            dicom_id = f"dicom_{sub_id}"
            folder = files_root / f"p{str(sub_id)[:2]}" / f"p{sub_id}" / f"s{study_id}"
            folder.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-patients', type=int, default=200)
    parser.add_argument('--readmit-rate', type=float, default=0.12)
    parser.add_argument('--ecg-rate', type=float, default=0.6)
    parser.add_argument('--cxr-rate', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='data/raw')
    parser.add_argument('--gzip', action='store_true')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(MIMIC_IV_DIR / "hosp", exist_ok=True)
    os.makedirs(ECG_DIR, exist_ok=True)
    os.makedirs(CXR_DIR / "files", exist_ok=True)
    
    create_mimic_iv_tables(args.n_patients, args.readmit_rate, args.seed)
    create_mimic_ecg(args.n_patients, args.ecg_rate, args.seed)
    create_mimic_cxr(args.n_patients, args.cxr_rate, args.seed)
    print("\\n--- Smoke test dataset generation complete. ---")

if __name__ == "__main__":
    main()
EOF