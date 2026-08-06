"""
scripts/generate_mock_clinical_tables.py
---------------------------------------------------------------------------------
Generates mock/dummy labevents.csv and chartevents.csv clinical tables for the
active cohort to bypass large raw file dependencies during testing or cloud execution.
Usage:
    python scripts/generate_mock_clinical_tables.py
"""

import os
import numpy as np
import pandas as pd

def main():
    cohort_path = "data/cohort.parquet"
    if not os.path.exists(cohort_path):
        print(f"Error: Cohort file not found at {cohort_path}")
        return

    print("Loading cohort...")
    cohort = pd.read_parquet(cohort_path)
    hadm_ids = cohort["hadm_id"].unique()
    
    print(f"Generating mock clinical data for {len(hadm_ids)} admissions...")
    
    # 1. Generate Labevents
    # Common itemids: Creatinine (50912), BNP (50963), Sodium (50983), Potassium (50971), BUN (50912), WBC (51301)
    np.random.seed(42)
    labs = []
    item_ranges = {
        50912: (0.5, 4.0),    # Creatinine
        50963: (100, 2000),   # BNP
        50983: (130, 145),    # Sodium
        50971: (3.0, 6.0),    # Potassium
        51301: (4.0, 20.0),   # WBC
        51222: (8.0, 16.0),   # Hemoglobin (HGB)
    }
    
    for hadm in hadm_ids:
        row = cohort[cohort["hadm_id"] == hadm].iloc[0]
        sub = row["subject_id"]
        disch = pd.to_datetime(row["dischtime"])
        
        # Generate 1-2 samples per lab item near discharge
        for itemid, (low, high) in item_ranges.items():
            val = np.random.uniform(low, high)
            charttime = disch - pd.Timedelta(hours=np.random.uniform(2, 48))
            labs.append({
                "subject_id": sub,
                "hadm_id": hadm,
                "itemid": itemid,
                "charttime": charttime.strftime("%Y-%m-%d %H:%M:%S"),
                "valuenum": val
            })
            
    df_labs = pd.DataFrame(labs)
    os.makedirs("data/raw/mimic-iv-2.2/hosp", exist_ok=True)
    df_labs.to_csv("data/raw/mimic-iv-2.2/hosp/labevents.csv", index=False)
    print("  -> Saved data/raw/mimic-iv-2.2/hosp/labevents.csv")

    # 2. Generate Chartevents (Vitals)
    # Common itemids: Heart Rate (220045), Systolic BP (220179), Diastolic BP (220180), Resp Rate (220210), Temp (223761)
    vitals = []
    vital_ranges = {
        220045: (60, 110),    # Heart Rate
        220179: (90, 150),    # Systolic BP
        220180: (50, 95),     # Diastolic BP
        220210: (12, 28),     # Resp Rate
        223761: (96.0, 101.0) # Temp (F)
    }
    
    for hadm in hadm_ids:
        row = cohort[cohort["hadm_id"] == hadm].iloc[0]
        sub = row["subject_id"]
        disch = pd.to_datetime(row["dischtime"])
        
        # Generate 3 vital checks within 24h before discharge
        for itemid, (low, high) in vital_ranges.items():
            for i in range(3):
                val = np.random.uniform(low, high)
                charttime = disch - pd.Timedelta(hours=np.random.uniform(1, 24))
                vitals.append({
                    "hadm_id": hadm,
                    "itemid": itemid,
                    "charttime": charttime.strftime("%Y-%m-%d %H:%M:%S"),
                    "valuenum": val
                })
                
    df_vitals = pd.DataFrame(vitals)
    os.makedirs("data/raw/mimic-iv-2.2/icu", exist_ok=True)
    df_vitals.to_csv("data/raw/mimic-iv-2.2/icu/chartevents.csv", index=False)
    print("  -> Saved data/raw/mimic-iv-2.2/icu/chartevents.csv")
    print("Mock clinical data generation complete!")

if __name__ == "__main__":
    main()
