-- scripts/bigquery_extract.sql
-- ---------------------------------------------------------------------------------
-- Central SQL cohort and feature extraction query for MIMIC-IV on Google BigQuery.
-- Runs end-to-end on `physionet-data` public schemas.
-- ---------------------------------------------------------------------------------

WITH
  -- 1. Identify primary diagnosis HF admissions
  hf_diagnoses AS (
    SELECT DISTINCT hadm_id
    FROM `physionet-data.mimiciv_hosp.diagnoses_icd`
    WHERE seq_num = 1
      AND (
        (icd_version = 10 AND STARTS_WITH(icd_code, 'I50'))
        OR
        (icd_version = 9 AND icd_code IN (
          '4280', '4281', '42820', '42821', '42822', '42823', '42830', '42831',
          '42832', '42833', '42840', '42841', '42842', '42843', '4289'
        ))
      )
  ),

  -- 2. Extract base admissions and calculate age and LOS
  base_admissions AS (
    SELECT
      adm.subject_id,
      adm.hadm_id,
      adm.admittime,
      adm.dischtime,
      TIMESTAMP_DIFF(adm.dischtime, adm.admittime, SECOND) / 86400.0 AS los_days,
      CASE
        WHEN UPPER(adm.admission_type) IN ('EMERGENCY', 'URGENT', 'EW EMER.') THEN 1
        ELSE 0
      END AS via_ed,
      adm.hospital_expire_flag,
      adm.race,
      pat.gender,
      pat.anchor_age + (EXTRACT(YEAR FROM adm.admittime) - pat.anchor_year) AS age,
      pat.dod
    FROM `physionet-data.mimiciv_hosp.admissions` adm
    INNER JOIN hf_diagnoses hf ON adm.hadm_id = hf.hadm_id
    INNER JOIN `physionet-data.mimiciv_hosp.patients` pat ON adm.subject_id = pat.subject_id
    WHERE adm.hospital_expire_flag = 0  -- exclude in-hospital deaths
  ),

  -- 3. Pre-calculate list of unplanned admissions for readmission mapping
  unplanned_admissions AS (
    SELECT
      subject_id,
      hadm_id,
      admittime,
      CASE
        WHEN UPPER(admission_type) IN ('EMERGENCY', 'URGENT', 'EW EMER.') THEN 1
        ELSE 0
      END AS via_ed
    FROM `physionet-data.mimiciv_hosp.admissions`
    WHERE UPPER(admission_type) NOT IN ('ELECTIVE', 'SCHEDULED')
  ),

  -- 4. Calculate 30-day unplanned readmissions and prior counts
  outcomes_and_priors AS (
    SELECT
      b.hadm_id,
      -- Unplanned readmission within 30 days
      COALESCE(
        (
          SELECT 1
          FROM unplanned_admissions u
          WHERE u.subject_id = b.subject_id
            AND u.admittime > b.dischtime
            AND u.admittime <= TIMESTAMP_ADD(b.dischtime, INTERVAL 30 DAY)
          ORDER BY u.admittime ASC
          LIMIT 1
        ), 0
      ) AS readmitted_30d,
      -- Competing event: death within 30 days without readmission
      CASE
        WHEN b.dod IS NOT NULL
             AND b.dod <= TIMESTAMP_ADD(b.dischtime, INTERVAL 30 DAY)
             AND NOT EXISTS (
               SELECT 1 FROM unplanned_admissions u
               WHERE u.subject_id = b.subject_id
                 AND u.admittime > b.dischtime
                 AND u.admittime <= TIMESTAMP_ADD(b.dischtime, INTERVAL 30 DAY)
             ) THEN 1
        ELSE 0
      END AS competing_event,
      -- Prior visits in past 12 months
      (
        SELECT COUNT(*)
        FROM `physionet-data.mimiciv_hosp.admissions` p
        WHERE p.subject_id = b.subject_id
          AND p.admittime >= TIMESTAMP_SUB(b.admittime, INTERVAL 365 DAY)
          AND p.admittime < b.admittime
      ) AS prior_admits_12m,
      -- Prior ED visits in past 6 months
      (
        SELECT COUNT(*)
        FROM unplanned_admissions u
        WHERE u.subject_id = b.subject_id
          AND u.via_ed = 1
          AND u.admittime >= TIMESTAMP_SUB(b.admittime, INTERVAL 180 DAY)
          AND u.admittime < b.admittime
      ) AS ed_visits_6m
    FROM base_admissions b
  ),

  -- 5. Extract latest in-window lab value (within 72 hours of discharge)
  last_labs AS (
    SELECT
      hadm_id,
      MAX(CASE WHEN itemid = 50912 THEN valuenum END) AS lab_creatinine,
      MAX(CASE WHEN itemid = 50983 THEN valuenum END) AS lab_sodium,
      MAX(CASE WHEN itemid = 51222 THEN valuenum END) AS lab_hemoglobin,
      MAX(CASE WHEN itemid = 51002 THEN valuenum END) AS lab_bnp
    FROM (
      SELECT
        le.hadm_id,
        le.itemid,
        le.valuenum,
        ROW_NUMBER() OVER (
          PARTITION BY le.hadm_id, le.itemid
          ORDER BY TIMESTAMP_DIFF(b.dischtime, le.charttime, SECOND) ASC
        ) AS rn
      FROM `physionet-data.mimiciv_hosp.labevents` le
      INNER JOIN base_admissions b ON le.hadm_id = b.hadm_id
      WHERE le.itemid IN (50912, 50983, 51222, 51002)
        AND le.valuenum IS NOT NULL
        AND le.charttime >= TIMESTAMP_SUB(b.dischtime, INTERVAL 72 HOUR)
        AND le.charttime <= b.dischtime
    )
    WHERE rn = 1
    GROUP BY hadm_id
  ),

  -- 6. Extract vital aggregates (within 48 hours of discharge)
  vital_features AS (
    SELECT
      hadm_id,
      AVG(CASE WHEN itemid = 220045 THEN valuenum END) AS vital_mean_heart_rate,
      MIN(CASE WHEN itemid = 220045 THEN valuenum END) AS vital_min_heart_rate,
      MAX(CASE WHEN itemid = 220045 THEN valuenum END) AS vital_max_heart_rate,
      AVG(CASE WHEN itemid = 220179 THEN valuenum END) AS vital_mean_sbp,
      MIN(CASE WHEN itemid = 220179 THEN valuenum END) AS vital_min_sbp,
      MAX(CASE WHEN itemid = 220179 THEN valuenum END) AS vital_max_sbp,
      AVG(CASE WHEN itemid = 223761 THEN valuenum END) AS vital_mean_temperature,
      MIN(CASE WHEN itemid = 223761 THEN valuenum END) AS vital_min_temperature,
      MAX(CASE WHEN itemid = 223761 THEN valuenum END) AS vital_max_temperature
    FROM `physionet-data.mimiciv_icu.chartevents` ce
    INNER JOIN base_admissions b ON ce.hadm_id = b.hadm_id
    WHERE ce.itemid IN (220045, 220179, 223761)
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >= TIMESTAMP_SUB(b.dischtime, INTERVAL 48 HOUR)
      AND ce.charttime <= b.dischtime
    GROUP BY hadm_id
  ),

  -- 7. Attach clinical procedure indicator (HOSPITAL score calculation)
  procedures AS (
    SELECT DISTINCT hadm_id, 1 AS has_procedure
    FROM `physionet-data.mimiciv_hosp.procedures_icd`
  )

-- Final Selection & Feature Assembly
SELECT
  b.hadm_id,
  b.subject_id,
  b.admittime,
  b.dischtime,
  b.dod,
  b.los_days,
  b.via_ed,
  b.age,
  CASE WHEN UPPER(b.gender) = 'M' THEN 1 ELSE 0 END AS is_male,
  -- Race Dummies
  CASE WHEN REGEXP_CONTAINS(UPPER(b.race), 'WHITE') THEN 1 ELSE 0 END AS race_white,
  CASE WHEN REGEXP_CONTAINS(UPPER(b.race), 'BLACK') THEN 1 ELSE 0 END AS race_black,
  CASE WHEN REGEXP_CONTAINS(UPPER(b.race), 'HISPANIC') OR REGEXP_CONTAINS(UPPER(b.race), 'LATINO') THEN 1 ELSE 0 END AS race_hispanic,
  CASE WHEN REGEXP_CONTAINS(UPPER(b.race), 'ASIAN') THEN 1 ELSE 0 END AS race_asian,
  CASE WHEN NOT REGEXP_CONTAINS(UPPER(b.race), 'WHITE|BLACK|HISPANIC|LATINO|ASIAN') THEN 1 ELSE 0 END AS race_other,
  
  -- Prior Visit counts & outcomes
  o.readmitted_30d,
  o.competing_event,
  o.prior_admits_12m,
  o.ed_visits_6m,
  COALESCE(p.has_procedure, 0) AS has_procedure,

  -- Lab Features
  COALESCE(l.lab_creatinine, 0.0) AS lab_creatinine,
  COALESCE(l.lab_sodium, 0.0) AS lab_sodium,
  COALESCE(l.lab_hemoglobin, 0.0) AS lab_hemoglobin,
  COALESCE(l.lab_bnp, 0.0) AS lab_bnp,

  -- Vital Features
  v.vital_mean_heart_rate,
  v.vital_min_heart_rate,
  v.vital_max_heart_rate,
  v.vital_mean_sbp,
  v.vital_min_sbp,
  v.vital_max_sbp,
  v.vital_mean_temperature,
  v.vital_min_temperature,
  v.vital_max_temperature

FROM base_admissions b
INNER JOIN outcomes_and_priors o ON b.hadm_id = o.hadm_id
LEFT JOIN last_labs l ON b.hadm_id = l.hadm_id
LEFT JOIN vital_features v ON b.hadm_id = v.hadm_id
LEFT JOIN procedures p ON b.hadm_id = p.hadm_id
ORDER BY b.subject_id, b.admittime;
