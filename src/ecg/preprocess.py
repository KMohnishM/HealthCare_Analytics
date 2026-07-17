"""
src/ecg/preprocess.py
----------------------
Preprocessing pipeline for MIMIC-IV-ECG 12-lead waveforms.

Steps:
  1. Identify ECG record closest to each patient's discharge
     within a configurable window (default 72h).
  2. Load waveform via WFDB.
  3. Bandpass filter (0.5–40 Hz) to remove baseline wander and EMG noise.
  4. Resample to 500 Hz if necessary.
  5. Trim or zero-pad to exactly target_len samples (default 5000).
  6. Z-score normalize per lead.

Usage:
    from src.ecg.preprocess import build_ecg_index, load_and_preprocess_ecg
    ecg_index = build_ecg_index(cfg)   # DataFrame of hadm_id -> wfdb record path
    signal = load_and_preprocess_ecg(record_path, cfg)  # (12, 5000) float32
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.signal import butter, filtfilt

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── ECG index: map admissions to nearest ECG record ──────────────────────────

def build_ecg_index(
    cohort: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """
    For each admission in the cohort, find the ECG record file closest
    to discharge within the configured window.

    MIMIC-IV-ECG directory structure:
        <mimic_ecg_dir>/files/<p_prefix>/<subject_id>/<study_id>/<record>.hea

    The ``record_list.csv`` file (shipped with MIMIC-IV-ECG) contains
    columns: subject_id, study_id, ecg_time, filename (relative path).

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort with ``subject_id``, ``hadm_id``, ``dischtime``.
    cfg : DictConfig
        Project configuration.

    Returns
    -------
    pd.DataFrame
        Columns: hadm_id, ecg_record_path, ecg_time, hours_before_discharge.
        Only rows where a qualifying ECG was found are included.
    """
    ecg_dir = Path(cfg.paths.mimic_ecg_dir)
    window_h = cfg.cohort.ecg_window_hours

    record_list_path = ecg_dir / "record_list.csv"
    if not record_list_path.exists():
        # Try machine.csv which is used in some MIMIC-IV-ECG versions
        record_list_path = ecg_dir / "machine_measurements.csv"
    if not record_list_path.exists():
        raise FileNotFoundError(
            f"Could not find ECG record list at {ecg_dir}. "
            "Expected 'record_list.csv' or 'machine_measurements.csv'."
        )

    log.info("Loading ECG record list from %s ...", record_list_path)
    ecg_records = pd.read_csv(record_list_path, low_memory=False)
    ecg_records["ecg_time"] = pd.to_datetime(ecg_records["ecg_time"], errors="coerce")
    ecg_records = ecg_records.dropna(subset=["ecg_time"])

    cohort_clean = cohort[["subject_id", "hadm_id", "dischtime"]].copy()
    cohort_clean["dischtime"] = pd.to_datetime(cohort_clean["dischtime"])

    merged = cohort_clean.merge(ecg_records, on="subject_id", how="left")

    # Keep only ECGs within the window before discharge
    merged["hours_before_discharge"] = (
        merged["dischtime"] - merged["ecg_time"]
    ).dt.total_seconds() / 3600

    in_window = merged[
        (merged["hours_before_discharge"] >= 0)
        & (merged["hours_before_discharge"] <= window_h)
    ].copy()

    # Keep closest ECG per admission
    in_window = in_window.sort_values("hours_before_discharge")
    closest = in_window.groupby("hadm_id").first().reset_index()

    # Build absolute path to WFDB record (without extension)
    ecg_root = ecg_dir / "files"

    def build_path(row: pd.Series) -> str:
        # MIMIC-IV-ECG filename field is the relative path including the .hea extension
        # We strip the extension as WFDB rdrecord expects path without extension
        rel = str(row.get("filename", "")).replace(".hea", "")
        return str(ecg_root / rel)

    closest["ecg_record_path"] = closest.apply(build_path, axis=1)

    coverage = 100 * len(closest) / len(cohort)
    log.info(
        "ECG index built: %d/%d admissions have a qualifying ECG (%.1f%%)",
        len(closest), len(cohort), coverage,
    )
    return closest[["hadm_id", "ecg_record_path", "ecg_time", "hours_before_discharge"]]


# ── Waveform loading & preprocessing ─────────────────────────────────────────

def load_and_preprocess_ecg(
    record_path: str,
    cfg: DictConfig,
) -> Optional[np.ndarray]:
    """
    Load a WFDB ECG record, filter, resample, pad/trim, and normalize.

    Parameters
    ----------
    record_path : str
        Absolute path to the WFDB record (without file extension).
    cfg : DictConfig
        Project configuration (reads ``cfg.ecg``).

    Returns
    -------
    np.ndarray of shape (12, target_len) and dtype float32, or None on error.
    """
    try:
        import wfdb

        rec = wfdb.rdrecord(record_path)
    except Exception as e:
        log.warning("Failed to load ECG record %s: %s", record_path, e)
        return None

    sig = rec.p_signal  # (N_samples, 12) float64

    if sig is None or sig.shape[1] < 12:
        log.warning("Record %s has <12 leads, skipping.", record_path)
        return None

    sig = sig[:, :12].T.astype(np.float32)  # -> (12, N)

    fs    = rec.fs
    target_fs  = cfg.ecg.sample_rate   # 500 Hz
    target_len = cfg.ecg.target_len    # 5000

    # ── Bandpass filter 0.5–40 Hz ───────────────────────────────────────────
    b, a = butter(3, [0.5, 40.0], btype="band", fs=fs)
    sig = filtfilt(b, a, sig, axis=1)

    # ── Resample to target sampling rate ────────────────────────────────────
    if fs != target_fs:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(target_fs), int(fs))
        sig = resample_poly(sig, int(target_fs) // g, int(fs) // g, axis=1)

    sig = sig.astype(np.float32)

    # ── Trim or zero-pad to target_len ─────────────────────────────────────
    n = sig.shape[1]
    if n >= target_len:
        sig = sig[:, :target_len]
    else:
        pad = np.zeros((12, target_len - n), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)

    # ── Z-score normalize per lead ─────────────────────────────────────────
    mean = sig.mean(axis=1, keepdims=True)
    std  = sig.std(axis=1,  keepdims=True) + 1e-8
    sig  = (sig - mean) / std

    return sig  # (12, 5000) float32
