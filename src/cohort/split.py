"""
src/cohort/split.py
--------------------
Subject-level stratified train / val / test split.

Splitting by subject_id (not hadm_id) ensures no patient's data
appears in more than one partition — critical for preventing leakage
in a clinical prediction setting.

Usage:
    from src.cohort.split import split_cohort
    train_df, val_df, test_df = split_cohort(cohort, cfg)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

log = get_logger(__name__)


def split_cohort(
    cohort: pd.DataFrame,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a subject-level stratified train / val / test split.

    Stratification is on the 30-day readmission label at the patient level
    (a patient is "positive" if they have any readmitted admission).

    Parameters
    ----------
    cohort : pd.DataFrame
        Output of ``build_cohort``.  Must contain ``subject_id`` and
        ``readmitted_30d`` columns.
    cfg : DictConfig
        Loaded project config (reads ``cfg.cohort`` section).

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Cohort subsets with no shared patients.
    """
    seed       = cfg.cohort.random_seed
    train_frac = cfg.cohort.train_frac
    val_frac   = cfg.cohort.val_frac
    test_frac  = cfg.cohort.test_frac

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "train_frac + val_frac + test_frac must sum to 1.0"

    # One row per subject — label = 1 if any admission was readmitted
    subjects = (
        cohort.groupby("subject_id")["readmitted_30d"]
        .max()
        .reset_index()
        .rename(columns={"readmitted_30d": "label"})
    )

    n_pos = subjects["label"].sum()
    n_neg = len(subjects) - n_pos
    log.info(
        "Splitting %d unique subjects (pos=%d, neg=%d) into "
        "%.0f/%.0f/%.0f%%",
        len(subjects), n_pos, n_neg,
        100 * train_frac, 100 * val_frac, 100 * test_frac,
    )

    # Determine if stratification is possible (requires at least 2 samples per class)
    class_counts = subjects["label"].value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    if not can_stratify:
        log.warning(
            "  Class count for least populated class is too small (min count: %s). "
            "  Falling back to non-stratified splitting.",
            class_counts.min() if len(class_counts) > 0 else 0
        )

    # First split: train vs (val + test)
    train_subjects, temp_subjects = train_test_split(
        subjects,
        test_size=(val_frac + test_frac),
        stratify=subjects["label"] if can_stratify else None,
        random_state=seed,
    )

    # Determine if stratification is possible for second split
    temp_class_counts = temp_subjects["label"].value_counts()
    can_stratify_temp = len(temp_class_counts) > 1 and temp_class_counts.min() >= 2

    # Second split: val vs test from the temp set
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=(test_frac / (val_frac + test_frac)),
        stratify=temp_subjects["label"] if can_stratify_temp else None,
        random_state=seed,
    )

    train_ids = set(train_subjects["subject_id"])
    val_ids   = set(val_subjects["subject_id"])
    test_ids  = set(test_subjects["subject_id"])

    # Verify no overlap
    assert len(train_ids & val_ids) == 0, "Train/val overlap detected!"
    assert len(train_ids & test_ids) == 0, "Train/test overlap detected!"
    assert len(val_ids & test_ids) == 0, "Val/test overlap detected!"

    train_df = cohort[cohort["subject_id"].isin(train_ids)].copy()
    val_df   = cohort[cohort["subject_id"].isin(val_ids)].copy()
    test_df  = cohort[cohort["subject_id"].isin(test_ids)].copy()

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = df["readmitted_30d"].mean()
        log.info(
            "  %-5s : %5d admissions | %4d subjects | readmit rate %.1f%%",
            split_name, len(df), df["subject_id"].nunique(), 100 * rate,
        )

    # Persist splits
    out_dir = Path(cfg.paths.cohort_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet",     index=False)
    test_df.to_parquet(out_dir / "test.parquet",   index=False)
    log.info("Saved train/val/test splits -> %s", out_dir)

    return train_df, val_df, test_df
