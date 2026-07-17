"""
src/cxr/preprocess.py
----------------------
Preprocessing pipeline for MIMIC-CXR chest radiographs.

Steps:
  1. Build index: for each admission, find the CXR (frontal view only)
     closest to discharge within the configured window.
  2. Load JPEG image from MIMIC-CXR-JPG.
  3. Convert to RGB, resize to 224×224, apply ImageNet normalization.

Usage:
    from src.cxr.preprocess import build_cxr_index, load_and_preprocess_cxr
    cxr_index = build_cxr_index(cohort, cfg)
    tensor = load_and_preprocess_cxr(image_path, is_train=False)  # (3, 224, 224)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torchvision import transforms

from src.utils.logger import get_logger

log = get_logger(__name__)

# ── ImageNet normalization constants ─────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transform(is_train: bool, image_size: int = 224) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline.

    Parameters
    ----------
    is_train : bool
        If True, includes data augmentation (flip, jitter, random crop).
    image_size : int
        Target image size (square).

    Returns
    -------
    transforms.Compose
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


# ── CXR index: map admissions to nearest frontal CXR ─────────────────────────

def build_cxr_index(
    cohort: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """
    For each admission, find the frontal CXR closest to discharge within window.

    MIMIC-CXR-JPG metadata files used:
    - ``mimic-cxr-2.0.0-metadata.csv.gz``  -> study_id, dicom_id, ViewPosition, StudyDate, StudyTime
    - ``mimic-cxr-2.0.0-split.csv.gz``     -> (ignored, we use our own split)

    Image path pattern:
    ``<mimic_cxr_dir>/files/p<subject_prefix>/p<subject_id>/s<study_id>/<dicom_id>.jpg``

    Parameters
    ----------
    cohort : pd.DataFrame
        Cohort with ``subject_id``, ``hadm_id``, ``dischtime``.
    cfg : DictConfig
        Project configuration.

    Returns
    -------
    pd.DataFrame
        Columns: hadm_id, cxr_path, study_datetime, hours_before_discharge.
    """
    cxr_dir = Path(cfg.paths.mimic_cxr_dir)
    window_h = cfg.cohort.cxr_window_hours

    meta_path = cxr_dir / "mimic-cxr-2.0.0-metadata.csv.gz"
    if not meta_path.exists():
        meta_path = cxr_dir / "mimic-cxr-2.0.0-metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"CXR metadata not found at {cxr_dir}. "
            "Expected 'mimic-cxr-2.0.0-metadata.csv.gz'."
        )

    log.info("Loading CXR metadata from %s ...", meta_path.name)
    meta = pd.read_csv(meta_path, low_memory=False)

    # Keep only frontal views (PA or AP)
    frontal = meta[meta["ViewPosition"].isin(["PA", "AP"])].copy()

    # Parse study datetime from StudyDate (YYYYMMDD) + StudyTime (HHMMSS.fff)
    frontal["study_datetime"] = pd.to_datetime(
        frontal["StudyDate"].astype(str) + " " + frontal["StudyTime"].astype(str),
        format="%Y%m%d %H%M%S.%f",
        errors="coerce",
    )
    # Fallback: date only
    mask_failed = frontal["study_datetime"].isna()
    frontal.loc[mask_failed, "study_datetime"] = pd.to_datetime(
        frontal.loc[mask_failed, "StudyDate"].astype(str),
        format="%Y%m%d", errors="coerce",
    )
    frontal = frontal.dropna(subset=["study_datetime"])

    cohort_clean = cohort[["subject_id", "hadm_id", "dischtime"]].copy()
    cohort_clean["dischtime"] = pd.to_datetime(cohort_clean["dischtime"])

    merged = cohort_clean.merge(frontal, on="subject_id", how="left")

    merged["hours_before_discharge"] = (
        merged["dischtime"] - merged["study_datetime"]
    ).dt.total_seconds() / 3600

    in_window = merged[
        (merged["hours_before_discharge"] >= 0)
        & (merged["hours_before_discharge"] <= window_h)
    ].copy()

    in_window = in_window.sort_values("hours_before_discharge")
    closest = in_window.groupby("hadm_id").first().reset_index()

    # Build absolute JPEG path
    files_root = cxr_dir / "files"

    def build_cxr_path(row: pd.Series) -> str:
        sid = str(int(row["subject_id"]))
        prefix = "p" + sid[:2]
        study = "s" + str(int(row["study_id"]))
        dicom = str(row["dicom_id"]) + ".jpg"
        return str(files_root / ("p" + prefix[1:]) / ("p" + sid) / study / dicom)

    closest["cxr_path"] = closest.apply(build_cxr_path, axis=1)

    coverage = 100 * len(closest) / len(cohort)
    log.info(
        "CXR index built: %d/%d admissions have a qualifying CXR (%.1f%%)",
        len(closest), len(cohort), coverage,
    )
    return closest[["hadm_id", "cxr_path", "study_datetime", "hours_before_discharge"]]


def load_and_preprocess_cxr(
    image_path: str,
    is_train: bool = False,
    image_size: int = 224,
) -> Optional[torch.Tensor]:
    """
    Load a JPEG CXR image and apply the preprocessing transform.

    Parameters
    ----------
    image_path : str
        Absolute path to the JPEG image file.
    is_train : bool
        If True, applies augmentation transforms.
    image_size : int
        Target image size.

    Returns
    -------
    torch.Tensor of shape (3, image_size, image_size), or None on error.
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        transform = get_transform(is_train=is_train, image_size=image_size)
        return transform(img)
    except Exception as e:
        log.warning("Failed to load CXR %s: %s", image_path, e)
        return None
