"""
src/tabular/impute.py
----------------------
Median imputation pipeline for the tabular feature matrix.

Uses sklearn's SimpleImputer so that imputation statistics are
fitted on training data only and applied consistently to val/test
— preventing data leakage.

Usage:
    from src.tabular.impute import fit_imputer, apply_imputer
    imputer = fit_imputer(X_train)
    X_train_imp = apply_imputer(imputer, X_train)
    X_test_imp  = apply_imputer(imputer, X_test)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger

log = get_logger(__name__)


def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    """
    Fit a median imputer on the training feature matrix.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (may contain NaN).

    Returns
    -------
    SimpleImputer
        Fitted imputer ready to transform val/test sets.
    """
    log.info(
        "Fitting median imputer on %d rows × %d cols "
        "(missing: %.1f%%)",
        len(X_train), X_train.shape[1],
        100 * X_train.isna().mean().mean(),
    )
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train.values)
    return imputer


def apply_imputer(
    imputer: SimpleImputer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply a fitted imputer and return a DataFrame preserving column names.

    Parameters
    ----------
    imputer : SimpleImputer
        A fitted SimpleImputer instance.
    X : pd.DataFrame
        Feature matrix to transform (same columns as training set).

    Returns
    -------
    pd.DataFrame
        Imputed feature matrix (no NaN values).
    """
    arr = imputer.transform(X.values)
    return pd.DataFrame(arr, index=X.index, columns=X.columns)


def save_imputer(imputer: SimpleImputer, path: str | Path) -> None:
    """Persist imputer to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, path)
    log.info("Imputer saved -> %s", path)


def load_imputer(path: str | Path) -> SimpleImputer:
    """Load a previously saved imputer from disk."""
    return joblib.load(path)


def missingness_report(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a per-feature missingness report.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (pre-imputation).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [feature, n_missing, pct_missing]
        sorted by pct_missing descending.
    """
    n_missing = X.isna().sum()
    pct_missing = 100 * n_missing / len(X)
    report = pd.DataFrame({
        "feature":     n_missing.index,
        "n_missing":   n_missing.values,
        "pct_missing": pct_missing.values,
    }).sort_values("pct_missing", ascending=False).reset_index(drop=True)
    return report
