"""
src/tabular/model.py
---------------------
XGBoost bootstrap ensemble for tabular HF readmission prediction.

The ensemble trains N=20 XGBoost models on bootstrap resamples of
the training set. At inference, the mean prediction is the risk score
and the standard deviation is used to derive a confidence estimate.

Usage:
    from src.tabular.model import TabularEnsemble
    model = TabularEnsemble(cfg)
    model.fit(X_train, y_train, X_val, y_val)
    result = model.predict(X_test)
    # result['score']      -> np.ndarray of risk probabilities
    # result['confidence'] -> np.ndarray in [0, 1]
    # result['std']        -> np.ndarray of prediction std
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.utils import resample

from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


class TabularEnsemble:
    """
    Bootstrap ensemble of XGBoost classifiers.

    Attributes
    ----------
    models : list of XGBClassifier
        Fitted base models.
    cfg : DictConfig
        Project configuration.
    feature_names : list of str
        Names of input features (set during fit).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.models: list[xgb.XGBClassifier] = []
        self.feature_names: list[str] = []

    def _make_base_model(
        self,
        scale_pos_weight: float,
        seed: int,
    ) -> xgb.XGBClassifier:
        """Instantiate a single XGBoost classifier with project hyperparameters."""
        xgb_cfg = self.cfg.tabular.xgb
        return xgb.XGBClassifier(
            n_estimators      = int(xgb_cfg.n_estimators),
            max_depth         = int(xgb_cfg.max_depth),
            learning_rate     = float(xgb_cfg.learning_rate),
            subsample         = float(xgb_cfg.subsample),
            colsample_bytree  = float(xgb_cfg.colsample_bytree),
            scale_pos_weight  = scale_pos_weight,
            eval_metric       = str(xgb_cfg.eval_metric),
            early_stopping_rounds = int(xgb_cfg.early_stopping),
            random_state      = seed,
            use_label_encoder = False,
            verbosity         = 0,
            device            = "cpu",   # Colab CPU-only
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "TabularEnsemble":
        """
        Train bootstrap ensemble on training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Imputed training feature matrix.
        y_train : pd.Series
            Binary readmission labels (0/1).
        X_val : pd.DataFrame
            Validation feature matrix (for early stopping).
        y_val : pd.Series
            Validation labels.

        Returns
        -------
        self
        """
        set_seed(int(self.cfg.tabular.xgb.random_seed))
        self.feature_names = X_train.columns.tolist()

        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = n_neg / max(n_pos, 1)
        log.info(
            "Training XGBoost ensemble (N=%d bootstraps) | "
            "pos=%d, neg=%d, scale_pos_weight=%.2f",
            int(self.cfg.tabular.xgb.n_bootstrap),
            n_pos, n_neg, scale_pos_weight,
        )

        X_arr  = X_train.values
        y_arr  = y_train.values
        Xv_arr = X_val.values
        yv_arr = y_val.values

        n_bootstrap = int(self.cfg.tabular.xgb.n_bootstrap)
        self.models = []

        for i in range(n_bootstrap):
            seed_i = int(self.cfg.tabular.xgb.random_seed) + i
            X_bs, y_bs = resample(X_arr, y_arr, random_state=seed_i)

            m = self._make_base_model(scale_pos_weight, seed_i)
            m.fit(
                X_bs, y_bs,
                eval_set=[(Xv_arr, yv_arr)],
                verbose=False,
            )
            self.models.append(m)

        # Compute and save a static global_max_std on the validation set for inference normalization
        val_preds = np.stack(
            [m.predict_proba(Xv_arr)[:, 1] for m in self.models],
            axis=1
        )
        self.global_max_std = max(float(val_preds.std(axis=1).max()), 1e-4)
        log.info("  Static global_max_std computed on validation set: %.6f", self.global_max_std)

        log.info("Ensemble training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate risk scores and confidence estimates.

        Parameters
        ----------
        X : pd.DataFrame
            Imputed feature matrix.

        Returns
        -------
        dict with keys:
            - ``score``      : mean predicted probability (N,)
            - ``std``        : std of predictions across bootstrap models (N,)
            - ``confidence`` : 1 − normalized_std, clamped to [0, 1] (N,)
            - ``all_preds``  : raw predictions from each model (N, n_bootstrap)
        """
        if not self.models:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        preds = np.stack(
            [m.predict_proba(X.values)[:, 1] for m in self.models],
            axis=1,
        )  # (N, n_bootstrap)

        mean_pred = preds.mean(axis=1)
        std_pred  = preds.std(axis=1)

        # Normalize std using static global_max_std to support N=1 clinical deployment
        global_max_std = getattr(self, "global_max_std", 0.5)
        confidence = np.clip(1.0 - (std_pred / global_max_std), 0.0, 1.0)

        return {
            "score":      mean_pred,
            "std":        std_pred,
            "confidence": confidence,
            "all_preds":  preds,
        }

    def save(self, path: str | Path) -> None:
        """Persist ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Tabular ensemble saved -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TabularEnsemble":
        """Load a previously saved ensemble."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return average feature importance across all ensemble members.

        Returns
        -------
        pd.DataFrame
            Columns: [feature, importance], sorted descending.
        """
        importances = np.mean(
            [m.feature_importances_ for m in self.models], axis=0
        )
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
