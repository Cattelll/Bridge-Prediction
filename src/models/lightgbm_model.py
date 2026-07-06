from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

from .base import BaseModel


class LGBMModel(BaseModel):
    name = "LightGBM"

    def __init__(self, **kwargs) -> None:
        defaults = dict(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(kwargs)
        self.clf = LGBMClassifier(**defaults)
        self.params = defaults

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            sample_weight: np.ndarray | None = None) -> None:
        self.clf.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.clf.feature_importances_

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, path: str | Path) -> "LGBMModel":
        with open(path, "rb") as fp:
            return pickle.load(fp)
