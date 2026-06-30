from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class RFModel(BaseModel):
    name = "RandomForest"

    def __init__(self, **kwargs) -> None:
        defaults = dict(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        defaults.update(kwargs)
        self.clf = RandomForestClassifier(**defaults)
        self.params = defaults

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.clf.fit(X_train, y_train)

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
    def load(cls, path: str | Path) -> "RFModel":
        with open(path, "rb") as fp:
            return pickle.load(fp)
