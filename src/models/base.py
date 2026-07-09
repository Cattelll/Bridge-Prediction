"""Base interface that all model wrappers implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Common contract for RFModel, XGBModel, and LGBMModel.

    Keeping a shared interface lets scripts/notebooks train, save,
    load, and evaluate the three algorithms interchangeably.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel": ...

    def feature_importances(self) -> np.ndarray | None:
        return None
