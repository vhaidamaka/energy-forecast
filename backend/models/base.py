"""
Abstract base predictor interface. All models implement this.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Callable
from dataclasses import dataclass


@dataclass
class PredictionResult:
    mae: float
    rmse: float
    mape: float
    forecast: list[dict]          # [{timestamp, predicted}]  — future horizon
    actual: list[dict]            # [{timestamp, actual}]      — test set ground truth
    test_predicted: list[dict]    # [{timestamp, predicted}]  — test set model output
    training_history: list[dict] | None = None  # [{epoch, loss, val_loss}]


LogCallback = Callable[[str], None]


class BasePredictor(ABC):

    def __init__(self, hyperparams: dict, log_callback: LogCallback | None = None):
        self.hyperparams = hyperparams
        self._log = log_callback or print

    def log(self, msg: str):
        self._log(msg)

    @abstractmethod
    def fit_predict(self, df: pd.DataFrame, horizon_days: int) -> PredictionResult:
        """
        Train on df and predict horizon_days into the future.
        df must have a DatetimeIndex and an 'energy' column.
        """
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def calc_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
        """Returns (MAE, RMSE, MAPE)."""
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        # Avoid division by zero
        mask = actual != 0
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 0.0
        return mae, rmse, mape

    @staticmethod
    def series_to_records(timestamps, values, key: str) -> list[dict]:
        return [{"timestamp": str(ts), key: round(float(v), 4)} for ts, v in zip(timestamps, values)]
