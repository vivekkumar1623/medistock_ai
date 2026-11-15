"""Baselines.

A baseline is not a formality. It is the number a reviewer subtracts from your
model's error to decide whether the ML earned its complexity. Both baselines here
are computable from origin-time features only.

``seasonal_naive_30d``
    Predict that the next 30 days repeat the last 30 days exactly:
    ``y_hat = sum_30``. For a 30-day aggregate this is a genuinely strong
    baseline -- it absorbs level, trend and seasonality to first order.

``moving_average_90d``
    Predict the trailing 90-day average rate applied to 30 days:
    ``y_hat = mean_90 * 30``. Smoother, slower to react.

Neither has parameters to fit, so neither can overfit the validation period.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class BaselineForecaster(Protocol):
    name: str

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class SeasonalNaive30:
    """Next 30 days = the last 30 days."""

    name = "seasonal_naive_30d"

    def __init__(self, window_column: str = "sum_30", horizon_days: int = 30) -> None:
        self.window_column = window_column
        self.horizon_days = horizon_days

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.window_column not in features.columns:
            raise KeyError(f"{self.name} requires the '{self.window_column}' feature")
        return np.clip(features[self.window_column].to_numpy(dtype=float), 0.0, None)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "rule": f"forecast = {self.window_column} (demand repeats the last 30 days)",
            "fitted_parameters": 0,
        }


class MovingAverage90:
    """Next 30 days = trailing 90-day daily rate x 30."""

    name = "moving_average_90d"

    def __init__(self, window_column: str = "mean_90", horizon_days: int = 30) -> None:
        self.window_column = window_column
        self.horizon_days = horizon_days

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.window_column not in features.columns:
            raise KeyError(f"{self.name} requires the '{self.window_column}' feature")
        return np.clip(
            features[self.window_column].to_numpy(dtype=float) * self.horizon_days, 0.0, None
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "rule": f"forecast = {self.window_column} * {self.horizon_days}",
            "fitted_parameters": 0,
        }
