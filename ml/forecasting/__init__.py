"""Forecasting package: baselines, pipelines, splits, training."""

from __future__ import annotations

from ml.forecasting.baseline import BaselineForecaster, MovingAverage90, SeasonalNaive30
from ml.forecasting.pipeline import CANDIDATE_BUILDERS, CANDIDATE_GRIDS, build_preprocessor
from ml.forecasting.splits import (
    RollingFold,
    chronological_split,
    rolling_origin_folds,
    subset_by_origins,
)

__all__ = [
    "CANDIDATE_BUILDERS",
    "CANDIDATE_GRIDS",
    "BaselineForecaster",
    "MovingAverage90",
    "RollingFold",
    "SeasonalNaive30",
    "build_preprocessor",
    "chronological_split",
    "rolling_origin_folds",
    "subset_by_origins",
]
