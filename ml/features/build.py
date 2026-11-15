"""Leakage-safe feature engineering.

The central design decision: a prediction at origin ``T`` may read **only** days
``<= T``. That is enforced structurally, not by convention:

* The demand matrix is a plain ``(n_days, n_products)`` array with a prefix-sum
  accumulator. A feature is built from ``cumsum[end_pos + 1] - cumsum[start]``
  where ``end_pos`` is the origin's row. There is no API to read row
  ``end_pos + 1`` while building features -- the target uses a *separate*,
  clearly named function that is only called for training rows.
* ``build_training_frame`` computes the target from ``(end_pos, end_pos + h]``
  after the feature frame is already materialised, so a feature cannot
  accidentally reference it.
* Feature names are frozen into ``feature_spec.json`` and the model pipeline is
  fitted on exactly those columns.

Deliberate exclusion: **inventory levels are not model features.** We only hold an
inventory snapshot at one date, so using stock at earlier origins would be
reconstructing the future. Stock is a *decision input* consumed after
forecasting, not a demand driver. See DECISIONS.md D-011.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FEATURE_SPEC_VERSION = "fs-v1"
TARGET_TEMPLATE = "demand_next_{horizon}d"

WINDOWS: tuple[int, ...] = (7, 14, 30, 60, 90, 180)

NUMERIC_FEATURES: tuple[str, ...] = (
    # Demand level
    "sum_7",
    "sum_14",
    "sum_30",
    "sum_60",
    "sum_90",
    "sum_180",
    "mean_7",
    "mean_30",
    "mean_90",
    # Demand variability and shape
    "std_30",
    "std_90",
    "cv_90",
    "max_daily_90",
    "ratio_7_30",
    "ratio_30_90",
    "trend_ratio",
    # Sparsity
    "zero_ratio_30",
    "zero_ratio_90",
    "days_since_last_sale",
    # Calendar (known in advance, therefore legitimate)
    "origin_month",
    "origin_quarter",
    "origin_day_of_year",
    "origin_is_q4",
    # Static commercial attributes
    "unit_price",
    "unit_cost",
    "margin_pct",
    "lead_time_days",
    "minimum_stock_level",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("category",)

ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class DemandMatrix:
    """Panel demand reshaped for fast, origin-bounded feature computation."""

    dates: pd.DatetimeIndex
    product_ids: np.ndarray
    values: np.ndarray  # (n_days, n_products) int64
    cumsum: np.ndarray  # (n_days + 1, n_products) int64, cumsum[0] == 0
    last_sale_pos: np.ndarray  # (n_days, n_products) latest position with qty > 0

    @classmethod
    def from_panel(cls, panel: pd.DataFrame) -> DemandMatrix:
        pivot = (
            panel.pivot_table(
                index="date",
                columns="product_id",
                values="quantity",
                aggfunc="sum",
                fill_value=0,
            )
            .sort_index()
            .sort_index(axis=1)
        )
        dates = pd.DatetimeIndex(pivot.index)
        product_ids = pivot.columns.to_numpy()
        values = pivot.to_numpy(dtype=np.int64)
        cumsum = np.vstack(
            [np.zeros((1, values.shape[1]), dtype=np.int64), np.cumsum(values, axis=0)]
        )

        positions = np.arange(values.shape[0], dtype=np.int64)[:, None]
        last_sale_pos = np.maximum.accumulate(np.where(values > 0, positions, -1), axis=0)
        return cls(
            dates=dates,
            product_ids=product_ids,
            values=values,
            cumsum=cumsum,
            last_sale_pos=last_sale_pos,
        )

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_products(self) -> int:
        return len(self.product_ids)

    def position_of(self, origin: pd.Timestamp) -> int:
        """Row index of ``origin``. Raises if the origin is outside the panel."""
        position = int(self.dates.searchsorted(origin.normalize(), side="right") - 1)
        if position < 0 or self.dates[position] != origin.normalize():
            raise ValueError(f"Origin {origin.date()} is not a date in the demand panel")
        return position

    def window_sum(self, end_pos: int, days: int) -> np.ndarray:
        """Sum over the ``days`` observations ending at ``end_pos`` (inclusive)."""
        start = max(0, end_pos - days + 1)
        return self.cumsum[end_pos + 1] - self.cumsum[start]

    def window_mean(self, end_pos: int, days: int) -> np.ndarray:
        span = min(days, end_pos + 1)
        return self.window_sum(end_pos, days) / max(span, 1)

    def window_std(self, end_pos: int, days: int) -> np.ndarray:
        start = max(0, end_pos - days + 1)
        block = self.values[start : end_pos + 1]
        return block.std(axis=0, ddof=0)

    def window_clip(self, end_pos: int, days: int) -> np.ndarray:
        return self.values[max(0, end_pos - days + 1) : end_pos + 1]


def compute_origin_features(
    matrix: DemandMatrix,
    products: pd.DataFrame,
    origin: pd.Timestamp,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """Features for every product at one origin, using only data <= origin.

    Returns a frame indexed by position, aligned to the matrix column order.
    """
    origin = pd.Timestamp(origin).normalize()
    end_pos = matrix.position_of(origin)

    # Hard structural guard: the slice below can never extend past the origin.
    assert matrix.dates[end_pos] == origin
    history = matrix.values[: end_pos + 1]
    assert history.shape[0] == end_pos + 1

    if end_pos + 1 < min(lookback_days, 30):
        raise ValueError(
            f"Origin {origin.date()} has only {end_pos + 1} days of history; "
            f"need at least 30 for a stable feature row"
        )

    eps = 1e-6
    sums = {window: matrix.window_sum(end_pos, window) for window in WINDOWS}
    means = {window: matrix.window_mean(end_pos, window) for window in WINDOWS}

    std_30 = matrix.window_std(end_pos, 30)
    std_90 = matrix.window_std(end_pos, 90)
    block_90 = matrix.window_clip(end_pos, 90)
    block_30 = matrix.window_clip(end_pos, 30)

    last_sale = matrix.last_sale_pos[end_pos]
    days_since_last_sale = np.where(last_sale >= 0, end_pos - last_sale, end_pos + 1)

    origin_month = origin.month
    origin_quarter = origin.quarter

    features = pd.DataFrame(
        {
            "sum_7": sums[7],
            "sum_14": sums[14],
            "sum_30": sums[30],
            "sum_60": sums[60],
            "sum_90": sums[90],
            "sum_180": sums[180],
            "mean_7": means[7],
            "mean_30": means[30],
            "mean_90": means[90],
            "std_30": std_30,
            "std_90": std_90,
            "cv_90": std_90 / (means[90] + eps),
            "max_daily_90": block_90.max(axis=0),
            "ratio_7_30": means[7] / (means[30] + eps),
            "ratio_30_90": means[30] / (means[90] + eps),
            "trend_ratio": means[30] / (means[90] + eps),
            "zero_ratio_30": (block_30 == 0).mean(axis=0),
            "zero_ratio_90": (block_90 == 0).mean(axis=0),
            "days_since_last_sale": days_since_last_sale,
            "origin_month": origin_month,
            "origin_quarter": origin_quarter,
            "origin_day_of_year": origin.dayofyear,
            "origin_is_q4": int(origin_month >= 10),
        },
        index=pd.RangeIndex(matrix.n_products),
    )

    static = products.set_index("product_id").reindex(matrix.product_ids)
    features["unit_price"] = static["unit_price"].to_numpy()
    features["unit_cost"] = static["unit_cost"].to_numpy()
    features["margin_pct"] = (
        static["unit_price"].to_numpy() - static["unit_cost"].to_numpy()
    ) / np.maximum(static["unit_price"].to_numpy(), eps)
    features["lead_time_days"] = static["lead_time_days"].to_numpy()
    features["minimum_stock_level"] = static["minimum_stock_level"].to_numpy()
    features["category"] = static["category"].to_numpy()

    features["product_id"] = matrix.product_ids
    features["origin_date"] = origin

    # Column order is part of the contract.
    ordered = ["product_id", "origin_date", *ALL_FEATURES]
    return features[ordered]


def compute_target(
    matrix: DemandMatrix, origin: pd.Timestamp, horizon_days: int
) -> np.ndarray | None:
    """Total demand in ``(origin, origin + horizon]``; None when it is unobservable.

    Kept separate from ``compute_origin_features`` so that the training loop must
    explicitly ask for the future. Nothing in the feature path can call this.
    """
    origin = pd.Timestamp(origin).normalize()
    end_pos = matrix.position_of(origin)
    target_end = end_pos + horizon_days
    if target_end >= matrix.n_days:
        return None
    return matrix.cumsum[target_end + 1] - matrix.cumsum[end_pos + 1]


def origin_grid(start: str, end: str, step_days: int) -> list[pd.Timestamp]:
    """Regular origin dates, inclusive of both ends."""
    return list(pd.date_range(start=start, end=end, freq=f"{step_days}D"))


def build_training_frame(
    panel: pd.DataFrame,
    products: pd.DataFrame,
    origins: list[pd.Timestamp],
    horizon_days: int,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """One row per (product, origin) with features and the forward-looking target."""
    matrix = DemandMatrix.from_panel(panel)
    target_column = TARGET_TEMPLATE.format(horizon=horizon_days)
    frames: list[pd.DataFrame] = []

    for origin in origins:
        origin = pd.Timestamp(origin).normalize()
        if origin not in matrix.dates:
            continue
        target = compute_target(matrix, origin, horizon_days)
        if target is None:
            continue
        features = compute_origin_features(matrix, products, origin, lookback_days)
        features[target_column] = target
        features["target_start"] = origin + pd.Timedelta(days=1)
        features["target_end"] = origin + pd.Timedelta(days=horizon_days)
        frames.append(features)

    if not frames:
        raise ValueError(
            "No usable origins: check the origin grid and horizon against the data range"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["origin_date"] = pd.to_datetime(combined["origin_date"])
    combined["target_start"] = pd.to_datetime(combined["target_start"])
    combined["target_end"] = pd.to_datetime(combined["target_end"])
    return combined.sort_values(["origin_date", "product_id"]).reset_index(drop=True)


def build_feature_spec(horizon_days: int, lookback_days: int) -> dict[str, Any]:
    """Machine-readable feature contract, written into the artifact bundle."""
    return {
        "feature_version": FEATURE_SPEC_VERSION,
        "target": TARGET_TEMPLATE.format(horizon=horizon_days),
        "horizon_days": horizon_days,
        "lookback_days": lookback_days,
        "features": [
            {
                "name": name,
                "dtype": "category" if name in CATEGORICAL_FEATURES else "float64",
                "role": "categorical" if name in CATEGORICAL_FEATURES else "numeric",
            }
            for name in ALL_FEATURES
        ],
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "excluded_by_design": {
            "inventory_levels": (
                "Only one inventory snapshot exists, so stock at earlier origins is "
                "unknowable. Stock is a decision input, not a demand feature (D-011)."
            )
        },
    }


def assert_no_target_features(feature_names: list[str], target: str) -> None:
    """Fail loudly if a target-derived column is used as an input feature."""
    forbidden = {target, "target_start", "target_end"}
    leaked = forbidden.intersection(feature_names)
    if leaked:
        raise AssertionError(f"Target-derived columns used as features: {sorted(leaked)}")
