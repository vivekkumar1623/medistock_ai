"""Leakage tests: structural, not accuracy-based.

An accuracy assertion cannot detect leakage. These tests assert the *shape of the
information flow*: features must be invariant to data that did not exist at the
origin, and the target must equal the actual future demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ml.features.build import (
    ALL_FEATURES,
    DemandMatrix,
    assert_no_target_features,
    compute_origin_features,
    compute_target,
)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=220, freq="D")
    records = []
    for product_index, product_id in enumerate(["MED-A", "MED-B", "MED-C"]):
        base = 4 + product_index * 3
        values = (base + (np.arange(len(dates)) % 7)).astype(int)
        records.append(pd.DataFrame({"product_id": product_id, "date": dates, "quantity": values}))
    return pd.concat(records, ignore_index=True)


@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["MED-A", "MED-B", "MED-C"],
            "medicine_name": ["A", "B", "C"],
            "category": ["Antibiotics", "Analgesics", "Vitamins & Supplements"],
            "unit_price": [10.0, 20.0, 30.0],
            "unit_cost": [6.0, 12.0, 20.0],
            "lead_time_days": [7, 5, 10],
            "minimum_stock_level": [20, 30, 40],
        }
    )


def test_features_are_invariant_to_future_data(synthetic_panel, products) -> None:
    """The single most important test in the project.

    Build features at origin T using (a) history only up to T, and (b) the full
    panel including later days. If any feature reads the future, the two frames
    diverge. They must be identical.
    """
    origin = pd.Timestamp("2024-06-01")

    history_only = synthetic_panel[synthetic_panel["date"] <= origin]
    full = synthetic_panel

    features_history = compute_origin_features(
        DemandMatrix.from_panel(history_only), products, origin
    )
    features_full = compute_origin_features(DemandMatrix.from_panel(full), products, origin)

    assert_frame_equal(features_history[list(ALL_FEATURES)], features_full[list(ALL_FEATURES)])


def test_target_equals_actual_future_demand(synthetic_panel, products) -> None:
    origin = pd.Timestamp("2024-06-01")
    horizon = 30
    matrix = DemandMatrix.from_panel(synthetic_panel)

    target = compute_target(matrix, origin, horizon)
    assert target is not None

    future = synthetic_panel[
        (synthetic_panel["date"] > origin)
        & (synthetic_panel["date"] <= origin + pd.Timedelta(days=horizon))
    ]
    expected = future.groupby("product_id")["quantity"].sum().reindex(matrix.product_ids).to_numpy()
    np.testing.assert_array_equal(target, expected)


def test_target_is_none_when_horizon_exceeds_data(synthetic_panel, products) -> None:
    matrix = DemandMatrix.from_panel(synthetic_panel)
    last_origin = matrix.dates[-1]
    assert compute_target(matrix, last_origin, 30) is None


def test_target_column_is_rejected_as_a_feature() -> None:
    with pytest.raises(AssertionError):
        assert_no_target_features(["sum_30", "demand_next_30d"], "demand_next_30d")
    with pytest.raises(AssertionError):
        assert_no_target_features(["target_start"], "demand_next_30d")
    assert_no_target_features(["sum_30", "mean_7"], "demand_next_30d")


def test_feature_set_contains_no_inventory_columns() -> None:
    """D-011: stock cannot be reconstructed historically, so it is not a feature."""
    forbidden = {"current_stock", "reorder_level", "remaining_quantity", "batch_quantity"}
    assert forbidden.isdisjoint(set(ALL_FEATURES))


def test_origin_outside_panel_is_rejected(synthetic_panel, products) -> None:
    matrix = DemandMatrix.from_panel(synthetic_panel)
    with pytest.raises(ValueError, match="not a date in the demand panel"):
        compute_origin_features(matrix, products, pd.Timestamp("2030-01-01"))


def test_window_features_match_manual_computation(synthetic_panel, products) -> None:
    origin = pd.Timestamp("2024-06-01")
    matrix = DemandMatrix.from_panel(synthetic_panel)
    features = compute_origin_features(matrix, products, origin)

    history = synthetic_panel[synthetic_panel["date"] <= origin]
    last_30 = (
        history[history["date"] > origin - pd.Timedelta(days=30)]
        .groupby("product_id")["quantity"]
        .sum()
        .reindex(matrix.product_ids)
        .to_numpy()
    )
    np.testing.assert_array_equal(features["sum_30"].to_numpy(), last_30)
