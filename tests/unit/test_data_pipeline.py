"""Generator determinism, invariants and metric edge cases."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml.data_generation.generate import generate_dataset
from ml.evaluation.metrics import evaluate, mae, rmse, wape


@pytest.fixture(scope="module")
def small_dataset():
    from ml.data_generation.config import GeneratorConfig

    config = replace(GeneratorConfig(), n_products=6)
    return generate_dataset(config)


def test_generation_is_deterministic(small_dataset) -> None:
    """Same seed must produce identical rows, column order and dtypes."""
    from ml.data_generation.config import GeneratorConfig

    again = generate_dataset(replace(GeneratorConfig(), n_products=6))

    pd.testing.assert_frame_equal(small_dataset.sales, again.sales)
    pd.testing.assert_frame_equal(small_dataset.products, again.products)
    pd.testing.assert_frame_equal(small_dataset.inventory, again.inventory)
    pd.testing.assert_frame_equal(small_dataset.batches, again.batches)
    pd.testing.assert_frame_equal(small_dataset.demand_panel, again.demand_panel)


def test_different_seed_changes_the_data(small_dataset) -> None:
    from ml.data_generation.config import GeneratorConfig

    other = generate_dataset(replace(GeneratorConfig(), n_products=6, seed=7))
    assert not small_dataset.sales["quantity_sold"].equals(other.sales["quantity_sold"])


def test_batch_stock_reconciles_with_inventory(small_dataset) -> None:
    batches = small_dataset.batches
    inventory = small_dataset.inventory
    totals = batches.groupby("product_id")["remaining_quantity"].sum()
    joined = (
        inventory.set_index("product_id")["current_stock"]
        .to_frame()
        .join(totals.rename("batch_total"), how="left")
        .fillna({"batch_total": 0})
    )
    assert (joined["current_stock"] == joined["batch_total"]).all()


def test_batch_dates_are_coherent(small_dataset) -> None:
    batches = small_dataset.batches
    snapshot = pd.Timestamp(small_dataset.config.snapshot_date)
    assert (batches["manufacturing_date"] < batches["expiry_date"]).all()
    assert (batches["batch_arrival_date"] <= snapshot).all()
    assert (batches["remaining_quantity"] <= batches["batch_quantity"]).all()
    assert (batches["remaining_quantity"] > 0).all()


def test_sales_are_valid_transactions(small_dataset) -> None:
    sales = small_dataset.sales
    assert (sales["quantity_sold"] > 0).all()
    assert (sales["selling_price"] > 0).all()
    assert sales["discount"].between(0, 1).all()
    assert not sales["transaction_id"].duplicated().any()
    assert sales["product_id"].isin(small_dataset.products["product_id"]).all()


def test_demand_panel_is_complete_grid(small_dataset) -> None:
    panel = small_dataset.demand_panel
    expected = small_dataset.products["product_id"].nunique() * small_dataset.dates.size
    assert len(panel) == expected
    assert not panel.duplicated(subset=["product_id", "date"]).any()
    assert (panel["quantity"] >= 0).all()


def test_panel_zeros_correspond_to_absent_transactions(small_dataset) -> None:
    panel = small_dataset.demand_panel
    sales = small_dataset.sales

    sold = sales.groupby(["product_id", "transaction_date"])["quantity_sold"].sum()
    sold.index = sold.index.set_names(["product_id", "date"])

    # Totals must agree before the row-level check: a mismatch here would mean the
    # panel and the transaction ledger disagree about how much was sold at all.
    assert panel["quantity"].sum() == sales["quantity_sold"].sum()
    assert (panel["quantity"] > 0).sum() == len(sold)

    joined = panel.set_index(["product_id", "date"]).join(sold.rename("sold"), how="left")
    joined["sold"] = joined["sold"].fillna(0).astype("int64")
    assert (joined["quantity"] == joined["sold"]).all()


def test_seasonal_and_weekday_structure_exists(small_dataset) -> None:
    sales = small_dataset.sales
    assert sales["day_of_week"].between(0, 6).all()
    assert sales["month"].between(1, 12).all()
    assert sales["season"].isin(["Winter", "Spring", "Summer", "Autumn"]).all()
    # The generator encodes weekday structure; a 6-product sample should still show
    # variation across weekdays rather than a flat series.
    by_day = sales.groupby("day_of_week")["quantity_sold"].mean()
    assert by_day.std() > 0


# --- Metrics ----------------------------------------------------------------
def test_wape_formula() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    assert wape(actual, predicted) == pytest.approx((2 + 2 + 3) / 60)


def test_wape_is_none_when_all_actuals_zero() -> None:
    """Undefined must be None, not 0.0 (which would look like a perfect model)."""
    assert wape(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])) is None


def test_wape_ignores_zero_rows_but_counts_their_error() -> None:
    actual = np.array([0.0, 10.0])
    predicted = np.array([5.0, 10.0])
    assert wape(actual, predicted) == pytest.approx(0.5)


def test_mae_rmse_values() -> None:
    assert mae([1, 2, 3], [1, 2, 3]) == 0
    assert rmse([0, 0], [3, 3]) == pytest.approx(3.0)


def test_evaluate_reports_bias_and_count() -> None:
    scores = evaluate([10, 20], [12, 18])
    assert scores["n_observations"] == 2
    assert scores["bias"] == pytest.approx(0.0)


def test_compare_requires_identical_row_counts() -> None:
    from ml.evaluation.metrics import compare_on_common_rows

    frame = pd.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="same rows"):
        compare_on_common_rows(frame, {"m": np.array([1.0])}, "y")
