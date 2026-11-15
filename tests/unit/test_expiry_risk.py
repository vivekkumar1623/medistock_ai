"""FEFO allocation and expiry-risk policy tests."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.expiry_risk.fefo import FefoPolicy, allocate_fefo, total_unsold
from ml.expiry_risk.risk import assess_batches, classify_severity, summarise

SNAPSHOT = pd.Timestamp("2025-10-01")


def _batches(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "batch_id": batch_id,
                "product_id": product_id,
                "remaining_quantity": quantity,
                "expiry_date": pd.Timestamp(expiry),
            }
            for batch_id, product_id, quantity, expiry in rows
        ]
    )


def test_no_demand_leaves_everything_unsold() -> None:
    batches = _batches([("B1", "P1", 100, "2025-11-01")])
    allocation = allocate_fefo(batches, {"P1": 0.0}, SNAPSHOT)
    assert allocation.iloc[0]["unsold_quantity"] == 100
    assert total_unsold(allocation) == 100


def test_demand_consumes_earliest_expiry_first() -> None:
    """The earlier batch absorbs demand until it expires; the later batch gets the rest.

    Numbers: 2 units/day from 2025-10-01.
      * B-EARLY expires in 31 days -> 62 units of demand available.
      * B-LATE expires in 243 days -> 486 units cumulative, minus the 62 that
        B-EARLY actually sold = 424 available to B-LATE.
    B-EARLY leaves 38 units unsold (they expire); B-LATE sells out.
    """
    batches = _batches(
        [
            ("B-LATE", "P1", 100, "2026-06-01"),
            ("B-EARLY", "P1", 100, "2025-11-01"),
        ]
    )
    allocation = allocate_fefo(batches, {"P1": 2.0}, SNAPSHOT)
    early = allocation[allocation["batch_id"] == "B-EARLY"].iloc[0]
    late = allocation[allocation["batch_id"] == "B-LATE"].iloc[0]

    assert early["allocatable_demand"] == pytest.approx(62.0)
    assert early["unsold_quantity"] == pytest.approx(38.0)
    assert late["allocatable_demand"] == pytest.approx(424.0)
    assert late["unsold_quantity"] == pytest.approx(0.0)


def test_expired_units_do_not_consume_later_demand() -> None:
    """Regression guard for the allocation bug found by this suite.

    B-EARLY holds 100 but can only sell 10 before it expires. The 90 discarded
    units must NOT reduce the demand available to B-LATE, otherwise every
    later batch looks artificially at risk.
    """
    batches = _batches(
        [
            ("B-EARLY", "P1", 100, "2025-10-11"),  # 10 days -> 10 units at 1/day
            ("B-LATE", "P1", 100, "2026-06-01"),  # plenty of demand afterwards
        ]
    )
    allocation = allocate_fefo(batches, {"P1": 1.0}, SNAPSHOT)
    early = allocation[allocation["batch_id"] == "B-EARLY"].iloc[0]
    late = allocation[allocation["batch_id"] == "B-LATE"].iloc[0]

    assert early["allocated_quantity"] == pytest.approx(10.0)
    assert early["unsold_quantity"] == pytest.approx(90.0)
    # 243 days * 1/day = 243 cumulative, minus 10 actually sold = 233 available.
    assert late["allocatable_demand"] == pytest.approx(233.0)
    assert late["unsold_quantity"] == pytest.approx(0.0)


def test_allocation_never_exceeds_total_demand() -> None:
    batches = _batches(
        [
            ("B1", "P1", 50, "2025-11-01"),
            ("B2", "P1", 50, "2026-01-15"),
            ("B3", "P1", 50, "2026-08-01"),
        ]
    )
    rate = 1.0
    allocation = allocate_fefo(batches, {"P1": rate}, SNAPSHOT)
    total_expected = max(row["expected_demand_before_expiry"] for _, row in allocation.iterrows())
    assert allocation["allocated_quantity"].sum() <= total_expected + 1e-6


def test_multiple_products_are_independent() -> None:
    batches = _batches([("A1", "PA", 10, "2025-11-01"), ("B1", "PB", 10, "2025-11-01")])
    allocation = allocate_fefo(batches, {"PA": 1.0, "PB": 0.0}, SNAPSHOT)
    unsold = dict(zip(allocation["batch_id"], allocation["unsold_quantity"], strict=True))
    assert unsold["A1"] < 10
    assert unsold["B1"] == 10


def test_unknown_product_is_treated_as_zero_demand() -> None:
    batches = _batches([("X1", "UNKNOWN", 25, "2025-11-01")])
    allocation = allocate_fefo(batches, {}, SNAPSHOT)
    assert allocation.iloc[0]["unsold_quantity"] == 25


def test_tie_break_on_equal_expiry_is_deterministic() -> None:
    batches = _batches([("B2", "P1", 10, "2025-11-01"), ("B1", "P1", 10, "2025-11-01")])
    first = allocate_fefo(batches, {"P1": 1.0}, SNAPSHOT)
    second = allocate_fefo(batches.sample(frac=1, random_state=1), {"P1": 1.0}, SNAPSHOT)
    assert list(first["batch_id"]) == list(second["batch_id"]) == ["B1", "B2"]


def test_policy_caps_extrapolation() -> None:
    batches = _batches([("B1", "P1", 10_000, "2030-01-01")])
    capped = allocate_fefo(
        batches, {"P1": 10.0}, SNAPSHOT, policy=FefoPolicy(max_extrapolation_days=30)
    )
    assert capped.iloc[0]["expected_demand_before_expiry"] == pytest.approx(300.0)


# --- Severity policy --------------------------------------------------------
THRESHOLDS = {"critical": 30, "high": 60, "medium": 90}


@pytest.mark.parametrize(
    ("ratio", "days", "expected"),
    [
        (0.0, 5, "low"),  # no exposure -> low even when expiry is imminent
        (0.10, 25, "high"),  # <=30 days, small exposure -> high
        (0.30, 25, "critical"),  # <=30 days, material exposure -> critical
        (0.10, 55, "medium"),
        (0.60, 55, "high"),
        (0.10, 80, "low"),
        (0.30, 80, "medium"),
        (0.60, 80, "critical"),
        (0.60, 200, "medium"),
        (0.90, 200, "high"),
    ],
)
def test_severity_table(ratio: float, days: int, expected: str) -> None:
    assert classify_severity(ratio, days, THRESHOLDS).value == expected


def test_assess_adds_consistent_columns() -> None:
    batches = _batches([("B1", "P1", 100, "2025-11-01"), ("B2", "P2", 40, "2026-03-01")])
    allocation = allocate_fefo(batches, {"P1": 0.5, "P2": 2.0}, SNAPSHOT)
    assessed = assess_batches(allocation, THRESHOLDS)

    assert {"risk_score", "risk_level", "at_risk", "reasons"} <= set(assessed.columns)
    assert assessed["risk_score"].between(0, 1).all()
    assert (assessed["at_risk"] == (assessed["unsold_quantity"] > 0)).all()
    assert assessed.loc[assessed["batch_id"] == "B1", "risk_level"].iloc[0] != "low"

    summary = summarise(assessed)
    assert summary["total_batches"] == 2
    assert summary["at_risk_batches"] == assessed["at_risk"].sum()


def test_assess_handles_empty_allocation() -> None:
    empty = pd.DataFrame(
        columns=[
            "batch_id",
            "product_id",
            "remaining_quantity",
            "days_to_expiry",
            "expected_demand_before_expiry",
            "allocatable_demand",
            "allocated_quantity",
            "unsold_quantity",
            "expiry_date",
        ]
    )
    assessed = assess_batches(empty, THRESHOLDS)
    assert assessed.empty
    summary = summarise(assessed)
    assert summary["total_batches"] == 0
