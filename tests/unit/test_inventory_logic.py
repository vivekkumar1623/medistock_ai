"""Replenishment arithmetic and recommendation rule-order tests."""

from __future__ import annotations

import math

import pytest

from ml.inventory.recommendations import decide_action
from ml.inventory.replenishment import (
    ReplenishmentPolicy,
    plan_replenishment,
    safety_stock,
)


def test_safety_stock_formula() -> None:
    # z * sigma * sqrt(lead time) = 1.65 * 4 * sqrt(9) = 19.8
    assert safety_stock(4.0, 9, 1.65) == pytest.approx(19.8)


def test_safety_stock_is_zero_without_lead_time() -> None:
    assert safety_stock(10.0, 0, 1.65) == 0.0


def test_plan_produces_documented_intermediates() -> None:
    policy = ReplenishmentPolicy(horizon_days=30)
    plan = plan_replenishment(
        current_stock=100, forecast_demand=300.0, lead_time_days=10, sigma_daily=4.0, policy=policy
    )
    mean_daily = 10.0
    expected_safety = 1.65 * 4.0 * math.sqrt(10)
    expected_target = mean_daily * (10 + 30) + expected_safety

    assert plan["mean_daily_demand"] == pytest.approx(mean_daily)
    assert plan["safety_stock"] == round(expected_safety)
    assert plan["target_level"] == round(expected_target)
    assert plan["suggested_quantity"] == math.ceil(max(0.0, expected_target - 100))
    assert plan["coverage_days"] == pytest.approx(10.0)


def test_no_order_when_stock_exceeds_target() -> None:
    plan = plan_replenishment(
        current_stock=5_000, forecast_demand=300.0, lead_time_days=7, sigma_daily=2.0
    )
    assert plan["suggested_quantity"] == 0
    assert plan["excess_units"] > 0


def test_zero_demand_yields_no_coverage_and_no_order() -> None:
    plan = plan_replenishment(
        current_stock=50, forecast_demand=0.0, lead_time_days=5, sigma_daily=0.0
    )
    assert plan["coverage_days"] is None
    assert plan["suggested_quantity"] == 0
    assert plan["excess_units"] == 0


def test_negative_stock_is_clamped() -> None:
    plan = plan_replenishment(
        current_stock=-10, forecast_demand=60.0, lead_time_days=5, sigma_daily=1.0
    )
    assert plan["suggested_quantity"] >= 0


# --- Rule ordering ----------------------------------------------------------
def _decision(**overrides):
    base = dict(
        current_stock=100,
        reorder_level=50,
        forecast_demand=300.0,
        coverage_days=10.0,
        lead_time_days=7,
        plan={"safety_stock": 20, "reorder_point": 90, "target_level": 200, "excess_units": 0},
        expiring_units=0.0,
        nearest_expiry_days=None,
    )
    base.update(overrides)
    return decide_action(**base)


def test_stockout_rule_fires_first() -> None:
    decision = _decision(current_stock=0, expiring_units=40.0, nearest_expiry_days=10)
    assert decision.action.value == "reorder"
    assert decision.priority.value == "high"


def test_rotation_beats_reordering_when_expiry_is_imminent() -> None:
    """A product can need stock and have expiring stock; rotation must win."""
    decision = _decision(
        current_stock=5, coverage_days=1.0, expiring_units=60.0, nearest_expiry_days=12
    )
    assert decision.action.value == "urgent_rotation"


def test_coverage_shorter_than_lead_time_reorders() -> None:
    decision = _decision(current_stock=100, coverage_days=3.0, lead_time_days=10)
    assert decision.action.value == "reorder"
    assert "lead time" in decision.reasons[0].lower() or "covers" in decision.reasons[0].lower()


def test_below_reorder_point_reorders() -> None:
    decision = _decision(current_stock=80, coverage_days=12.0)
    assert decision.action.value == "reorder"
    assert decision.priority.value == "medium"


def test_expiry_exposure_without_immediacy() -> None:
    decision = _decision(
        current_stock=300, coverage_days=40.0, expiring_units=15.0, nearest_expiry_days=70
    )
    assert decision.action.value == "expiry_exposure"


def test_overstock_detected() -> None:
    decision = _decision(current_stock=4_000, coverage_days=200.0)
    assert decision.action.value == "overstock"


def test_sufficient_stock() -> None:
    decision = _decision(current_stock=250, coverage_days=35.0)
    assert decision.action.value == "sufficient"


def test_every_decision_carries_inputs_and_reasons() -> None:
    decision = _decision()
    assert decision.reasons
    assert decision.inputs["current_stock"] == 100
    assert decision.inputs["lead_time_days"] == 7
