"""Replenishment mathematics, with every assumption visible.

``safety_stock = z * sigma_daily * sqrt(lead_time)``

That is the standard buffer for demand variability during lead time under a
normal approximation. ``z = 1.65`` corresponds to roughly a 95% cycle service
level (probability of not stocking out in a cycle). It is a policy choice, not a
derived truth: raising ``z`` costs holding cost, lowering it costs stockouts, and
the right value is a business decision the pharmacy owner makes.

``reorder_point = mean_daily * lead_time + safety_stock``
    Stock level at which a replenishment should be triggered, so that expected
    lead-time demand plus the buffer is covered.

``target_level = mean_daily * (lead_time + horizon) + safety_stock``
    Enough to cover lead time *and* the forecast horizon, so the next review
    period does not immediately re-trigger.

``suggested_quantity = ceil(max(0, target_level - current_stock))``

Deliberately excluded: outstanding purchase orders, budget constraints, pack-size
rounding and shelf-life limits on incoming stock. Each would be a real
requirement in production and each is listed in ``LIMITATIONS.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplenishmentPolicy:
    service_level_z: float = 1.65
    horizon_days: int = 30
    excess_coverage_days: int = 90
    policy_version: str = "replenishment-v1"

    def describe(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "safety_stock_formula": "z * sigma_daily * sqrt(lead_time_days)",
            "service_level_z": self.service_level_z,
            "implied_service_level": "~95% cycle service level (normal approximation)",
            "reorder_point_formula": "mean_daily * lead_time + safety_stock",
            "target_level_formula": "mean_daily * (lead_time + horizon) + safety_stock",
            "suggested_quantity_formula": "ceil(max(0, target_level - current_stock))",
            "excess_threshold_days": self.excess_coverage_days,
            "excluded": [
                "outstanding purchase orders",
                "budget constraints",
                "pack-size rounding",
                "incoming batch shelf-life limits",
            ],
        }


def mean_daily_demand(forecast_demand: float, horizon_days: int) -> float:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    return max(0.0, forecast_demand) / horizon_days


def safety_stock(sigma_daily: float, lead_time_days: int, z: float) -> float:
    if lead_time_days <= 0:
        return 0.0
    return max(0.0, z * max(0.0, sigma_daily) * math.sqrt(lead_time_days))


def plan_replenishment(
    current_stock: int,
    forecast_demand: float,
    lead_time_days: int,
    sigma_daily: float,
    policy: ReplenishmentPolicy | None = None,
) -> dict[str, float | int]:
    """Return every intermediate number, so the recommendation is auditable."""
    policy = policy or ReplenishmentPolicy()
    mean_daily = mean_daily_demand(forecast_demand, policy.horizon_days)
    stock = max(0, int(current_stock))

    safety = safety_stock(sigma_daily, lead_time_days, policy.service_level_z)
    reorder_point = mean_daily * lead_time_days + safety
    target_level = mean_daily * (lead_time_days + policy.horizon_days) + safety
    suggested = math.ceil(max(0.0, target_level - stock))

    coverage_days = (stock / mean_daily) if mean_daily > 0 else None
    excess_reference = mean_daily * policy.excess_coverage_days
    excess_units = int(max(0.0, stock - excess_reference)) if mean_daily > 0 else 0

    return {
        "mean_daily_demand": round(mean_daily, 4),
        "safety_stock": int(round(safety)),
        "reorder_point": int(round(reorder_point)),
        "target_level": int(round(target_level)),
        "suggested_quantity": int(suggested),
        "coverage_days": None if coverage_days is None else round(float(coverage_days), 2),
        "excess_units": excess_units,
        "service_level_z": policy.service_level_z,
    }
