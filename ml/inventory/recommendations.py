"""Recommendation rules: an ordered, explicit decision table.

Each rule is checked in order; the first whose condition holds wins, and the
recommendation carries **why** it fired plus the numbers behind it. Order matters
and is itself a policy decision:

1. Stockout with positive demand -> reorder urgently, before anything else.
2. Critical expiry exposure -> rotate, because ordering more of a product that is
   about to expire makes the loss worse.
3. Coverage shorter than the supplier lead time -> reorder, because the product
   can run out before a delivery arrives.
4. Below the reorder point -> reorder.
5. Coverage far beyond the horizon -> overstock, stop buying.
6. Coverage inside the healthy band -> sufficient.
7. Otherwise -> monitor.

Rule 2 before rule 3 is deliberate: a product can simultaneously need stock and
have expiring stock, and the rotation message is the one that prevents a loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.common import Priority, RecommendationAction


@dataclass
class RecommendationDecision:
    action: RecommendationAction
    priority: Priority
    reasons: list[str] = field(default_factory=list)
    inputs: dict[str, float | int | str] = field(default_factory=dict)


def decide_action(
    *,
    current_stock: int,
    reorder_level: int,
    forecast_demand: float,
    coverage_days: float | None,
    lead_time_days: int,
    plan: dict,
    expiring_units: float,
    nearest_expiry_days: int | None,
    excess_coverage_days: int = 90,
) -> RecommendationDecision:
    """Apply the ordered rule table and return the decision with its rationale."""
    inputs: dict[str, float | int | str] = {
        "current_stock": int(current_stock),
        "reorder_level": int(reorder_level),
        "forecast_demand_30d": round(float(forecast_demand), 2),
        "coverage_days": "unbounded" if coverage_days is None else round(coverage_days, 2),
        "lead_time_days": int(lead_time_days),
        "safety_stock": int(plan["safety_stock"]),
        "reorder_point": int(plan["reorder_point"]),
        "expiring_units_within_90d": round(float(expiring_units), 2),
        "nearest_expiry_days": nearest_expiry_days,
    }

    # 1. Stockout with demand.
    if current_stock <= 0 < forecast_demand:
        return RecommendationDecision(
            action=RecommendationAction.REORDER,
            priority=Priority.HIGH,
            reasons=[
                "No stock on hand while forecast demand is positive.",
                f"Supplier lead time is {lead_time_days} days, so a stockout will persist "
                "until a delivery arrives.",
            ],
            inputs=inputs,
        )

    # 2. Critical expiry exposure takes precedence over buying more.
    if expiring_units > 0 and nearest_expiry_days is not None and nearest_expiry_days <= 30:
        return RecommendationDecision(
            action=RecommendationAction.URGENT_ROTATION,
            priority=Priority.HIGH,
            reasons=[
                f"{expiring_units:,.0f} units are forecast to remain unsold before expiry, "
                f"with the nearest batch expiring in {nearest_expiry_days} days.",
                "Rotate the earliest-expiring batch first (FEFO) or arrange a supplier "
                "return before considering a new order.",
            ],
            inputs=inputs,
        )

    # 3. Coverage shorter than lead time.
    if coverage_days is not None and coverage_days < lead_time_days:
        return RecommendationDecision(
            action=RecommendationAction.REORDER,
            priority=Priority.HIGH,
            reasons=[
                f"Current stock covers {coverage_days:.1f} days of forecast demand but the "
                f"supplier lead time is {lead_time_days} days.",
                "The product can run out before a replenishment arrives.",
            ],
            inputs=inputs,
        )

    # 4. Below reorder point.
    if current_stock < plan["reorder_point"]:
        return RecommendationDecision(
            action=RecommendationAction.REORDER,
            priority=Priority.MEDIUM,
            reasons=[
                f"Stock ({current_stock:,}) is below the reorder point "
                f"({plan['reorder_point']:,}), which covers lead-time demand plus "
                f"{plan['safety_stock']:,} units of safety stock.",
            ],
            inputs=inputs,
        )

    # 5. Expiry exposure that is not yet critical.
    if expiring_units > 0:
        return RecommendationDecision(
            action=RecommendationAction.EXPIRY_EXPOSURE,
            priority=Priority.MEDIUM,
            reasons=[
                f"{expiring_units:,.0f} units are forecast to remain unsold before expiry.",
                "Monitor and prioritise FEFO rotation; no immediate purchase is needed.",
            ],
            inputs=inputs,
        )

    # 6. Overstock.
    if coverage_days is not None and coverage_days > excess_coverage_days:
        return RecommendationDecision(
            action=RecommendationAction.OVERSTOCK,
            priority=Priority.LOW,
            reasons=[
                f"Coverage of {coverage_days:.0f} days exceeds the "
                f"{excess_coverage_days}-day threshold.",
                f"Approximately {plan['excess_units']:,} units are in excess of that horizon; "
                "pause replenishment for this product.",
            ],
            inputs=inputs,
        )

    # 7. Healthy band.
    if current_stock >= plan["reorder_point"]:
        return RecommendationDecision(
            action=RecommendationAction.SUFFICIENT,
            priority=Priority.LOW,
            reasons=[
                "Stock is above the reorder point and within the healthy coverage band.",
            ],
            inputs=inputs,
        )

    return RecommendationDecision(
        action=RecommendationAction.MONITOR,
        priority=Priority.LOW,
        reasons=["No rule fired; keep the product under routine review."],
        inputs=inputs,
    )
