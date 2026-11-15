"""FEFO (First Expiry, First Out) allocation.

Why FEFO: a pharmacy dispenses the earliest-expiring stock first, because
otherwise stock expires on the shelf. FIFO (first *received* first out) is the
inventory-accounting convention and is the wrong rule for perishable goods;
swapping the two is a standard reviewer question ("what changes if you use
FIFO?") and the answer is that batches with later arrival but earlier expiry stop
being consumed, which makes expiry risk materially worse.

The allocation is deliberately simple and inspectable:

    For a batch B expiring at T_B:
        cumulative_demand(T_B) = demand the pharmacy is expected to sell by T_B
        earlier_remaining       = remaining quantity of all batches expiring before B
        allocatable(B)          = max(0, cumulative_demand(T_B) - earlier_remaining)
        allocated(B)            = min(remaining(B), allocatable(B))
        unsold(B)               = remaining(B) - allocated(B)

Interpretation: by the time B expires, demand has arrived for everything earlier
first; whatever is left over is what B can still sell. This is a *steady-state*
approximation -- it ignores intra-window seasonality and assumes demand is
evenly distributed at the forecast rate -- stated explicitly in
``LIMITATIONS.md`` 3.2.

Ties on expiry date are broken by ``batch_id`` so the result is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FefoPolicy:
    """Parameters of the allocation rule (recorded in the artifact bundle)."""

    max_extrapolation_days: int = 365
    """Cap on how far beyond the forecast horizon we extrapolate demand."""

    policy_version: str = "fefo-v1"


def _daily_rate_map(daily_rate: pd.Series | dict[str, float]) -> dict[str, float]:
    if isinstance(daily_rate, pd.Series):
        return {str(k): float(v) for k, v in daily_rate.items()}
    return {str(k): float(v) for k, v in daily_rate.items()}


def allocate_fefo(
    batches: pd.DataFrame,
    daily_rate: pd.Series | dict[str, float],
    snapshot: pd.Timestamp,
    policy: FefoPolicy | None = None,
) -> pd.DataFrame:
    """Allocate expected demand to batches in expiry order.

    Args:
        batches: one row per batch with ``batch_id``, ``product_id``,
            ``remaining_quantity`` and ``expiry_date``.
        daily_rate: expected units per day per product (forecast / horizon).
        snapshot: the assessment date; days-to-expiry is measured from here.

    Returns a frame with one row per input batch, adding
    ``days_to_expiry``, ``expected_demand_before_expiry``,
    ``allocatable_demand``, ``allocated_quantity`` and ``unsold_quantity``.
    """
    policy = policy or FefoPolicy()
    snapshot = pd.Timestamp(snapshot).normalize()
    rates = _daily_rate_map(daily_rate)

    frame = batches.copy()
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"]).dt.normalize()
    frame["remaining_quantity"] = frame["remaining_quantity"].astype(float)

    frame["days_to_expiry"] = (frame["expiry_date"] - snapshot).dt.days.astype(int)

    records: list[dict[str, object]] = []
    for product_id, group in frame.groupby("product_id", sort=False):
        rate = rates.get(str(product_id), 0.0)
        ordered = group.sort_values(["expiry_date", "batch_id"], kind="mergesort")

        consumed_by_earlier = 0.0
        for row in ordered.itertuples(index=False):
            days = int(row.days_to_expiry)
            horizon_days = min(max(days, 0), policy.max_extrapolation_days)
            cumulative_demand = max(0.0, rate * horizon_days)
            allocatable = max(0.0, cumulative_demand - consumed_by_earlier)
            allocated = min(float(row.remaining_quantity), allocatable)
            # What earlier batches actually ABSORB is what they sell, not what they
            # held. Units that expired on their shelf are discarded and cannot have
            # consumed demand that arrives later. Using remaining_quantity here
            # overstates later-batch risk (a genuine bug this line's comment exists
            # to prevent regressing).
            consumed_by_earlier += allocated

            records.append(
                {
                    "batch_id": row.batch_id,
                    "product_id": product_id,
                    "remaining_quantity": float(row.remaining_quantity),
                    "expiry_date": row.expiry_date,
                    "days_to_expiry": days,
                    "expected_demand_before_expiry": round(cumulative_demand, 4),
                    "allocatable_demand": round(allocatable, 4),
                    "allocated_quantity": round(allocated, 4),
                    "unsold_quantity": round(float(row.remaining_quantity) - allocated, 4),
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    return result.sort_values(["product_id", "expiry_date", "batch_id"]).reset_index(drop=True)


def total_unsold(allocation: pd.DataFrame) -> float:
    if allocation.empty:
        return 0.0
    return float(np.sum(allocation["unsold_quantity"]))
