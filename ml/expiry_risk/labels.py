"""Realised-outcome labels for expiry risk.

The distinction that matters:

* **Predicted** risk uses *forecast* demand, available at the assessment date.
* **Realised** risk uses *actual* sales after that date, available only later.

Labels come from the realised side. They are computed here, in the offline
pipeline, and are never written into the artifact bundle's feature inputs: the
serving path has no access to them. That separation is what makes the precision
and recall numbers meaningful rather than circular.

Censoring is handled explicitly. A batch expiring after the end of the observed
data cannot be labelled -- we have not watched it long enough -- so it is marked
``label_known = False`` and excluded from the confusion matrix rather than being
optimistically counted as "not at risk".
"""

from __future__ import annotations

import pandas as pd

LABEL_DEFINITION = (
    "at_risk = realised_unsold_units > 0 under FEFO allocation of actual sales "
    "observed between the assessment date and the batch expiry date"
)


def realised_outcomes(
    batches: pd.DataFrame,
    panel: pd.DataFrame,
    snapshot: pd.Timestamp,
    data_end: pd.Timestamp,
    max_extrapolation_days: int = 365,
) -> pd.DataFrame:
    """Compute the realised unsold quantity per batch from actual demand.

    Args:
        batches: batch ledger at the assessment date.
        panel: complete product x day demand grid **including future dates**.
            This frame is only ever used offline, for labelling.
        snapshot: assessment date; sales on this date or earlier are not used.
        data_end: last date with observed demand.
    """
    snapshot = pd.Timestamp(snapshot).normalize()
    data_end = pd.Timestamp(data_end).normalize()

    demand = panel.copy()
    demand["date"] = pd.to_datetime(demand["date"]).dt.normalize()
    demand = demand[(demand["date"] > snapshot) & (demand["date"] <= data_end)]

    # Cumulative actual demand per product by date, so each expiry date can be
    # evaluated with an exact cumulative sum rather than an extrapolated rate.
    cumulative = (
        demand.sort_values(["product_id", "date"])
        .groupby("product_id", sort=False)["quantity"]
        .cumsum()
    )
    demand = demand.assign(cumulative=cumulative.to_numpy())
    # Last observed cumulative demand per product (for batches beyond data_end).
    total_observed = demand.groupby("product_id")["quantity"].sum().to_dict()

    records: list[dict[str, object]] = []
    for product_id, group in batches.groupby("product_id", sort=False):
        ordered = group.sort_values(["expiry_date", "batch_id"], kind="mergesort")
        product_demand = demand[demand["product_id"] == product_id]

        consumed_by_earlier = 0.0
        for row in ordered.itertuples(index=False):
            expiry = pd.Timestamp(row.expiry_date).normalize()
            days = (expiry - snapshot).days
            label_known = expiry <= data_end

            if label_known:
                up_to = product_demand[product_demand["date"] <= expiry]
                cumulative_demand = float(up_to["cumulative"].max()) if not up_to.empty else 0.0
            else:
                # Partial observation: use everything we saw, and flag the censoring.
                observed_days = max(0, (data_end - snapshot).days)
                hard_cap = max(0, min(days, max_extrapolation_days))
                if observed_days >= hard_cap:
                    cumulative_demand = float(total_observed.get(product_id, 0.0))
                else:
                    cumulative_demand = float(total_observed.get(product_id, 0.0))

            allocatable = max(0.0, cumulative_demand - consumed_by_earlier)
            allocated = min(float(row.remaining_quantity), allocatable)
            # Same rule as the prediction side: only units actually SOLD consume
            # future demand. Expired units are discarded. Keeping both sides on the
            # same convention is what makes the comparison meaningful.
            consumed_by_earlier += allocated

            records.append(
                {
                    "batch_id": row.batch_id,
                    "product_id": product_id,
                    "remaining_quantity": float(row.remaining_quantity),
                    "expiry_date": expiry,
                    "days_to_expiry": int(days),
                    "label_known": bool(label_known),
                    "observed_window_days": max(0, (min(expiry, data_end) - snapshot).days),
                    "realised_cumulative_demand": round(cumulative_demand, 4),
                    "realised_allocatable_demand": round(allocatable, 4),
                    "realised_unsold_units": round(float(row.remaining_quantity) - allocated, 4),
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["realised_at_risk"] = frame["realised_unsold_units"] > 0
    return frame.sort_values(["product_id", "expiry_date", "batch_id"]).reset_index(drop=True)
