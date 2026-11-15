"""Executive Overview: the five numbers a pharmacy manager checks first."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard.components import (
    api_error_state,
    callout,
    empty_state,
    kpi_row,
    page_header,
)
from dashboard.theme import ACTION_COLORS, PLOTLY_CONFIG, RISK_COLORS


def _bar(labels: list[str], values: list[float], colors: list[str], title: str) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:,.0f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=max(220, 34 * len(labels) + 90),
        margin=dict(l=10, r=60, t=44, b=20),
        xaxis=dict(showgrid=True),
        yaxis=dict(autorange="reversed"),
    )
    return figure


def render() -> None:
    page_header(
        "Executive Overview",
        "Inventory position, forecast demand and expiry exposure for the current "
        "prediction origin.",
    )

    try:
        health = api.health()
        inventory = api.get("/api/v1/inventory/overview")
        expiry = api.get("/api/v1/expiry-risk", {"limit": 500})
        recommendations = api.get("/api/v1/recommendations", {"limit": 500})
    except api.ApiError as exc:
        api_error_state(exc)
        return

    artifacts = next(c for c in health["components"] if c["name"] == "artifacts")
    if artifacts["status"] != "ok":
        empty_state("No trained model artifact is available.", artifacts["detail"])
        return

    rec_items = recommendations["items"]
    forecast_total = sum(item["forecast_demand"] for item in rec_items)
    reorder_alerts = [item for item in rec_items if item["action"] == "reorder"]
    urgent = [item for item in rec_items if item["action"] == "urgent_rotation"]

    kpi_row(
        [
            {
                "label": "Products tracked",
                "value": f"{inventory['total_products']:,}",
                "hint": f"{len(rec_items)} scored at this origin",
            },
            {
                "label": "Units on hand",
                "value": f"{inventory['total_units_on_hand']:,}",
                "hint": f"at cost {inventory['inventory_value_at_cost']:,.0f}",
            },
            {
                "label": "Forecast demand (30d)",
                "value": f"{forecast_total:,.0f}",
                "hint": "sum of product-level forecasts",
            },
            {
                "label": "At-risk batches",
                "value": f"{expiry['at_risk_batches']:,}",
                "hint": f"of {expiry['total_batches']:,} batches",
                "tone": RISK_COLORS["high"] if expiry["at_risk_batches"] else None,
            },
            {
                "label": "Expiry exposure",
                "value": f"{expiry['estimated_unsold_value_at_cost']:,.0f}",
                "hint": "estimated unsold value at cost",
                "tone": RISK_COLORS["critical"]
                if expiry["estimated_unsold_value_at_cost"]
                else None,
            },
            {
                "label": "Replenishment alerts",
                "value": f"{len(reorder_alerts):,}",
                "hint": f"{len(urgent)} urgent rotations",
                "tone": ACTION_COLORS["reorder"] if reorder_alerts else None,
            },
        ]
    )

    st.markdown("")
    left, middle, right = st.columns([1.1, 1.1, 1.3], gap="medium")

    with left:
        status_counts = (
            pd.DataFrame(inventory["items"])["stock_status"]
            .value_counts()
            .reindex(["critical", "below_reorder", "adequate", "excess"])
            .fillna(0)
        )
        st.plotly_chart(
            _bar(
                [s.replace("_", " ").title() for s in status_counts.index],
                status_counts.to_list(),
                ["#B42318", "#B45309", "#067647", "#4C5FD5"],
                "Stock status",
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with middle:
        levels = ["critical", "high", "medium", "low"]
        level_counts = [expiry["by_risk_level"].get(level, 0) for level in levels]
        st.plotly_chart(
            _bar(
                [level.title() for level in levels],
                level_counts,
                [RISK_COLORS[level] for level in levels],
                "Batches by expiry risk",
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with right:
        action_counts = (
            pd.DataFrame(rec_items)["action"].value_counts() if rec_items else pd.Series(dtype=int)
        )
        figure = go.Figure(
            go.Pie(
                labels=[a.replace("_", " ").title() for a in action_counts.index],
                values=action_counts.to_list(),
                hole=0.62,
                marker=dict(
                    colors=[ACTION_COLORS.get(a, "#98A2B3") for a in action_counts.index],
                    line=dict(color="white", width=2),
                ),
                textinfo="none",
                hovertemplate="%{label}: %{value}<extra></extra>",
            )
        )
        figure.update_layout(
            title="Recommendation mix",
            height=340,
            margin=dict(l=10, r=10, t=44, b=10),
            showlegend=True,
            annotations=[
                dict(
                    text=f"<b>{len(rec_items)}</b><br><span style='font-size:11px'>products</span>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=18),
                )
            ],
        )
        st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("### Highest-priority actions")
    if not rec_items:
        empty_state("No recommendations produced.")
    else:
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(
            rec_items, key=lambda item: (priority_rank[item["priority"]], -item["forecast_demand"])
        )[:8]
        table = pd.DataFrame(
            [
                {
                    "Product": item["product_id"],
                    "Medicine": item["medicine_name"],
                    "Action": item["action"].replace("_", " ").title(),
                    "Priority": item["priority"].title(),
                    "Suggested qty": item["suggested_replenishment_qty"],
                    "Forecast (30d)": round(item["forecast_demand"], 1),
                    "Stock": item["current_stock"],
                    "Coverage (d)": item["coverage_days"],
                    "Why": item["reasons"][0] if item["reasons"] else "",
                }
                for item in ordered
            ]
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Suggested qty": st.column_config.NumberColumn(format="%d"),
                "Forecast (30d)": st.column_config.NumberColumn(format="%.1f"),
                "Coverage (d)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    callout(
        "Every figure on this page comes from the loaded model artifact at a single "
        "prediction origin. Exposure is a forecast-conditioned estimate under a FEFO "
        "rotation assumption, not a realised loss."
    )
