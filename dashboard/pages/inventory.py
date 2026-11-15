"""Inventory Optimisation: coverage, reorder guidance and FEFO rotation."""

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
    style_status_column,
)
from dashboard.theme import ACTION_COLORS, PLOTLY_CONFIG, STATUS_COLORS


def render() -> None:
    page_header(
        "Inventory Optimisation",
        "Coverage against forecast demand, replenishment quantities with their arithmetic, "
        "and FEFO rotation guidance.",
    )

    try:
        overview = api.get("/api/v1/inventory/overview")
        recommendations = api.get("/api/v1/recommendations", {"limit": 500})
    except api.ApiError as exc:
        api_error_state(exc)
        return

    items = pd.DataFrame(overview["items"])
    recs = pd.DataFrame(recommendations["items"])

    if items.empty:
        empty_state("No inventory records.")
        return

    coverage = items["coverage_days"].dropna()
    kpi_row(
        [
            {
                "label": "Units on hand",
                "value": f"{overview['total_units_on_hand']:,}",
                "hint": f"value {overview['inventory_value_at_cost']:,.0f} at cost",
            },
            {
                "label": "Below reorder point",
                "value": f"{overview['products_below_reorder']:,}",
                "hint": "need replenishment",
                "tone": STATUS_COLORS["below_reorder"],
            },
            {
                "label": "Stocked out",
                "value": f"{int((items['current_stock'] == 0).sum()):,}",
                "hint": "zero units on hand",
                "tone": STATUS_COLORS["critical"],
            },
            {
                "label": "Overstocked",
                "value": f"{overview['products_excess']:,}",
                "hint": "coverage beyond 90 days",
                "tone": STATUS_COLORS["excess"],
            },
            {
                "label": "Median coverage",
                "value": f"{coverage.median():,.0f} d" if not coverage.empty else "n/a",
                "hint": "stock / forecast daily rate",
            },
        ]
    )

    st.markdown("")
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        status_counts = (
            items["stock_status"]
            .value_counts()
            .reindex(["critical", "below_reorder", "adequate", "excess"])
            .fillna(0)
        )
        figure = go.Figure(
            go.Bar(
                x=[s.replace("_", " ").title() for s in status_counts.index],
                y=status_counts.to_list(),
                marker=dict(
                    color=[STATUS_COLORS[s] for s in status_counts.index], line=dict(width=0)
                ),
                text=status_counts.to_list(),
                textposition="outside",
                hovertemplate="%{x}: %{y} products<extra></extra>",
            )
        )
        figure.update_layout(title="Products by stock status", height=320, yaxis_title="products")
        st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        plot_frame = items.dropna(subset=["coverage_days"])
        figure = go.Figure(
            go.Histogram(
                x=plot_frame["coverage_days"].clip(upper=180),
                nbinsx=28,
                marker=dict(color="#CBD5E1", line=dict(color="white", width=1)),
                hovertemplate="coverage %{x:.0f}d: %{y} products<extra></extra>",
            )
        )
        figure.add_vline(x=30, line=dict(color=ACTION_COLORS["reorder"], width=2, dash="dash"))
        figure.update_layout(
            title="Coverage distribution (clipped at 180 days)",
            height=320,
            xaxis_title="days of cover",
            yaxis_title="products",
        )
        st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("### Replenishment and rotation queue")
    if recs.empty:
        empty_state("No recommendations.")
    else:
        action_filter = st.multiselect(
            "Action",
            sorted(recs["action"].unique()),
            default=[a for a in ["reorder", "urgent_rotation"] if a in set(recs["action"])],
        )
        filtered = recs[recs["action"].isin(action_filter)] if action_filter else recs
        filtered = filtered.sort_values(
            ["priority", "suggested_replenishment_qty"], ascending=[True, False]
        ).head(60)

        display = pd.DataFrame(
            {
                "Product": filtered["product_id"],
                "Medicine": filtered["medicine_name"],
                "Action": filtered["action"].str.replace("_", " ").str.title(),
                "Priority": filtered["priority"].str.title(),
                "Order qty": filtered["suggested_replenishment_qty"],
                "Forecast 30d": filtered["forecast_demand"].round(1),
                "Stock": filtered["current_stock"],
                "Safety": filtered["safety_stock"],
                "Lead time": filtered["lead_time_days"],
                "Coverage (d)": filtered["coverage_days"],
                "Excess": filtered["excess_units"],
                "Reason": filtered["reasons"].str[0],
            }
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Order qty": st.column_config.NumberColumn(format="%d"),
                "Forecast 30d": st.column_config.NumberColumn(format="%.1f"),
                "Coverage (d)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        with st.expander("Show the arithmetic behind one recommendation"):
            product_id = st.selectbox("Product", filtered["product_id"].tolist())
            row = filtered[filtered["product_id"] == product_id].iloc[0]
            st.markdown(f"**{row['medicine_name']}** · {product_id} · `{row['action']}`")
            for reason in row["reasons"]:
                st.markdown(f"- {reason}")
            st.markdown("**Inputs used**")
            st.json(row["inputs"], expanded=True)
            st.markdown(
                '<p class="ms-note">Reorder point = mean daily demand × lead time + safety stock. '
                "Safety stock = z × σ_daily × √lead time, with z = 1.65 (~95% cycle service "
                "level). Suggested quantity is a suggestion only — no order is placed.</p>",
                unsafe_allow_html=True,
            )

    st.markdown("### Stock position by product")
    status_filter = st.multiselect(
        "Status", ["critical", "below_reorder", "adequate", "excess"], default=[]
    )
    table = items if not status_filter else items[items["stock_status"].isin(status_filter)]
    renamed = table[
        [
            "product_id",
            "medicine_name",
            "category",
            "current_stock",
            "reorder_level",
            "supplier_lead_time",
            "coverage_days",
            "stock_status",
        ]
    ].rename(
        columns={
            "product_id": "Product",
            "medicine_name": "Medicine",
            "category": "Category",
            "current_stock": "Stock",
            "reorder_level": "Reorder level",
            "supplier_lead_time": "Lead time",
            "coverage_days": "Coverage (d)",
            "stock_status": "Status",
        }
    )
    # Style after renaming: the styler must reference the displayed column name.
    st.dataframe(
        style_status_column(renamed, column="Status"),
        use_container_width=True,
        hide_index=True,
    )

    callout(
        "Outstanding purchase orders, budget limits and pack-size rounding are not modelled, "
        "so suggested quantities are a planning starting point rather than an order proposal."
    )
