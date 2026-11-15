"""Product Explorer: everything the system knows about one medicine, in one place."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard.components import api_error_state, callout, empty_state, kpi_row, page_header
from dashboard.theme import ACCENT, PLOTLY_CONFIG, RISK_COLORS


def render() -> None:
    page_header(
        "Product Explorer",
        "Master data, stock position, forecast and batch ledger for a single medicine.",
    )

    try:
        products = api.get("/api/v1/products", {"limit": 500})
        inventory = api.get("/api/v1/inventory/overview")
        batches = api.get("/api/v1/expiry-risk", {"limit": 500})
    except api.ApiError as exc:
        api_error_state(exc)
        return

    items = products["items"]
    if not items:
        empty_state("No products available.")
        return

    labels = {f"{p['product_id']} · {p['medicine_name']}": p["product_id"] for p in items}
    selected = st.selectbox("Product", list(labels), index=0)
    product_id = labels[selected]
    product = next(p for p in items if p["product_id"] == product_id)

    stock_row = next((i for i in inventory["items"] if i["product_id"] == product_id), None)

    margin = (product["unit_price"] - product["unit_cost"]) / max(product["unit_price"], 1e-9)
    kpi_row(
        [
            {"label": "Category", "value": product["category"], "hint": product["manufacturer"]},
            {
                "label": "Selling price",
                "value": f"{product['unit_price']:,.2f}",
                "hint": f"cost {product['unit_cost']:,.2f} · margin {margin:.0%}",
            },
            {
                "label": "Supplier",
                "value": product["supplier_id"],
                "hint": f"lead time {product['lead_time_days']} d",
            },
            {
                "label": "Stock on hand",
                "value": f"{stock_row['current_stock']:,}" if stock_row else "n/a",
                "hint": f"reorder level {stock_row['reorder_level']:,}" if stock_row else "",
                "tone": RISK_COLORS["critical"]
                if stock_row and stock_row["current_stock"] == 0
                else None,
            },
            {
                "label": "Coverage",
                "value": f"{stock_row['coverage_days']:,.1f} d"
                if stock_row and stock_row["coverage_days"] is not None
                else "n/a",
                "hint": "stock / forecast daily rate",
            },
        ]
    )

    st.markdown("")
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        try:
            forecast = api.get(f"/api/v1/forecast/{product_id}")
        except api.ApiError as exc:
            api_error_state(exc)
            forecast = None

        if forecast:
            history = pd.DataFrame(forecast["recent_history"])
            history["date"] = pd.to_datetime(history["date"])
            figure = go.Figure()
            figure.add_trace(
                go.Bar(
                    x=history["date"],
                    y=history["quantity"],
                    name="Daily units",
                    marker=dict(color="#CBD5E1", line=dict(width=0)),
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=history["date"],
                    y=history["quantity"].rolling(7, min_periods=1).mean(),
                    name="7-day mean",
                    mode="lines",
                    line=dict(color=ACCENT, width=2),
                )
            )
            figure.update_layout(
                title=(
                    f"Recent demand · forecast next 30d = "
                    f"{forecast['predicted_demand']:,.0f} units"
                ),
                height=330,
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)

            st.markdown(
                f'<p class="ms-note">Confidence: <b>{forecast["confidence"]}</b>. '
                f'Model: {forecast["model_name"].replace("_", " ").title()}, artifact '
                f'{forecast["artifact_version"]}.</p>',
                unsafe_allow_html=True,
            )
            if forecast["drivers"]:
                drivers = pd.DataFrame(forecast["drivers"]).head(5)
                bar = go.Figure(
                    go.Bar(
                        x=drivers["contribution"],
                        y=drivers["feature"],
                        orientation="h",
                        marker=dict(
                            color=[
                                ACCENT if v > 0 else RISK_COLORS["critical"]
                                for v in drivers["contribution"]
                            ],
                            line=dict(width=0),
                        ),
                        hovertemplate="%{y}: %{x:+.1f}<extra></extra>",
                    )
                )
                bar.update_layout(
                    title="Main contributing factors (units)",
                    height=260,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(bar, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        st.markdown("### Batch ledger (FEFO order)")
        product_batches = [b for b in batches["items"] if b["product_id"] == product_id]
        if not product_batches:
            empty_state(
                "No batches recorded for this product.",
                "A product can legitimately have zero stock and therefore zero batches.",
            )
        else:
            table = pd.DataFrame(
                [
                    {
                        "Batch": b["batch_id"],
                        "Remaining": b["remaining_quantity"],
                        "Expires": b["expiry_date"],
                        "Days": b["days_to_expiry"],
                        "Est. unsold": round(b["estimated_unsold_units"], 1),
                        "Risk": b["risk_level"].upper(),
                    }
                    for b in sorted(product_batches, key=lambda x: x["expiry_date"])
                ]
            )
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Remaining": st.column_config.NumberColumn(format="%d"),
                    "Est. unsold": st.column_config.NumberColumn(format="%.1f"),
                },
            )

        st.markdown("### Stock and replenishment")
        if stock_row:
            st.markdown(
                "\n".join(
                    [
                        f"- Current stock: **{stock_row['current_stock']:,}** units",
                        f"- Reorder level: **{stock_row['reorder_level']:,}** units",
                        f"- Supplier lead time: **{stock_row['supplier_lead_time']}** days",
                        f"- Minimum stock level: **{product['minimum_stock_level']:,}** units",
                        f"- Status: **{stock_row['stock_status'].replace('_', ' ')}**",
                    ]
                )
            )

    callout(
        "Product Explorer is read-only. It exposes the same numbers the forecasting and "
        "inventory pages use, because all three read the same artifact-backed API."
    )
