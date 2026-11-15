"""Expiry Intelligence: which batches will not sell through before they expire."""

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
from dashboard.theme import PLOTLY_CONFIG, RISK_COLORS


def render() -> None:
    page_header(
        "Expiry Intelligence",
        "Batch-level exposure under a FEFO rotation assumption: estimated unsold units, "
        "days remaining, and the severity rule applied.",
    )

    try:
        payload = api.get("/api/v1/expiry-risk", {"limit": 500})
        info = api.get("/api/v1/model/info")
    except api.ApiError as exc:
        api_error_state(exc)
        return

    items = payload["items"]
    if not items:
        empty_state("No batches in the bundle.", "Run training and expiry assessment.")
        return

    frame = pd.DataFrame(items)
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"])

    kpi_row(
        [
            {
                "label": "Batches assessed",
                "value": f"{payload['total_batches']:,}",
                "hint": f"as of {payload['as_of_date']}",
            },
            {
                "label": "At risk",
                "value": f"{payload['at_risk_batches']:,}",
                "hint": "estimated unsold > 0",
                "tone": RISK_COLORS["high"],
            },
            {
                "label": "Estimated unsold",
                "value": f"{payload['estimated_unsold_units']:,.0f}",
                "hint": "units",
            },
            {
                "label": "Value at cost",
                "value": f"{payload['estimated_unsold_value_at_cost']:,.0f}",
                "hint": "estimated exposure",
                "tone": RISK_COLORS["critical"],
            },
            {
                "label": "Critical batches",
                "value": f"{payload['by_risk_level'].get('critical', 0):,}",
                "hint": "policy band, 30-day escalation",
                "tone": RISK_COLORS["critical"],
            },
        ]
    )

    st.markdown("")
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        figure = go.Figure()
        for level in ["low", "medium", "high", "critical"]:
            subset = frame[frame["risk_level"] == level]
            if subset.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=subset["days_to_expiry"],
                    y=subset["estimated_unsold_units"],
                    mode="markers",
                    name=level.title(),
                    marker=dict(
                        color=RISK_COLORS[level],
                        size=(subset["remaining_quantity"].clip(lower=1) ** 0.5) * 2.2,
                        opacity=0.75,
                        line=dict(width=0.5, color="white"),
                    ),
                    customdata=subset[["batch_id", "medicine_name", "remaining_quantity"]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>%{customdata[1]}"
                        "<br>days to expiry: %{x}"
                        "<br>estimated unsold: %{y:,.0f}"
                        "<br>remaining: %{customdata[2]:,.0f}<extra></extra>"
                    ),
                )
            )
        figure.update_layout(
            title="Exposure vs days remaining (bubble size = remaining quantity)",
            height=400,
            xaxis_title="days to expiry",
            yaxis_title="estimated unsold units",
        )
        st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)

        by_category = (
            frame[frame["estimated_unsold_units"] > 0]
            .groupby("product_id")["estimated_unsold_units"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        names = api.get("/api/v1/products", {"limit": 500})["items"]
        name_map = {p["product_id"]: p["medicine_name"] for p in names}
        by_category["label"] = by_category["product_id"].map(name_map)
        bar = go.Figure(
            go.Bar(
                x=by_category["estimated_unsold_units"],
                y=by_category["label"],
                orientation="h",
                marker=dict(color=RISK_COLORS["high"], line=dict(width=0)),
                hovertemplate="%{y}: %{x:,.0f} units<extra></extra>",
            )
        )
        bar.update_layout(
            title="Top products by estimated exposure",
            height=360,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(bar, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        st.markdown("### Severity policy")
        st.markdown(
            "\n".join(
                [
                    "- **At risk** when FEFO-allocatable demand before expiry is less than "
                    "the remaining quantity.",
                    "- **Score** = estimated unsold / remaining quantity.",
                    "- **Critical** escalates when fewer than 30 days remain and more than "
                    "25% of the batch is exposed.",
                    "- A batch with zero exposure is always **low**, whatever its expiry date.",
                ]
            )
        )
        st.markdown(
            f'<p class="ms-note">Policy version: '
            f'{info["training_config"].get("selection_metric", "n/a")}. '
            "Thresholds come from configuration and are recorded in the artifact bundle.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("### Filter")
        levels = st.multiselect(
            "Risk level",
            ["low", "medium", "high", "critical"],
            default=["high", "critical"],
        )

    st.markdown("### Batch exposure detail")
    filtered = frame[frame["risk_level"].isin(levels)] if levels else frame
    filtered = filtered.sort_values(["risk_score", "days_to_expiry"], ascending=[False, True]).head(
        60
    )

    if filtered.empty:
        empty_state("No batches match the selected risk levels.")
    else:
        display = pd.DataFrame(
            {
                "Batch": filtered["batch_id"],
                "Product": filtered["product_id"],
                "Medicine": filtered["medicine_name"],
                "Remaining": filtered["remaining_quantity"].astype(int),
                "Days left": filtered["days_to_expiry"].astype(int),
                "Expiry": filtered["expiry_date"].dt.strftime("%d %b %Y"),
                "Allocatable": filtered["fefo_allocatable_demand"].round(1),
                "Est. unsold": filtered["estimated_unsold_units"].round(1),
                "Score": (filtered["risk_score"] * 100).round(1),
                "Risk": filtered["risk_level"].str.upper(),
            }
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Exposure %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Remaining": st.column_config.NumberColumn(format="%d"),
                "Est. unsold": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        with st.expander("Why is this batch flagged? (audit view)"):
            batch_id = st.selectbox("Batch", filtered["batch_id"].tolist())
            try:
                detail = api.get(f"/api/v1/expiry-risk/{batch_id}")
            except api.ApiError as exc:
                api_error_state(exc)
            else:
                st.markdown(f"**{detail['medicine_name']}** · {detail['batch_id']}")
                for reason in detail["reasons"]:
                    st.markdown(f"- {reason}")
                st.json(detail["inputs"], expanded=False)

    callout(
        f"{payload['disclaimer']} Batches expiring beyond the observed data window cannot be "
        "labelled from realised outcomes, which is why offline evaluation excludes them.",
    )
