"""Demand Forecasting: one product, one horizon, with its drivers and the model table."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard.components import api_error_state, empty_state, kpi_row, page_header
from dashboard.theme import ACCENT, MUTED, PLOTLY_CONFIG


def _history_chart(history: list[dict], origin: str, predicted: float, lower, upper) -> go.Figure:
    frame = pd.DataFrame(history)
    frame["date"] = pd.to_datetime(frame["date"])
    rolling = frame["quantity"].rolling(7, min_periods=1).mean()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame["quantity"],
            name="Daily units",
            marker=dict(color="#CBD5E1", line=dict(width=0)),
            hovertemplate="%{x|%d %b}: %{y} units<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=rolling,
            name="7-day mean",
            mode="lines",
            line=dict(color=ACCENT, width=2.2),
            hovertemplate="%{x|%d %b}: %{y:.1f}<extra></extra>",
        )
    )

    origin_ts = pd.Timestamp(origin)
    horizon_end = origin_ts + pd.Timedelta(days=30)
    figure.add_vrect(
        x0=origin_ts,
        x1=horizon_end,
        fillcolor=ACCENT,
        opacity=0.06,
        line_width=0,
        annotation_text="forecast window",
        annotation_position="top left",
        annotation_font=dict(size=10, color=MUTED),
    )
    figure.add_trace(
        go.Scatter(
            x=[origin_ts, horizon_end],
            y=[predicted / 30, predicted / 30],
            name="Forecast daily rate",
            mode="lines",
            line=dict(color=ACCENT, width=2, dash="dash"),
            hovertemplate="forecast rate: {y:.1f} units/day<extra></extra>",
        )
    )
    figure.update_layout(
        height=360,
        bargap=0.15,
        title=f"Demand history and forecast (30-day total {predicted:,.0f} units"
        + (f", interval {lower:,.0f}–{upper:,.0f}" if lower is not None else "")
        + ")",
        legend=dict(orientation="h", y=1.08),
        yaxis_title="units / day",
    )
    return figure


def _drivers_chart(drivers: list[dict]) -> go.Figure:
    frame = pd.DataFrame(drivers)
    frame = frame.reindex(frame["contribution"].abs().sort_values().index)
    colors = [ACCENT if value > 0 else "#B42318" for value in frame["contribution"]]
    figure = go.Figure(
        go.Bar(
            x=frame["contribution"],
            y=frame["feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}: %{x:+.1f} units<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(200, 30 * len(frame) + 80),
        title="What moves this forecast (units, vs population reference)",
        xaxis_title="effect on 30-day demand (units)",
        yaxis=dict(autorange="reversed"),
    )
    return figure


def render() -> None:
    page_header(
        "Demand Forecasting",
        "Product-level 30-day demand forecast from the saved model, with the features "
        "that move it and the model comparison behind it.",
    )

    try:
        products = api.get("/api/v1/products", {"limit": 500})
        metrics = api.get("/api/v1/model/metrics")
        info = api.get("/api/v1/model/info")
    except api.ApiError as exc:
        api_error_state(exc)
        return

    items = products["items"]
    if not items:
        empty_state("No products available.")
        return

    labels = {
        f"{p['product_id']} · {p['medicine_name']} ({p['category']})": p["product_id"]
        for p in items
    }
    with st.container():
        left, right = st.columns([3, 1])
        with left:
            selected_label = st.selectbox("Product", list(labels), index=0)
        with right:
            st.metric("Horizon", f"{info['horizon_days']} days")

    product_id = labels[selected_label]

    try:
        forecast = api.get(f"/api/v1/forecast/{product_id}")
    except api.ApiError as exc:
        api_error_state(exc)
        return

    interval = forecast.get("prediction_interval") or {}
    lower = interval.get("lower")
    upper = interval.get("upper")

    kpi_row(
        [
            {
                "label": "Forecast demand (30d)",
                "value": f"{forecast['predicted_demand']:,.0f}",
                "hint": f"units · {forecast['confidence']} confidence",
            },
            {
                "label": "Prediction interval",
                "value": f"{lower:,.0f}–{upper:,.0f}" if lower is not None else "n/a",
                "hint": interval.get("method", "no interval available").replace("_", " "),
            },
            {
                "label": "Model",
                "value": forecast["model_name"].replace("_", " ").title(),
                "hint": f"artifact {forecast['artifact_version']}",
            },
            {
                "label": "Prediction origin",
                "value": forecast["as_of_date"],
                "hint": "only history up to this date was used",
            },
        ]
    )

    st.plotly_chart(
        _history_chart(
            forecast["recent_history"],
            forecast["as_of_date"],
            forecast["predicted_demand"],
            lower,
            upper,
        ),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("### Why this number")
        if forecast["drivers"]:
            st.plotly_chart(
                _drivers_chart(forecast["drivers"]), use_container_width=True, config=PLOTLY_CONFIG
            )
            top = forecast["drivers"][0]
            st.markdown(
                f'<p class="ms-note">Method: {top.get("method", "ablation")}. Each bar is the '
                "change in this product's forecast when that one feature is set to the "
                "population reference value. Effects are measured one at a time and are not "
                "additive; this is not a SHAP attribution.</p>",
                unsafe_allow_html=True,
            )
        else:
            empty_state("No driver attributions available for this artifact.")

    with right:
        st.markdown("### Caveats for this prediction")
        notes = forecast.get("notes") or []
        if notes:
            for note in notes:
                st.markdown(f"- {note}")
        else:
            st.markdown("- No product-specific caveats raised.")
        st.markdown(f'<p class="ms-note">{forecast["disclaimer"]}</p>', unsafe_allow_html=True)

    st.markdown("### Model comparison (identical rows per split)")
    rows = [
        {
            "Model": row["model_name"].replace("_", " ").title(),
            "Split": row["split"],
            "MAE": round(row["mae"], 2),
            "RMSE": round(row["rmse"], 2),
            "WAPE": f"{row['wape']:.3f}",
            "n": row["n_observations"],
        }
        for row in metrics["demand_metrics"]
    ]
    table = pd.DataFrame(rows).sort_values(["Split", "WAPE"])
    st.dataframe(table, use_container_width=True, hide_index=True)

    selected_label = metrics["selected_model"].replace("_", " ").title()
    st.markdown(
        f'<p class="ms-note">Selected: <b>{selected_label}</b>. '
        f'{metrics["selection_rationale"]}</p>',
        unsafe_allow_html=True,
    )
