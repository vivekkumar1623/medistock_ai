"""Model Performance: metrics, selection rationale, confusion matrix, importance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard.components import api_error_state, callout, empty_state, kpi_row, page_header
from dashboard.theme import ACCENT, INDIGO, PLOTLY_CONFIG


def render() -> None:
    page_header(
        "Model Performance",
        "Demand metrics on identical rows per split, expiry-risk classification quality, "
        "and global feature importance.",
    )

    try:
        metrics = api.get("/api/v1/model/metrics")
        info = api.get("/api/v1/model/info")
    except api.ApiError as exc:
        api_error_state(exc)
        return

    demand = pd.DataFrame(metrics["demand_metrics"])
    if demand.empty:
        empty_state("No metrics recorded in the artifact.")
        return

    test_rows = demand[demand["split"] == "test"].sort_values("wape")
    best = test_rows.iloc[0]

    expiry = metrics.get("expiry_metrics")
    kpi_row(
        [
            {
                "label": "Selected model",
                "value": metrics["selected_model"].replace("_", " ").title(),
                "hint": f"horizon {info['horizon_days']} days",
            },
            {
                "label": "Test WAPE",
                "value": f"{best['wape']:.3f}",
                "hint": f"best of {demand['model_name'].nunique()} candidates",
            },
            {
                "label": "Test MAE",
                "value": f"{best['mae']:.1f}",
                "hint": "units per product-origin",
            },
            {
                "label": "Expiry precision",
                "value": f"{expiry['precision']:.3f}" if expiry else "n/a",
                "hint": "at-risk classification",
            },
            {
                "label": "Expiry recall",
                "value": f"{expiry['recall']:.3f}" if expiry else "n/a",
                "hint": "missed expiries cost most",
            },
        ]
    )

    st.markdown("")
    left, right = st.columns([1, 1], gap="large")

    with left:
        figure = go.Figure()
        for split, colour in (("validation", "#CBD5E1"), ("test", ACCENT)):
            subset = demand[demand["split"] == split].sort_values("wape")
            figure.add_trace(
                go.Bar(
                    x=subset["model_name"].str.replace("_", " ").str.title(),
                    y=subset["wape"],
                    name=split.title(),
                    marker=dict(color=colour, line=dict(width=0)),
                    text=subset["wape"].round(3),
                    textposition="outside",
                    hovertemplate="%{x}: %{y:.3f}<extra></extra>",
                )
            )
        figure.update_layout(
            title="WAPE by model (lower is better)",
            barmode="group",
            height=340,
            yaxis_title="WAPE",
        )
        st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)

    with right:
        st.markdown("### Selection rationale")
        st.markdown(
            f'<p class="ms-note">{metrics["selection_rationale"]}</p>', unsafe_allow_html=True
        )
        selection_metric = info["training_config"].get("selection_metric", "n/a")
        st.markdown(
            "\n".join(
                [
                    f"- Selection metric: `{selection_metric}`",
                    f"- Baseline: `{metrics['baseline_model']}`",
                    f"- Rolling folds: {info['training_config'].get('n_rolling_folds', 'n/a')}",
                    f"- Test rows: {info['training_config'].get('test_rows', 'n/a')}",
                ]
            )
        )
        for note in metrics.get("notes", []):
            st.markdown(f"- {note}")

    st.markdown("### All metrics (identical rows within each split)")
    display = demand.copy()
    display["model"] = display["model_name"].str.replace("_", " ").str.title()
    st.dataframe(
        display[["model", "split", "mae", "rmse", "wape", "n_observations"]].rename(
            columns={
                "model": "Model",
                "split": "Split",
                "mae": "MAE",
                "rmse": "RMSE",
                "wape": "WAPE",
                "n_observations": "n",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "WAPE": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown("### Expiry-risk confusion matrix")
        if not expiry:
            empty_state("Expiry evaluation not present in this artifact.")
        else:
            matrix = expiry["confusion"]
            grid = [
                [matrix["true_negative"], matrix["false_positive"]],
                [matrix["false_negative"], matrix["true_positive"]],
            ]
            figure = go.Figure(
                go.Heatmap(
                    z=grid,
                    x=["Predicted safe", "Predicted at risk"],
                    y=["Actually safe", "Actually at risk"],
                    colorscale=[[0, "#F8FAFC"], [1, "#0E7C86"]],
                    text=[[str(v) for v in row] for row in grid],
                    texttemplate="%{text}",
                    textfont=dict(size=18),
                    showscale=False,
                    hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
                )
            )
            figure.update_layout(
                height=340,
                yaxis=dict(autorange="reversed"),
                title=f"n={expiry['n_batches']} evaluable batches",
            )
            st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown(
                f"- Precision **{expiry['precision']:.3f}** · Recall **{expiry['recall']:.3f}** "
                f"· F1 **{expiry['f1']:.3f}** · Accuracy **{expiry['accuracy']:.3f}**\n"
                f"- Rule: `{expiry['threshold_rule']}`\n"
                f"- Critical band: precision **{expiry['critical_band']['precision']:.3f}**, "
                f"recall **{expiry['critical_band']['recall']:.3f}** "
                f"(n={expiry['critical_band']['n_batches']})"
            )

    with right:
        st.markdown("### Global feature importance (permutation, test set)")
        importance = pd.DataFrame(metrics.get("feature_importance", []))
        if importance.empty:
            empty_state("No importance recorded.")
        else:
            importance = importance.sort_values("importance").tail(12)
            figure = go.Figure(
                go.Bar(
                    x=importance["importance"],
                    y=importance["feature"],
                    orientation="h",
                    marker=dict(color=INDIGO, line=dict(width=0)),
                    hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                )
            )
            figure.update_layout(
                height=400,
                xaxis_title="increase in MAE when shuffled",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(figure, use_container_width=True, config=PLOTLY_CONFIG)

    callout(
        "Global importance answers 'what matters across products'. It does not explain one "
        "prediction: per-product attributions are computed separately by one-at-a-time "
        "ablation and shown on the Demand Forecasting page.",
        tone="warn",
    )
