"""Data Quality and System Health: is the platform trustworthy right now?"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import api_client as api
from dashboard.components import api_error_state, callout, empty_state, kpi_row, page_header

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _load_report(name: str) -> dict | None:
    path = REPORTS_DIR / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _chip(status: str) -> str:
    kind = {"ok": "ok", "degraded": "warn", "unavailable": "bad"}.get(status, "low")
    return f'<span class="ms-chip ms-chip-{kind}">{status.upper()}</span>'


def render() -> None:
    page_header(
        "Data Quality & System Health",
        "Live component health, artifact provenance, and the offline data-quality reports "
        "that were executed during the pipeline.",
    )

    try:
        health = api.health()
    except api.ApiError as exc:
        api_error_state(exc)
        return

    kpi_row(
        [
            {
                "label": "Overall status",
                "value": health["status"].upper(),
                "hint": f"{health['app']} v{health['version']} · {health['environment']}",
            },
            {
                "label": "Components reporting",
                "value": f"{len(health['components'])}",
                "hint": "api · artifacts · database",
            },
            {
                "label": "Checked at (UTC)",
                "value": health["checked_at_utc"][11:19],
                "hint": health["checked_at_utc"][:10],
            },
        ]
    )

    st.markdown("")
    st.markdown("### Component health")
    for component in health["components"]:
        st.markdown(
            f"{_chip(component['status'])} &nbsp; **{component['name']}** — "
            f"<span style='color:#64748B;font-size:0.85rem'>{component.get('detail') or ''}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("### Model artifact")
    try:
        info = api.get("/api/v1/model/info")
    except api.ApiError as exc:
        api_error_state(exc)
        info = None

    if info:
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Artifact version", info["artifact_version"]],
                        ["Contract version", info["contract_version"]],
                        ["Created (UTC)", info["created_at_utc"]],
                        ["Selected model", info["selected_model"]],
                        ["Prediction origin", info["prediction_origin"]],
                        ["Horizon (days)", info["horizon_days"]],
                        ["Features", info["n_features"]],
                        [
                            "Explainability",
                            "available" if info["explainability_available"] else "absent",
                        ],
                        [
                            "Expiry evaluation",
                            "available" if info["expiry_evaluation_available"] else "absent",
                        ],
                    ],
                    columns=["Property", "Value"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.markdown("**Dataset provenance**")
            st.json(info["dataset"], expanded=False)
            st.markdown("**Library versions (recorded at training time)**")
            st.json(info["library_versions"], expanded=False)

    st.markdown("### Data quality (offline pipeline reports)")
    quality = _load_report("data_quality_report.json")
    profile = _load_report("data_profile.json")

    if quality is None:
        empty_state(
            "No data-quality report found.",
            "Run: python -m scripts.validate_data",
        )
    else:
        raw = quality.get("raw", {})
        after = quality.get("after_cleaning", {})
        cleaning = quality.get("cleaning", {})
        kpi_row(
            [
                {
                    "label": "Raw validation errors",
                    "value": f"{raw.get('error_count', 0)}",
                    "hint": "structural / referential checks",
                },
                {
                    "label": "Warnings",
                    "value": f"{raw.get('warning_count', 0)}",
                    "hint": "plausible but noteworthy",
                },
                {
                    "label": "Post-clean errors",
                    "value": f"{after.get('error_count', 0)}",
                    "hint": "must be zero to proceed",
                },
                {
                    "label": "Cleaning actions",
                    "value": f"{cleaning.get('total_rows_affected', 0)}",
                    "hint": "rows modified by repair",
                },
            ]
        )

        issues = raw.get("issues", [])
        if issues:
            st.dataframe(
                pd.DataFrame(issues)[["severity", "table", "check", "count", "message"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No validation issues were raised on the current dataset.")

        if profile:
            st.markdown("### Dataset profile")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Table": table["table"],
                            "Rows": table["rows"],
                            "Columns": table["columns"],
                            "Duplicate rows": table["duplicate_rows"],
                        }
                        for table in profile.get("tables", [])
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            demand = profile.get("demand_characteristics", {})
            zero_share = demand.get("panel_zero_share", 0)
            sparse_products = demand.get("products_with_gt_50pct_zero_days", 0)
            st.markdown(
                f"- Zero-demand share across the panel: **{zero_share:.2%}**\n"
                f"- Products with more than half of days at zero: "
                f"**{sparse_products}**\n"
                f"- Mean daily units per product-day: **{demand.get('mean_daily_units', 0):.2f}**"
            )

    callout(
        "Health is evaluated live on every load. A degraded component is reported as degraded "
        "rather than hidden, because a decision-support tool that silently loses its model is "
        "worse than one that refuses to answer.",
        tone="warn",
    )
