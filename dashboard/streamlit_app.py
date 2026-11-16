"""MediStock AI — Streamlit entrypoint.

Run from the repository root:

    streamlit run dashboard/streamlit_app.py

The dashboard is a pure API client: it never imports the ML pipeline and never
reads a model artifact directly. If the API is down, pages render an explicit
error rather than a placeholder number.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from dashboard import api_client as api  # noqa: E402
from dashboard.pages import (  # noqa: E402
    executive,
    expiry,
    forecasting,
    inventory,
    model_performance,
    product_explorer,
    system_health,
)
from dashboard.theme import install_theme  # noqa: E402

st.set_page_config(
    page_title="MediStock AI",
    page_icon=":material/medication:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Analytics": [
        st.Page(
            executive.render,
            title="Executive Overview",
            icon=":material/dashboard:",
            url_path="executive-overview",
            default=True,
        ),
        st.Page(
            forecasting.render,
            title="Demand Forecasting",
            icon=":material/trending_up:",
            url_path="demand-forecasting",
        ),
        st.Page(
            expiry.render,
            title="Expiry Intelligence",
            icon=":material/event_busy:",
            url_path="expiry-intelligence",
        ),
        st.Page(
            inventory.render,
            title="Inventory Optimisation",
            icon=":material/inventory_2:",
            url_path="inventory-optimisation",
        ),
        st.Page(
            product_explorer.render,
            title="Product Explorer",
            icon=":material/search:",
            url_path="product-explorer",
        ),
    ],
    "Model": [
        st.Page(
            model_performance.render,
            title="Model Performance",
            icon=":material/monitoring:",
            url_path="model-performance",
        ),
    ],
    "System": [
        st.Page(
            system_health.render,
            title="Data Quality & Health",
            icon=":material/health_and_safety:",
            url_path="data-quality-health",
        ),
    ],
}

navigation = st.navigation(PAGES)
install_theme()


def _status_badge() -> None:
    try:
        health = api.health()
    except api.ApiError:
        st.markdown(
            '<span class="ms-chip ms-chip-bad">API OFFLINE</span>',
            unsafe_allow_html=True,
        )
        st.caption("Start the backend:  .\\scripts\\run_api.ps1")
        return

    status = health["status"]
    kind = {"ok": "ok", "degraded": "warn", "unavailable": "bad"}.get(status, "low")
    st.markdown(
        f'<span class="ms-chip ms-chip-{kind}">API {status.upper()}</span>',
        unsafe_allow_html=True,
    )
    artifacts = next((c for c in health["components"] if c["name"] == "artifacts"), None)
    if artifacts and artifacts["status"] != "ok":
        st.caption(f"Artifacts: {artifacts['detail']}")


with st.sidebar:
    st.markdown("### MediStock AI")
    st.caption("Pharmacy demand forecasting & expiry-risk intelligence")
    _status_badge()
    st.divider()

    if st.button("Refresh data", use_container_width=True):
        api.clear_cache()
        st.rerun()

    st.divider()
    st.caption(
        "Decision support only. Not medical advice, dosage or treatment guidance. "
        "No purchase or disposal is executed."
    )
    st.caption("Dataset: synthetic simulator output, not real pharmacy records.")

navigation.run()
