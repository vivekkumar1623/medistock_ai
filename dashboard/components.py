"""Reusable UI atoms: page header, KPI cards, chips, error states, footers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import streamlit as st

from dashboard.theme import MUTED, RISK_COLORS, STATUS_COLORS


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(f'<p class="ms-page-sub">{subtitle}</p>', unsafe_allow_html=True)


def kpi_row(cards: list[dict]) -> None:
    """Render KPI cards. Each card: label, value, hint (optional), tone (optional)."""
    columns = st.columns(len(cards), gap="small")
    for column, card in zip(columns, cards, strict=False):
        hint = card.get("hint", "")
        tone = card.get("tone")
        value = card["value"]
        if tone:
            value = f'<span style="color:{tone}">{value}</span>'
        with column:
            st.markdown(
                f"""
                <div class="ms-kpi">
                    <div class="ms-kpi-label">{card['label']}</div>
                    <div class="ms-kpi-value">{value}</div>
                    <div class="ms-kpi-hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def chip(text: str, kind: str) -> str:
    return f'<span class="ms-chip ms-chip-{kind}">{text}</span>'


def risk_chip(level: str) -> str:
    return chip(level.upper(), level if level in {"low", "medium", "high", "critical"} else "low")


def status_chip(status: str) -> str:
    kind = {
        "critical": "bad",
        "below_reorder": "warn",
        "adequate": "ok",
        "excess": "warn",
    }.get(status, "low")
    return chip(status.replace("_", " ").upper(), kind)


def callout(text: str, tone: str = "info") -> None:
    css = "ms-callout" if tone == "info" else "ms-callout ms-callout-warn"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def empty_state(message: str, hint: str | None = None) -> None:
    st.markdown(
        f"""
        <div style="border:1px dashed #D0D5DD;border-radius:12px;padding:2rem;
                    text-align:center;background:#FCFCFD;">
            <div style="font-weight:600;color:#344054;">{message}</div>
            <div style="color:{MUTED};font-size:0.82rem;margin-top:0.35rem;">{hint or ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def api_error_state(exc: Exception) -> None:
    st.error(str(exc))
    st.markdown(
        '<p class="ms-note">The dashboard renders only live API responses. If the API is '
        "not running, every figure here is intentionally unavailable rather than replaced "
        "with placeholder data.</p>",
        unsafe_allow_html=True,
    )


def footer(bundle_version: str | None, origin: str | None, model: str | None) -> None:
    parts = ["MediStock AI — decision support only, not medical advice."]
    parts.append("Data: synthetic simulator (not real pharmacy records).")
    if bundle_version:
        parts.append(f"Artifact: {bundle_version}")
    if model:
        parts.append(f"Model: {model}")
    if origin:
        parts.append(f"Prediction origin: {origin}")
    st.markdown(f'<div class="ms-footer">{" · ".join(parts)}</div>', unsafe_allow_html=True)


def style_risk_column(
    frame: pd.DataFrame, column: str = "risk_level"
) -> pd.io.formats.style.Styler:
    def colour(value: str) -> str:
        colour_value = RISK_COLORS.get(str(value), MUTED)
        return f"color:{colour_value};font-weight:600;text-transform:uppercase;font-size:0.72rem;"

    return frame.style.map(colour, subset=[column])


def style_status_column(
    frame: pd.DataFrame, column: str = "stock_status"
) -> pd.io.formats.style.Styler:
    def colour(value: str) -> str:
        colour_value = STATUS_COLORS.get(str(value), MUTED)
        return f"color:{colour_value};font-weight:600;font-size:0.78rem;"

    return frame.style.map(colour, subset=[column])


def metric_table(rows: Iterable[dict], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))[columns]
