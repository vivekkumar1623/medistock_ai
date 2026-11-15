"""Visual system: design tokens, global CSS and the Plotly template.

Deliberately restrained. Enterprise analytics UIs are quiet: one accent colour,
neutral greys, generous spacing, and status colour reserved for status. Decorative
gradients and rainbow charts make a dashboard look like a demo, not a product.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Tokens -----------------------------------------------------------------
INK = "#0F172A"  # primary text
MUTED = "#64748B"  # secondary text
BORDER = "#E2E8F0"  # hairlines
SURFACE = "#FFFFFF"
CANVAS = "#F6F8FB"
SIDEBAR = "#0B1220"

ACCENT = "#0E7C86"  # teal: primary series / active nav
ACCENT_SOFT = "#E6F4F5"
INDIGO = "#4C5FD5"
AMBER = "#B45309"
RED = "#B42318"
GREEN = "#067647"

RISK_COLORS = {
    "low": "#98A2B3",
    "medium": "#D97706",
    "high": "#DC6803",
    "critical": "#B42318",
}
STATUS_COLORS = {
    "critical": RED,
    "below_reorder": AMBER,
    "adequate": GREEN,
    "excess": INDIGO,
}
ACTION_COLORS = {
    "reorder": ACCENT,
    "urgent_rotation": RED,
    "expiry_exposure": "#DC6803",
    "overstock": INDIGO,
    "sufficient": "#98A2B3",
    "monitor": "#98A2B3",
}

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
    }}
    .stApp {{ background: {CANVAS}; }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {SIDEBAR};
        border-right: 1px solid #1E293B;
    }}
    section[data-testid="stSidebar"] * {{ color: #CBD5E1 !important; }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong {{ color: #F8FAFC !important; }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #94A3B8 !important; font-size: 0.82rem;
    }}

    /* Typography */
    h1 {{ font-size: 1.55rem !important; font-weight: 700 !important; color: {INK} !important;
         letter-spacing: -0.02em; margin-bottom: 0.1rem !important; }}
    h2 {{ font-size: 1.05rem !important; font-weight: 600 !important; color: {INK} !important;
         margin-top: 1.5rem !important; }}
    h3 {{ font-size: 0.92rem !important; font-weight: 600 !important; color: {INK} !important; }}
    p, label, span {{ color: {INK}; }}

    .ms-page-sub {{ color: {MUTED}; font-size: 0.86rem; margin: 0 0 1.1rem 0; }}

    /* KPI card */
    .ms-kpi {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem 1.1rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
    }}
    .ms-kpi-label {{
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: {MUTED}; margin-bottom: 0.45rem;
    }}
    .ms-kpi-value {{
        font-size: 1.6rem; font-weight: 700; color: {INK};
        line-height: 1.15; letter-spacing: -0.02em;
    }}
    .ms-kpi-hint {{ font-size: 0.75rem; color: {MUTED}; margin-top: 0.35rem; }}

    /* Chips */
    .ms-chip {{
        display: inline-block; padding: 0.16rem 0.55rem; border-radius: 999px;
        font-size: 0.70rem; font-weight: 600; letter-spacing: 0.02em;
        border: 1px solid transparent;
    }}
    .ms-chip-low      {{ background:#F2F4F7; color:#475467; border-color:#E4E7EC; }}
    .ms-chip-medium   {{ background:#FFFAEB; color:#B54708; border-color:#FEDF89; }}
    .ms-chip-high     {{ background:#FEF0C7; color:#B54708; border-color:#FEC84B; }}
    .ms-chip-critical {{ background:#FEE4E2; color:#B42318; border-color:#FDA29B; }}
    .ms-chip-ok       {{ background:#ECFDF3; color:#067647; border-color:#ABEFC6; }}
    .ms-chip-warn     {{ background:#FFFAEB; color:#B54708; border-color:#FEDF89; }}
    .ms-chip-bad      {{ background:#FEE4E2; color:#B42318; border-color:#FDA29B; }}

    /* Callout */
    .ms-callout {{
        border-left: 3px solid {ACCENT};
        background: {ACCENT_SOFT};
        padding: 0.7rem 0.9rem; border-radius: 0 8px 8px 0;
        font-size: 0.82rem; color: #134E4A;
    }}
    .ms-callout-warn {{
        border-left-color: {AMBER}; background: #FFFAEB; color: #7A2E0E;
    }}
    .ms-note {{ font-size: 0.78rem; color: {MUTED}; }}

    /* Tables */
    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 0.85rem; font-weight: 600; color: {MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}

    /* Divider */
    hr {{ border-color: {BORDER}; }}

    .ms-footer {{
        margin-top: 2.5rem; padding-top: 0.9rem; border-top: 1px solid {BORDER};
        font-size: 0.72rem; color: {MUTED};
    }}
</style>
"""


def plotly_template() -> go.layout.Template:
    """One template so every chart shares type scale, grid and colour logic."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=INK),
        title=dict(font=dict(size=14, color=INK), x=0, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=[ACCENT, INDIGO, "#7C3AED", "#0EA5E9", "#F59E0B", "#EF4444"],
        margin=dict(l=48, r=24, t=44, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=BORDER),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor="#EDF1F7",
            zeroline=False,
            linecolor=BORDER,
            ticks="outside",
            tickcolor=BORDER,
            tickfont=dict(size=11, color=MUTED),
        ),
        yaxis=dict(
            gridcolor="#EDF1F7",
            zeroline=False,
            linecolor=BORDER,
            ticks="",
            tickfont=dict(size=11, color=MUTED),
        ),
    )
    return template


def install_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    pio.templates["medistock"] = plotly_template()
    pio.templates.default = "medistock"


PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
