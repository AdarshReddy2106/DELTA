"""
DELTA — Delivery Estimate, Lateness & Trust Analytics
Streamlit Dashboard  •  DS3030 Data Analytics Final Project
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────
# 0. PAGE CONFIG & DESIGN SYSTEM
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DELTA Analytics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color tokens
BG_CANVAS = "#0b0f17"
BG_SURFACE = "#111827"
BG_CARD = "#141c2e"
BORDER_SUBTLE = "rgba(148, 163, 184, 0.14)"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED = "#64748b"

CYAN = "#38bdf8"
GREEN = "#22c55e"
TEAL = "#2dd4bf"
AMBER = "#f59e0b"
ORANGE = "#fb923c"
RED = "#f43f5e"
INDIGO = "#818cf8"
SLATE_LINE = "#475569"
COLORS = [CYAN, GREEN, AMBER, RED, INDIGO, TEAL, ORANGE, "#a78bfa"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

/* Base canvas & typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: radial-gradient(circle at 18% 0%, rgba(56, 189, 248, 0.06) 0%, transparent 42%),
                radial-gradient(circle at 82% 0%, rgba(34, 197, 94, 0.04) 0%, transparent 40%),
                #0b0f17;
    color: #f1f5f9;
}

/* Top header bar — blend into dark canvas */
[data-testid="stHeader"] {
    background: rgba(11, 15, 23, 0.82) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
}

/* Main container breathing room */
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0e1420 !important;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}

/* Brand block in sidebar */
.sidebar-brand {
    padding: 14px 16px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(45, 212, 191, 0.05) 100%);
    border: 1px solid rgba(56, 189, 248, 0.22);
    margin-bottom: 12px;
}
.sidebar-brand-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
    margin: 0;
}
.sidebar-brand-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
    line-height: 1.35;
}

/* Hero banner */
.hero-wrap {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 18px 22px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(20, 28, 46, 0.92) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(148, 163, 184, 0.14);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    margin-bottom: 18px;
}
.hero-kicker {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 8px;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 4px 0 0 0;
}
.hero-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #cbd5e1;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.18);
    padding: 8px 14px;
    border-radius: 10px;
    white-space: nowrap;
}

/* Custom KPI cards */
.kpi-card {
    background: linear-gradient(180deg, #151e31 0%, #111827 100%);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-top: 3px solid var(--accent, #38bdf8);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    height: 100%;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover {
    border-color: rgba(148, 163, 184, 0.28);
    transform: translateY(-1px);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.15;
    white-space: nowrap;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Sub-metric cards inside tabs (native st.metric) */
[data-testid="stMetric"] {
    background: #131b2c;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 12px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}

/* Bordered chart containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 24, 39, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.13) !important;
    border-radius: 14px !important;
}

/* Tabs navigation bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #111827;
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 12px;
    padding: 5px;
    margin-bottom: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8 !important;
    background-color: transparent;
    border: none !important;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #f1f5f9 !important;
    background-color: rgba(148, 163, 184, 0.08);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.22) 0%, rgba(45, 212, 191, 0.16) 100%) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* Section banner */
.section-banner {
    padding: 12px 16px;
    border-radius: 12px;
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.09) 0%, rgba(17, 24, 39, 0.3) 100%);
    border-left: 3px solid #38bdf8;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    border-right: 1px solid rgba(148, 163, 184, 0.06);
    border-bottom: 1px solid rgba(148, 163, 184, 0.06);
    margin-bottom: 14px;
}
.section-kicker {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38bdf8;
}
.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 2px 0;
}
.section-sub {
    font-size: 0.84rem;
    color: #94a3b8;
    margin: 0;
}

/* Insight callout card */
.insight-card {
    padding: 14px 18px;
    border-radius: 12px;
    background: rgba(20, 28, 46, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-left: 3px solid var(--callout-accent, #38bdf8);
    margin-top: 10px;
}
.insight-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--callout-accent, #38bdf8);
    margin-bottom: 6px;
}
.insight-body {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.55;
}

/* Hide default chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
hr { border-color: rgba(148, 163, 184, 0.12) !important; margin: 14px 0 !important; }

/* Custom subtle scrollbars */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: #0b0f17; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# UI & CHART HELPERS
# ──────────────────────────────────────────────────────────────
def render_kpi_card(label: str, value: str, subtext: str = "", accent: str = CYAN):
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_banner(kicker: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="section-banner">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <p class="section-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(title: str, body: str, accent: str = CYAN):
    st.markdown(
        f"""
        <div class="insight-card" style="--callout-accent:{accent};">
            <div class="insight-title">💡 {title}</div>
            <div class="insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, title: str = "", height: int = 360, **extra) -> go.Figure:
    layout_cfg = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#cbd5e1"),
        margin=dict(l=20, r=20, t=50 if title else 24, b=28),
        height=height,
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#334155",
            font=dict(color="#f8fafc", size=12),
        ),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.08)",
            zerolinecolor="rgba(148, 163, 184, 0.14)",
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.08)",
            zerolinecolor="rgba(148, 163, 184, 0.14)",
        ),
    )
    if title:
        layout_cfg["title"] = dict(
            text=title,
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=14.5, color="#f8fafc"),
            x=0.01,
            xanchor="left",
            y=0.96,
        )
    layout_cfg.update(extra)
    fig.update_layout(**layout_cfg)
    return fig


def format_brl_compact(amount: float) -> tuple[str, str]:
    if amount >= 1_000_000:
        return f"R$ {amount / 1_000_000:.2f}M", f"Exact: R$ {amount:,.0f}"
    if amount >= 1_000:
        return f"R$ {amount / 1_000:.1f}K", f"Exact: R$ {amount:,.0f}"
    return f"R$ {amount:,.0f}", "Total merchandise revenue"


# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING (cached)
# ──────────────────────────────────────────────────────────────
DATA = Path("PROCESSED_DATA")


@st.cache_data(show_spinner="Loading order data …")
def load_orders():
    return pd.read_csv(
        DATA / "master_orders.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "purchase_date",
        ],
    )


@st.cache_data(show_spinner="Loading seller scorecard …")
def load_sellers():
    return pd.read_csv(DATA / "seller_scorecard.csv")


@st.cache_data(show_spinner="Loading time-series …")
def load_timeseries():
    return pd.read_csv(DATA / "daily_orders_smoothed.csv", parse_dates=["purchase_date"])


@st.cache_data(show_spinner="Loading route data …")
def load_routes():
    return pd.read_csv(DATA / "graph_route_performance.csv")


@st.cache_data(show_spinner="Loading state metrics …")
def load_state_metrics():
    return pd.read_csv(DATA / "graph_state_metrics.csv")


@st.cache_data(show_spinner="Loading NLP data …")
def load_nlp():
    top = pd.read_csv(DATA / "nlp_top_terms.csv")
    dist = pd.read_csv(DATA / "nlp_distinctive_terms.csv")
    return top, dist


@st.cache_data(show_spinner="Loading association rules …")
def load_rules():
    risk = pd.read_csv(DATA / "rules_bad_review_risk.csv")
    copurchase = pd.read_csv(DATA / "rules_category_copurchase.csv")
    return risk, copurchase


orders = load_orders()
sellers = load_sellers()
ts = load_timeseries()
routes = load_routes()
state_metrics = load_state_metrics()
nlp_top, nlp_dist = load_nlp()
rules_risk, rules_copurchase = load_rules()

BUCKET_ORDER = [
    "Early (>1d early)",
    "On Time (-1d to 0d)",
    "1-3 Days Late",
    "4-7 Days Late",
    ">7 Days Late",
]

# ──────────────────────────────────────────────────────────────
# 2. SIDEBAR — GLOBAL FILTERS (clean, clutter-free defaults)
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">📦 DELTA</div>
            <div class="sidebar-brand-sub">Delivery Estimate, Lateness & Trust Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Global Filters")

    years = sorted(orders["purchase_year"].dropna().unique().astype(int).tolist())
    sel_years = st.pills(
        "Purchase Year",
        options=years,
        selection_mode="multi",
        default=years,
        key="year_filter",
    )
    if not sel_years:
        sel_years = years

    delivery_scope = st.segmented_control(
        "Order Scope",
        options=["All", "Delivered", "Late Only"],
        default="All",
        key="scope_filter",
    )

    all_states = sorted(orders["customer_state"].dropna().unique().tolist())
    sel_states = st.multiselect(
        "Customer State",
        options=all_states,
        default=[],
        placeholder="All 27 states (click to narrow)",
        key="state_filter",
    )

    all_cats = orders["primary_category"].value_counts().index.tolist()
    sel_cats = st.multiselect(
        "Product Category",
        options=all_cats,
        default=[],
        placeholder=f"All {len(all_cats)} categories (click to narrow)",
        key="cat_filter",
    )

    # Apply global filters
    mask = orders["purchase_year"].isin(sel_years)
    if delivery_scope == "Delivered":
        mask &= orders["order_status"] == "delivered"
    elif delivery_scope == "Late Only":
        mask &= orders["is_late"] == 1
    if sel_states:
        mask &= orders["customer_state"].isin(sel_states)
    if sel_cats:
        mask &= orders["primary_category"].isin(sel_cats)

    filt = orders[mask].copy()

    st.markdown("---")
    coverage = len(filt) / len(orders) if len(orders) else 0.0
    st.caption(f"**Active Scope:** {len(filt):,} / {len(orders):,} orders ({coverage * 100:.1f}%)")
    st.progress(min(max(coverage, 0.0), 1.0))

    st.markdown(
        "<div style='color:#64748b;font-size:0.74rem;margin-top:14px;line-height:1.45;'>"
        "DS3030 • Data Analytics<br>Olist Brazilian E-Commerce (2016–2018)"
        "</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# 3. HERO HEADER & KPI CARDS
# ──────────────────────────────────────────────────────────────
active_years_lbl = ", ".join(str(y) for y in sorted(sel_years))
active_state_lbl = ", ".join(sel_states[:4]) + ("…" if len(sel_states) > 4 else "") if sel_states else "All 27 States"
active_cat_lbl = f"{len(sel_cats)} Categories" if sel_cats else f"All {len(all_cats)} Categories"

st.markdown(
    f"""
    <div class="hero-wrap">
        <div>
            <div class="hero-kicker">DS3030 • Olist E-Commerce Analytics</div>
            <h1 class="hero-title">DELTA — Delivery Estimate, Lateness &amp; Trust Analytics</h1>
            <p class="hero-sub">End-to-end breakdown of delivery bottlenecks, geographic risk routes, NLP complaint drivers, and seller clusters.</p>
        </div>
        <div class="hero-pill">
            {active_years_lbl} &nbsp;|&nbsp; {active_state_lbl} &nbsp;|&nbsp; {active_cat_lbl}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
rev_total = filt["total_price"].sum() if len(filt) else 0.0
rev_short, rev_sub = format_brl_compact(rev_total)
late_pct = filt["is_late"].mean() * 100 if filt["is_late"].notna().any() else 0.0
late_count = int((filt["is_late"] == 1).sum())
avg_rev = filt["review_score"].mean() if filt["review_score"].notna().any() else 0.0
avg_del = filt["total_delivery_days"].mean() if filt["total_delivery_days"].notna().any() else 0.0
med_del = filt["total_delivery_days"].median() if filt["total_delivery_days"].notna().any() else 0.0
avg_dist = filt["distance_km"].mean() if filt["distance_km"].notna().any() else 0.0
med_dist = filt["distance_km"].median() if filt["distance_km"].notna().any() else 0.0

with k1:
    render_kpi_card("Total Orders", f"{len(filt):,}", f"{len(filt)/len(orders)*100:.1f}% of dataset", CYAN)
with k2:
    render_kpi_card("Total Revenue", rev_short, rev_sub, GREEN)
with k3:
    render_kpi_card("Late Delivery %", f"{late_pct:.1f}%", f"{late_count:,} late orders", RED)
with k4:
    render_kpi_card("Avg Review Score", f"{avg_rev:.2f} ★", f"Median: {filt['review_score'].median():.0f} ★" if filt["review_score"].notna().any() else "—", AMBER)
with k5:
    render_kpi_card("Avg Delivery", f"{avg_del:.1f} d", f"Median: {med_del:.1f} days", TEAL)
with k6:
    render_kpi_card("Avg Distance", f"{avg_dist:.0f} km", f"Median: {med_dist:.0f} km", INDIGO)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

if len(filt) == 0:
    st.warning("No orders match the current sidebar filters. Clear or widen one of the filters to view charts.")
    st.stop()

# ──────────────────────────────────────────────────────────────
# 4. TABBED ANALYTICS WORKSPACE
# ──────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📦 Pipeline",
    "⏱️ Trends",
    "🗺️ Geography",
    "🛡️ Data Quality",
    "💬 Reviews & NLP",
    "🏪 Sellers",
    "⚠️ Risk Rules",
    "🔎 Explorer",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DELIVERY PIPELINE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    section_banner(
        "Research Question 1 • Stage Decomposition",
        "Where Do Delivery Delays Actually Come From?",
        "Splits end-to-end fulfillment into Payment Approval → Seller Handling → Carrier Transit (Notebook 01).",
    )

    stage_data = filt[["approval_hours", "seller_handling_days", "carrier_transit_days"]].dropna()
    col_a, col_b = st.columns([3, 2], gap="medium")

    with col_a:
        with st.container(border=True):
            if len(stage_data):
                app_days = stage_data["approval_hours"].median() / 24.0
                app_hrs = stage_data["approval_hours"].median()
                hand_days = stage_data["seller_handling_days"].median()
                trans_days = stage_data["carrier_transit_days"].median()

                stages_y = ["Payment Approval", "Seller Handling", "Carrier Transit"]
                stages_x = [app_days, hand_days, trans_days]
                stages_txt = [
                    f"{app_days:.2f}d ({app_hrs:.1f} hrs)",
                    f"{hand_days:.2f} days",
                    f"{trans_days:.2f} days",
                ]

                fig_pipe = go.Figure(go.Bar(
                    x=stages_x,
                    y=stages_y,
                    orientation="h",
                    marker=dict(color=[CYAN, AMBER, GREEN], cornerradius=6),
                    text=stages_txt,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Median: %{text}<extra></extra>",
                ))
                style_fig(
                    fig_pipe,
                    title="Median Duration per Delivery Stage (Unified in Days)",
                    height=330,
                    xaxis_title="Median Days",
                    yaxis=dict(autorange="reversed", showgrid=False),
                    margin=dict(l=20, r=90, t=52, b=36),
                )
                st.plotly_chart(fig_pipe, width="stretch")

    with col_b:
        with st.container(border=True):
            bucket_counts = filt["lateness_bucket"].value_counts().reindex(BUCKET_ORDER).dropna()
            if len(bucket_counts):
                fig_bucket = go.Figure(go.Pie(
                    labels=bucket_counts.index,
                    values=bucket_counts.values,
                    hole=0.58,
                    sort=False,
                    marker=dict(
                        colors=[GREEN, CYAN, AMBER, ORANGE, RED],
                        line=dict(color="#0b0f17", width=2),
                    ),
                    textinfo="percent",
                    textposition="inside",
                    insidetextorientation="horizontal",
                    hovertemplate="<b>%{label}</b><br>Orders: %{value:,}<br>Share: %{percent}<extra></extra>",
                ))
                style_fig(
                    fig_bucket,
                    title="Order Lateness Distribution",
                    height=330,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.04,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=11),
                    ),
                    margin=dict(l=16, r=16, t=48, b=56),
                )
                st.plotly_chart(fig_bucket, width="stretch")

    with st.container(border=True):
        stage_by_bucket = (
            filt.dropna(subset=["lateness_bucket", "approval_hours", "seller_handling_days", "carrier_transit_days"])
            .groupby("lateness_bucket")[["approval_hours", "seller_handling_days", "carrier_transit_days"]]
            .median()
            .reindex(BUCKET_ORDER)
            .dropna()
        )
        if len(stage_by_bucket):
            stage_by_bucket["approval_days"] = stage_by_bucket["approval_hours"] / 24.0
            fig_stack = go.Figure()
            for col, name, color in [
                ("approval_days", "Payment Approval", CYAN),
                ("seller_handling_days", "Seller Handling", AMBER),
                ("carrier_transit_days", "Carrier Transit", GREEN),
            ]:
                fig_stack.add_trace(go.Bar(
                    x=stage_by_bucket.index,
                    y=stage_by_bucket[col].round(2),
                    name=name,
                    marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.2f}} days<extra></extra>",
                ))
            style_fig(
                fig_stack,
                title="Stage-wise Median Breakdown Across Lateness Buckets",
                height=360,
                barmode="stack",
                yaxis_title="Median Days",
                legend=dict(orientation="h", y=1.12, x=0.01),
            )
            st.plotly_chart(fig_stack, width="stretch")

    insight_box(
        "Key Finding — Carrier Transit Dominates Delays",
        "<b>Carrier transit</b> is the single largest bottleneck (~7.1 days median vs. ~1.8 days seller handling and ~20 minutes payment approval). "
        "For orders arriving &gt;7 days late, courier transit balloons dramatically while approval and packing change far less—showing that carrier routing and courier SLA enforcement offer the biggest lever to cut late deliveries.",
        accent=GREEN,
    )

# ══════════════════════════════════════════════════════════════
# TAB 2 — TIME-SERIES TRENDS
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    section_banner(
        "Lecture 5 • Time-Series Smoothing & Seasonality",
        "Order Volume, Revenue & Late Rate Over Time",
        "Raw daily counts carry strong weekday-vs-weekend oscillations; SMA, EWMA, and Rolling Median filters expose true trend & Black Friday surges.",
    )

    with st.container(border=True):
        c_m, c_w = st.columns([2, 1])
        metric_labels = {
            "raw_orders": "Daily Orders",
            "raw_revenue": "Daily Revenue (R$)",
            "late_rate": "Daily Late Rate (%)",
            "avg_review": "Daily Avg Review Score",
        }
        with c_m:
            ts_metric = st.selectbox(
                "Time-Series Metric",
                list(metric_labels.keys()),
                format_func=lambda k: metric_labels[k],
                index=0,
            )
        with c_w:
            ts_window = st.segmented_control("Smoothing Window", ["7d", "14d", "30d"], default="14d") or "14d"

        w_num = int(ts_window.replace("d", ""))
        ts_plot = ts.copy()
        if ts_metric == "late_rate":
            ts_plot["raw_val"] = ts_plot["late_rate"] * 100
        else:
            ts_plot["raw_val"] = ts_plot[ts_metric]

        # Use precomputed columns for raw_orders; compute dynamically for other metrics so scales match!
        if ts_metric == "raw_orders":
            ts_plot["sma"] = ts_plot[f"sma_{ts_window}"]
            ts_plot["ewma"] = ts_plot[f"ewma_{ts_window}"]
            ts_plot["med"] = ts_plot[f"median_{ts_window}"]
        else:
            ts_plot["sma"] = ts_plot["raw_val"].rolling(window=w_num, min_periods=1).mean()
            ts_plot["ewma"] = ts_plot["raw_val"].ewm(span=w_num, adjust=False).mean()
            ts_plot["med"] = ts_plot["raw_val"].rolling(window=w_num, min_periods=1).median()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts_plot["purchase_date"], y=ts_plot["raw_val"],
            mode="lines", name="Raw Daily",
            line=dict(color="rgba(148, 163, 184, 0.32)", width=1),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.04)",
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_plot["purchase_date"], y=ts_plot["sma"],
            mode="lines", name=f"SMA ({ts_window})",
            line=dict(color=CYAN, width=2.2),
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_plot["purchase_date"], y=ts_plot["ewma"],
            mode="lines", name=f"EWMA ({ts_window})",
            line=dict(color=GREEN, width=2.2),
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_plot["purchase_date"], y=ts_plot["med"],
            mode="lines", name=f"Rolling Median ({ts_window})",
            line=dict(color=AMBER, width=2, dash="dash"),
        ))

        bf_date = pd.Timestamp("2017-11-24")
        fig_ts.add_shape(
            type="line", x0=bf_date, x1=bf_date, y0=0, y1=1, xref="x", yref="paper",
            line=dict(color=AMBER, width=1.5, dash="dot"),
        )
        fig_ts.add_annotation(
            x=bf_date, y=1.0, xref="x", yref="paper",
            text="Black Friday (Nov 24, 2017)",
            showarrow=False, font=dict(color=AMBER, size=11), yshift=10, xanchor="left",
        )
        style_fig(
            fig_ts,
            title=f"{metric_labels[ts_metric]} — Raw vs {ts_window} Smoothed Filters",
            height=420,
            yaxis_title=metric_labels[ts_metric],
            legend=dict(orientation="h", y=1.1, x=0.01),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ts, width="stretch")

    with st.container(border=True):
        monthly = (
            filt.groupby([
                filt["purchase_date"].dt.year.rename("year"),
                filt["purchase_date"].dt.month.rename("month"),
            ])
            .agg(
                order_count=("order_id", "count"),
                revenue=("total_price", "sum"),
                late_pct=("is_late", "mean"),
            )
            .reset_index()
        )
        monthly["late_pct"] = monthly["late_pct"] * 100

        heatmap_metric = st.segmented_control(
            "Monthly Calendar Heatmap Metric",
            options=["order_count", "revenue", "late_pct"],
            default="order_count",
            format_func=lambda x: {"order_count": "Orders", "revenue": "Revenue (R$)", "late_pct": "Late Rate (%)"}[x],
        ) or "order_count"

        if len(monthly):
            pivot = monthly.pivot(index="year", columns="month", values=heatmap_metric).fillna(0)
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            pivot.columns = [month_names[m - 1] for m in pivot.columns]

            if heatmap_metric == "late_pct":
                cscale, txt_fmt = [[0, "#0f2922"], [0.4, "#22c55e"], [0.7, "#f59e0b"], [1, "#f43f5e"]], ".1f"
            elif heatmap_metric == "revenue":
                cscale, txt_fmt = [[0, "#0f172a"], [0.5, "#0ea5e9"], [1, "#2dd4bf"]], ".0f"
            else:
                cscale, txt_fmt = [[0, "#0f172a"], [0.5, "#0284c7"], [1, "#38bdf8"]], ".0f"

            fig_heat = px.imshow(
                pivot.values,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index.tolist()],
                color_continuous_scale=cscale,
                text_auto=txt_fmt,
                aspect="auto",
                labels=dict(x="Month", y="Year", color=heatmap_metric),
            )
            style_fig(
                fig_heat,
                title=f"Monthly { {'order_count':'Orders','revenue':'Revenue (R$)','late_pct':'Late Rate (%)'}[heatmap_metric] } Heatmap",
                height=260,
            )
            st.plotly_chart(fig_heat, width="stretch")

    insight_box(
        "Key Finding — Demand Surges Trigger Subsequent Delay Waves",
        "Daily volume scales rapidly through 2017 into early 2018 with a sharp spike on <b>Black Friday (Nov 24, 2017)</b>. "
        "Notice how a 14–30 day Rolling Median smooths weekly dips without getting dragged upward by single-day promo spikes, and how late rates climb right after peak seasonal surges when carrier capacity saturates.",
        accent=CYAN,
    )

# ══════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC INTELLIGENCE
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    section_banner(
        "Lecture 10 • Shipping Network Graph & Route Risk",
        "Shipping Corridors, Hub States & Geographic Delay Patterns",
        "27-state directed shipping network (`Seller State → Customer State`) combined with Haversine distance and intra-state vs. cross-state breakdowns.",
    )

    col_g1, col_g2 = st.columns(2, gap="medium")

    with col_g1:
        with st.container(border=True):
            state_agg = (
                filt.groupby("customer_state")
                .agg(orders=("order_id", "count"), late_pct=("is_late", "mean"), avg_review=("review_score", "mean"))
                .reset_index()
                .sort_values("orders", ascending=False)
                .head(15)
            )
            fig_state = make_subplots(specs=[[{"secondary_y": True}]])
            fig_state.add_trace(
                go.Bar(
                    x=state_agg["customer_state"], y=state_agg["orders"],
                    name="Orders", marker_color=CYAN, opacity=0.82,
                ),
                secondary_y=False,
            )
            fig_state.add_trace(
                go.Scatter(
                    x=state_agg["customer_state"], y=(state_agg["late_pct"] * 100).round(1),
                    name="Late Rate %", mode="lines+markers",
                    line=dict(color=RED, width=2.4), marker=dict(size=7, color=RED),
                ),
                secondary_y=True,
            )
            style_fig(
                fig_state,
                title="Top 15 Destination States: Volume vs Late %",
                height=380,
                legend=dict(orientation="h", y=1.12, x=0.01),
            )
            fig_state.update_yaxes(title_text="Order Count", secondary_y=False, gridcolor="rgba(148,163,184,0.08)")
            fig_state.update_yaxes(title_text="Late Rate (%)", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_state, width="stretch")

    with col_g2:
        with st.container(border=True):
            dist_df = filt.dropna(subset=["distance_km", "total_delivery_days", "review_score"])
            if len(dist_df):
                sample = dist_df.sample(n=min(3000, len(dist_df)), random_state=42)
                fig_dist = px.scatter(
                    sample,
                    x="distance_km",
                    y="total_delivery_days",
                    color="review_score",
                    color_continuous_scale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                    opacity=0.55,
                    labels={"distance_km": "Distance (km)", "total_delivery_days": "Delivery Days", "review_score": "Stars"},
                )
                style_fig(fig_dist, title="Shipping Distance vs Actual Delivery Days", height=380)
                st.plotly_chart(fig_dist, width="stretch")

    col_g3, col_g4 = st.columns([3, 2], gap="medium")

    with col_g3:
        with st.container(border=True):
            st.markdown("**Worst Shipping Routes (`Seller State → Customer State`)**")
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                min_orders = st.slider("Min orders/route", 20, 500, 100, step=20)
            with rc2:
                sort_metric = st.segmented_control(
                    "Rank worst by",
                    options=["late_rate", "avg_days_late", "avg_review"],
                    default="late_rate",
                    format_func=lambda x: {"late_rate": "Highest Late %", "avg_days_late": "Avg Days Late", "avg_review": "Lowest Review"}[x],
                ) or "late_rate"

            routes_filt = routes[routes["order_count"] >= min_orders].copy()
            # Ascending only when sorting by lowest avg_review; descending for highest late_rate / avg_days_late
            sort_asc = sort_metric == "avg_review"
            worst = routes_filt.sort_values(sort_metric, ascending=sort_asc).head(15).copy()

            if len(worst):
                worst["route"] = worst["seller_state"] + " → " + worst["customer_state"]
                worst["late_pct_disp"] = worst["late_rate"] * 100
                x_col = "late_pct_disp" if sort_metric == "late_rate" else sort_metric
                x_title = "Late Rate (%)" if sort_metric == "late_rate" else sort_metric.replace("_", " ").title()
                txt_vals = worst[x_col].round(1).astype(str) + ("%" if sort_metric == "late_rate" else "")

                fig_worst = px.bar(
                    worst,
                    x=x_col,
                    y="route",
                    orientation="h",
                    color=x_col,
                    color_continuous_scale="RdYlGn" if sort_metric == "avg_review" else "RdYlGn_r",
                    text=txt_vals,
                    custom_data=["order_count", "avg_review", "late_pct_disp"],
                )
                fig_worst.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Orders: %{customdata[0]:,}<br>Late Rate: %{customdata[2]:.1f}%<br>Avg Review: %{customdata[1]:.2f} ★<extra></extra>",
                )
                style_fig(
                    fig_worst,
                    title=f"Top 15 Worst Routes ({len(routes_filt)} routes with ≥{min_orders} orders)",
                    height=440,
                    xaxis_title=x_title,
                    yaxis_title="",
                    yaxis=dict(autorange="reversed", showgrid=False),
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=60, t=50, b=36),
                )
                st.plotly_chart(fig_worst, width="stretch")
            else:
                st.info("No routes meet the minimum order count threshold.")

    with col_g4:
        with st.container(border=True):
            st.markdown("**State Network Centrality vs Destination Risk**")
            sm_plot = state_metrics.copy()
            sm_plot["dest_late_pct"] = sm_plot["dest_late_rate"] * 100
            sm_plot["total_flow"] = sm_plot["in_strength"] + sm_plot["out_strength"]
            fig_net = px.scatter(
                sm_plot,
                x="pagerank",
                y="dest_late_pct",
                size="total_flow",
                color="dest_avg_review",
                text="state",
                color_continuous_scale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                labels={
                    "pagerank": "Graph PageRank (Hub Centrality)",
                    "dest_late_pct": "Inbound Late Rate (%)",
                    "dest_avg_review": "Avg Review",
                },
                size_max=40,
            )
            fig_net.update_traces(textposition="top center", textfont=dict(size=10, color="#e2e8f0"))
            style_fig(
                fig_net,
                title="27-State Network: PageRank Hub vs Delay Risk",
                height=340,
                margin=dict(l=20, r=20, t=48, b=32),
            )
            st.plotly_chart(fig_net, width="stretch")

            # Same-state vs Cross-state comparison table/cards
            cross = (
                filt.groupby("is_cross_state")
                .agg(
                    orders=("order_id", "count"),
                    avg_delivery=("total_delivery_days", "mean"),
                    late_pct=("is_late", "mean"),
                    avg_review=("review_score", "mean"),
                )
                .reset_index()
            )
            cross["Route Type"] = cross["is_cross_state"].map({0: "Same State", 1: "Cross State"})
            cross["Late %"] = (cross["late_pct"] * 100).round(1).astype(str) + "%"
            cross["Delivery (d)"] = cross["avg_delivery"].round(1)
            cross["Avg Review"] = cross["avg_review"].round(2)
            st.dataframe(
                cross[["Route Type", "orders", "Delivery (d)", "Late %", "Avg Review"]].rename(columns={"orders": "Orders"}),
                width="stretch",
                hide_index=True,
            )

    insight_box(
        "Key Finding — São Paulo Hub Dominance & Northeast/North Bottlenecks",
        "<b>SP</b> is the network's overwhelming shipping origin (highest PageRank and ~61% of inter-state outbound flow). "
        "While intra-state orders keep late rates low (~6.1%), long-haul corridors from SP to Northern/Northeastern states like <b>AL (26.3% late)</b> and <b>MA (21.2% late)</b> suffer severe delays—and high-volume corridors like <b>SP → RJ (15.5% late)</b> drive huge absolute complaint counts.",
        accent=ORANGE,
    )

# ══════════════════════════════════════════════════════════════
# TAB 4 — DATA QUALITY (OUTLIERS & IMPUTATION)
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    section_banner(
        "Lectures 6, 7 & 8 • Data Quality, Sensitivity & Imputation",
        "Outlier Detection, Distributional Sensitivity & Missing-Value Experiments",
        "Compares IQR vs Z-score flagging, measures Winsorization & Log1p stabilization, and evaluates Mean, Median, kNN, and MICE imputation.",
    )

    subtab1, subtab2, subtab3 = st.tabs([
        "🎯 Outlier Detection",
        "⚖️ Sensitivity & Transforms",
        "🧪 Missing Data & Imputation",
    ])

    with subtab1:
        with st.container(border=True):
            col_o1, col_o2, col_o3 = st.columns([2, 1, 1])
            var_labels = {
                "total_price": "Total Price (R$)",
                "total_freight": "Total Freight Value (R$)",
                "total_weight_g": "Product Weight (grams)",
                "total_delivery_days": "Delivery Time (days)",
                "days_late": "Days Late (days)",
                "distance_km": "Customer-Seller Distance (km)",
                "approval_hours": "Approval Time (hours)",
            }
            with col_o1:
                outlier_var = st.selectbox(
                    "Select Variable to Analyze",
                    list(var_labels.keys()),
                    format_func=lambda x: var_labels.get(x, x),
                    key="outlier_var_select",
                )
            with col_o2:
                iqr_mult = st.slider("IQR Multiplier (k)", 1.0, 3.0, 1.5, 0.25, key="iqr_mult_slider")
            with col_o3:
                z_thresh = st.slider("Z-Score Cutoff (|z|)", 2.0, 4.0, 3.0, 0.25, key="z_thresh_slider")

            series = filt[outlier_var].dropna()
            if len(series) > 0:
                q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
                iqr_val = q3 - q1
                iqr_lower, iqr_upper = q1 - iqr_mult * iqr_val, q3 + iqr_mult * iqr_val
                iqr_mask = (series < iqr_lower) | (series > iqr_upper)

                mean_val = float(series.mean())
                std_val = float(series.std()) if float(series.std()) > 0 else 1.0
                z_mask = ((series - mean_val) / std_val).abs() > z_thresh
                overlap_count = int((iqr_mask & z_mask).sum())

                mo1, mo2, mo3, mo4 = st.columns(4)
                mo1.metric("Filtered Sample Size", f"{len(series):,}")
                mo2.metric(f"IQR Outliers (k={iqr_mult})", f"{int(iqr_mask.sum()):,} ({iqr_mask.mean()*100:.1f}%)")
                mo3.metric(f"Z-Score Outliers (|z|>{z_thresh})", f"{int(z_mask.sum()):,} ({z_mask.mean()*100:.1f}%)")
                mo4.metric("Agreement (Overlap)", f"{overlap_count:,} ({overlap_count/len(series)*100:.1f}%)")

                vh1, vh2 = st.columns(2)
                plot_sample = series.sample(min(5000, len(series)), random_state=42)
                with vh1:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=plot_sample, nbinsx=50, marker_color=CYAN, opacity=0.78, name="Distribution"))
                    fig_hist.add_vline(x=iqr_upper, line_dash="dash", line_color=RED)
                    fig_hist.add_annotation(x=iqr_upper, y=1.0, yref="paper", text=f"IQR Upper ({iqr_upper:.1f})", showarrow=False, font=dict(color=RED, size=10), yshift=10)
                    z_upper = mean_val + z_thresh * std_val
                    fig_hist.add_vline(x=z_upper, line_dash="dot", line_color=AMBER)
                    fig_hist.add_annotation(x=z_upper, y=0.86, yref="paper", text=f"Z-Cutoff ({z_upper:.1f})", showarrow=False, font=dict(color=AMBER, size=10))
                    style_fig(fig_hist, title=f"Distribution & Fences: {var_labels[outlier_var]}", height=330, xaxis_title=var_labels[outlier_var], yaxis_title="Count")
                    st.plotly_chart(fig_hist, width="stretch")

                with vh2:
                    fig_box = go.Figure(go.Box(
                        y=plot_sample,
                        name=var_labels[outlier_var],
                        boxpoints="outliers",
                        marker=dict(color=ORANGE, size=4, opacity=0.6),
                        line_color=TEAL,
                    ))
                    style_fig(fig_box, title="Box Plot (Whiskers vs Flagged Points)", height=330)
                    st.plotly_chart(fig_box, width="stretch")

        with st.container(border=True):
            st.markdown("**Full-Dataset Outlier Benchmark (Notebook 03)**")
            outlier_summary_data = pd.DataFrame([
                {"Feature": "Total Price (R$)", "IQR Count": 7913, "IQR %": "8.02%", "Z-Score Count": 1709, "Z-Score %": "1.73%", "Interpretation": "Z-score flags only extreme luxury purchases"},
                {"Feature": "Total Freight Value (R$)", "IQR Count": 9941, "IQR %": "10.08%", "Z-Score Count": 1581, "Z-Score %": "1.60%", "Interpretation": "Cross-state heavy / Amazonian routes"},
                {"Feature": "Product Weight (grams)", "IQR Count": 13804, "IQR %": "13.99%", "Z-Score Count": 1962, "Z-Score %": "1.99%", "Interpretation": "Furniture & appliances create heavy right skew"},
                {"Feature": "Total Delivery Days", "IQR Count": 4899, "IQR %": "5.08%", "Z-Score Count": 1571, "Z-Score %": "1.63%", "Interpretation": "Severe carrier disruptions / strikes"},
                {"Feature": "Days Late", "IQR Count": 4875, "IQR %": "5.05%", "Z-Score Count": 1333, "Z-Score %": "1.38%", "Interpretation": "Orders delayed >2–3 weeks"},
                {"Feature": "Distance (km)", "IQR Count": 7622, "IQR %": "7.76%", "Z-Score Count": 1564, "Z-Score %": "1.59%", "Interpretation": "Transcontinental shipments (~3,500+ km)"},
            ])
            st.dataframe(outlier_summary_data, width="stretch", hide_index=True)

    with subtab2:
        sens_df = pd.DataFrame([
            {"Variable": "Total Price (R$)", "Treatment": "Original", "Count": 98666, "Mean": 137.75, "Median": 86.90, "Std Dev": 210.65, "Skewness": 9.73},
            {"Variable": "Total Price (R$)", "Treatment": "Winsorized (1-99%)", "Count": 98666, "Mean": 131.82, "Median": 86.90, "Std Dev": 155.95, "Skewness": 3.28},
            {"Variable": "Total Price (R$)", "Treatment": "Log Transformed (log1p)", "Count": 98666, "Mean": 4.46, "Median": 4.48, "Std Dev": 0.92, "Skewness": 0.29},
            {"Variable": "Total Price (R$)", "Treatment": "IQR Truncated", "Count": 90753, "Mean": 94.44, "Median": 79.00, "Std Dev": 66.06, "Skewness": 1.03},
            {"Variable": "Total Freight Value (R$)", "Treatment": "Original", "Count": 98666, "Mean": 22.82, "Median": 17.17, "Std Dev": 21.65, "Skewness": 12.05},
            {"Variable": "Total Freight Value (R$)", "Treatment": "Winsorized (1-99%)", "Count": 98666, "Mean": 22.28, "Median": 17.17, "Std Dev": 16.31, "Skewness": 2.75},
            {"Variable": "Total Freight Value (R$)", "Treatment": "Log Transformed (log1p)", "Count": 98666, "Mean": 2.99, "Median": 2.90, "Std Dev": 0.56, "Skewness": 0.47},
            {"Variable": "Total Freight Value (R$)", "Treatment": "IQR Truncated", "Count": 88725, "Mean": 17.79, "Median": 16.34, "Std Dev": 7.16, "Skewness": 0.92},
            {"Variable": "Total Delivery Days", "Treatment": "Original", "Count": 96476, "Mean": 12.56, "Median": 10.22, "Std Dev": 9.55, "Skewness": 3.83},
            {"Variable": "Total Delivery Days", "Treatment": "Winsorized (1-99%)", "Count": 96476, "Mean": 12.40, "Median": 10.22, "Std Dev": 8.41, "Skewness": 1.62},
            {"Variable": "Total Delivery Days", "Treatment": "Log Transformed (log1p)", "Count": 96476, "Mean": 2.42, "Median": 2.42, "Std Dev": 0.60, "Skewness": 0.04},
            {"Variable": "Total Delivery Days", "Treatment": "IQR Truncated", "Count": 91577, "Mean": 11.05, "Median": 9.91, "Std Dev": 6.09, "Skewness": 0.77},
            {"Variable": "Distance (km)", "Treatment": "Original", "Count": 98174, "Mean": 601.29, "Median": 433.74, "Std Dev": 593.90, "Skewness": 1.65},
            {"Variable": "Distance (km)", "Treatment": "Winsorized (1-99%)", "Count": 98174, "Mean": 598.81, "Median": 433.74, "Std Dev": 584.94, "Skewness": 1.58},
            {"Variable": "Distance (km)", "Treatment": "Log Transformed (log1p)", "Count": 98174, "Mean": 5.78, "Median": 6.07, "Std Dev": 1.37, "Skewness": -1.04},
            {"Variable": "Distance (km)", "Treatment": "IQR Truncated", "Count": 90552, "Mean": 466.01, "Median": 392.48, "Std Dev": 373.89, "Skewness": 0.96},
        ])

        with st.container(border=True):
            sel_sens_var = st.selectbox("Filter Variable for Treatment Comparison", sens_df["Variable"].unique(), index=0)
            sub_sens = sens_df[sens_df["Variable"] == sel_sens_var]

            col_st1, col_st2 = st.columns([3, 2])
            with col_st1:
                st.dataframe(
                    sub_sens.style.background_gradient(subset=["Skewness"], cmap="RdYlGn_r").format(
                        {"Mean": "{:.2f}", "Median": "{:.2f}", "Std Dev": "{:.2f}", "Skewness": "{:.2f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
            with col_st2:
                fig_skew = px.bar(
                    sub_sens,
                    x="Treatment", y="Skewness",
                    color="Treatment",
                    color_discrete_sequence=[RED, AMBER, GREEN, CYAN],
                    text=sub_sens["Skewness"].round(2),
                )
                fig_skew.update_traces(textposition="outside", cliponaxis=False)
                style_fig(fig_skew, title=f"Skewness Under Treatments: {sel_sens_var}", height=280, showlegend=False)
                st.plotly_chart(fig_skew, width="stretch")

        with st.container(border=True):
            pos_prices = filt[filt["total_price"] > 0]["total_price"].dropna()
            if len(pos_prices):
                sample_sens = pos_prices.sample(min(3000, len(pos_prices)), random_state=42)
                p1, p99 = sample_sens.quantile(0.01), sample_sens.quantile(0.99)
                fig_treat = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=["Raw Distribution", "Winsorized (1–99%)", "Log1p Transformed"],
                )
                fig_treat.add_trace(go.Histogram(x=sample_sens, nbinsx=40, marker_color=RED, opacity=0.8), row=1, col=1)
                fig_treat.add_trace(go.Histogram(x=sample_sens.clip(p1, p99), nbinsx=40, marker_color=AMBER, opacity=0.8), row=1, col=2)
                fig_treat.add_trace(go.Histogram(x=np.log1p(sample_sens), nbinsx=40, marker_color=GREEN, opacity=0.8), row=1, col=3)
                style_fig(fig_treat, title="Live Filtered Price Distribution Under Treatments", height=310, showlegend=False)
                st.plotly_chart(fig_treat, width="stretch")

    with subtab3:
        col_m1, col_m2 = st.columns(2, gap="medium")
        with col_m1:
            with st.container(border=True):
                comm_by_score = filt.dropna(subset=["review_score"]).groupby("review_score")["has_comment"].mean().reset_index()
                if len(comm_by_score):
                    comm_by_score["Stars"] = comm_by_score["review_score"].astype(int).astype(str) + " ★"
                    comm_by_score["comment_rate"] = (comm_by_score["has_comment"] * 100).round(1)
                    fig_comm_score = px.bar(
                        comm_by_score, x="Stars", y="comment_rate",
                        color="comment_rate", color_continuous_scale=[[0, CYAN], [0.5, AMBER], [1, RED]],
                        text="comment_rate",
                    )
                    fig_comm_score.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
                    style_fig(fig_comm_score, title="Comment Rate by Star Rating (MNAR Bias)", height=340, yaxis_range=[0, 95], yaxis_title="% With Written Comment", coloraxis_showscale=False)
                    st.plotly_chart(fig_comm_score, width="stretch")

        with col_m2:
            with st.container(border=True):
                comm_by_late = filt.groupby("lateness_bucket")["has_comment"].mean().reindex(BUCKET_ORDER).dropna().reset_index()
                if len(comm_by_late):
                    comm_by_late["comment_rate"] = (comm_by_late["has_comment"] * 100).round(1)
                    fig_comm_late = px.bar(
                        comm_by_late, x="lateness_bucket", y="comment_rate",
                        color="comment_rate", color_continuous_scale=[[0, CYAN], [0.5, AMBER], [1, RED]],
                        text="comment_rate",
                    )
                    fig_comm_late.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
                    style_fig(fig_comm_late, title="Comment Rate by Delivery Lateness", height=340, yaxis_range=[0, 85], yaxis_title="% With Written Comment", xaxis_title="", coloraxis_showscale=False)
                    st.plotly_chart(fig_comm_late, width="stretch")

        with st.container(border=True):
            imputation_results = pd.DataFrame([
                {"Feature": "Product Weight (g)", "Method": "Mean", "MAE": 2550.90, "RMSE": 4368.12},
                {"Feature": "Product Weight (g)", "Method": "Median", "MAE": 1964.98, "RMSE": 4649.87},
                {"Feature": "Product Weight (g)", "Method": "kNN (k=5)", "MAE": 2130.60, "RMSE": 3944.67},
                {"Feature": "Product Weight (g)", "Method": "MICE (Iterative)", "MAE": 2300.23, "RMSE": 3523.95},
                {"Feature": "Product Length (cm)", "Method": "Mean", "MAE": 12.65, "RMSE": 16.77},
                {"Feature": "Product Length (cm)", "Method": "Median", "MAE": 11.92, "RMSE": 17.74},
                {"Feature": "Product Length (cm)", "Method": "kNN (k=5)", "MAE": 11.02, "RMSE": 15.91},
                {"Feature": "Product Length (cm)", "Method": "MICE (Iterative)", "MAE": 10.68, "RMSE": 15.06},
                {"Feature": "Product Height (cm)", "Method": "Mean", "MAE": 9.68, "RMSE": 13.89},
                {"Feature": "Product Height (cm)", "Method": "Median", "MAE": 9.29, "RMSE": 14.52},
                {"Feature": "Product Height (cm)", "Method": "kNN (k=5)", "MAE": 9.51, "RMSE": 13.77},
                {"Feature": "Product Height (cm)", "Method": "MICE (Iterative)", "MAE": 9.25, "RMSE": 12.76},
                {"Feature": "Product Width (cm)", "Method": "Mean", "MAE": 9.40, "RMSE": 12.04},
                {"Feature": "Product Width (cm)", "Method": "Median", "MAE": 8.77, "RMSE": 12.45},
                {"Feature": "Product Width (cm)", "Method": "kNN (k=5)", "MAE": 7.43, "RMSE": 10.71},
                {"Feature": "Product Width (cm)", "Method": "MICE (Iterative)", "MAE": 7.75, "RMSE": 10.62},
            ])
            c_is, c_im = st.columns([2, 1])
            with c_is:
                sel_imp_feat = st.selectbox("Masked Feature (15% Masking Experiment, n=4,942)", imputation_results["Feature"].unique(), index=0)
            with c_im:
                sel_imp_metric = st.segmented_control("Error Metric", ["RMSE", "MAE"], default="RMSE") or "RMSE"

            sub_imp = imputation_results[imputation_results["Feature"] == sel_imp_feat]
            fig_imp = px.bar(
                sub_imp, x="Method", y=sel_imp_metric, color="Method",
                color_discrete_sequence=[CYAN, AMBER, INDIGO, GREEN],
                text=sub_imp[sel_imp_metric].round(2),
            )
            fig_imp.update_traces(textposition="outside", cliponaxis=False)
            style_fig(fig_imp, title=f"Imputation Error ({sel_imp_metric}) — {sel_imp_feat}", height=320, showlegend=False)
            st.plotly_chart(fig_imp, width="stretch")

    insight_box(
        "Key Findings — Skewness Control, Non-Random Missingness & MICE Advantage",
        "1. <b>Log1p stabilization</b> cuts price skewness from 9.73 → 0.29 and freight skewness from 12.05 → 0.47 without dropping valid extreme orders.<br>"
        "2. <b>Missing comments are informative (MNAR):</b> ~77.6% of 1★ reviewers leave text vs. ~35.9% of 5★ reviewers, and &gt;7-day late orders leave comments ~65.4% of the time.<br>"
        "3. <b>MICE wins on correlated dimensions</b> (lowest RMSE across weight, length, height, and width in the 15% masking test), while Median gets lowest MAE on heavily skewed product weight.",
        accent=INDIGO,
    )

# ══════════════════════════════════════════════════════════════
# TAB 5 — REVIEW & SENTIMENT ANALYSIS (NLP)
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    section_banner(
        "Research Questions 2 & 4 • Rating Cliff & Portuguese NLP",
        "How Late Is Too Late — And What Do Customers Complain About?",
        "Combines delivery-tolerance thresholds (Notebook 02) with negation-preserving Portuguese TF-IDF analysis on 40,556 review texts (Notebook 04).",
    )

    col_r1, col_r2 = st.columns(2, gap="medium")
    with col_r1:
        with st.container(border=True):
            review_dist = filt["review_score"].dropna().astype(int).value_counts().sort_index()
            score_cmap = {1: RED, 2: ORANGE, 3: AMBER, 4: CYAN, 5: GREEN}
            fig_rev = go.Figure(go.Bar(
                x=[f"{s} ★" for s in review_dist.index],
                y=review_dist.values,
                marker=dict(color=[score_cmap.get(s, CYAN) for s in review_dist.index], cornerradius=6),
                text=[f"{v:,}" for v in review_dist.values],
                textposition="outside",
                cliponaxis=False,
            ))
            style_fig(fig_rev, title="Review Score Distribution", height=340, xaxis_title="Rating", yaxis_title="Orders", margin=dict(l=20, r=20, t=50, b=36))
            st.plotly_chart(fig_rev, width="stretch")

    with col_r2:
        with st.container(border=True):
            late_review = (
                filt.dropna(subset=["lateness_bucket", "review_score"])
                .groupby("lateness_bucket")["review_score"].mean()
                .reindex(BUCKET_ORDER)
                .dropna()
            )
            fig_lr = go.Figure(go.Bar(
                x=late_review.index,
                y=late_review.values,
                marker=dict(color=[GREEN, CYAN, AMBER, ORANGE, RED], cornerradius=6),
                text=[f"{v:.2f} ★" for v in late_review.values],
                textposition="outside",
                cliponaxis=False,
            ))
            style_fig(fig_lr, title="Avg Review Score by Lateness Bucket (The Patience Cliff)", height=340, yaxis_range=[0, 5.4], yaxis_title="Avg Stars")
            st.plotly_chart(fig_lr, width="stretch")

    with st.container(border=True):
        nlp_mode = st.segmented_control(
            "Compare Top TF-IDF Terms By",
            options=["1-star vs 5-star", "late vs on-time"],
            default="1-star vs 5-star",
        ) or "1-star vs 5-star"

        g_left, g_right = ("1-star", "5-star") if nlp_mode == "1-star vs 5-star" else ("late", "on-time")
        lbl_left = "1★ Complaint Reviews" if g_left == "1-star" else "Late Delivery Reviews"
        lbl_right = "5★ Praise Reviews" if g_right == "5-star" else "On-Time Delivery Reviews"

        col_n1, col_n2 = st.columns(2, gap="medium")
        with col_n1:
            t_left = nlp_top[nlp_top["group"] == g_left].head(15)
            if len(t_left):
                fig_tl = go.Figure(go.Bar(
                    x=t_left["mean_tfidf"], y=t_left["term"], orientation="h",
                    marker=dict(color=RED, cornerradius=4),
                    text=t_left["mean_tfidf"].round(4), textposition="outside", cliponaxis=False,
                ))
                style_fig(fig_tl, title=f"Top 15 Terms — {lbl_left}", height=410, xaxis_title="Mean TF-IDF", yaxis=dict(autorange="reversed", showgrid=False), margin=dict(l=20, r=60, t=48, b=30))
                st.plotly_chart(fig_tl, width="stretch")

        with col_n2:
            t_right = nlp_top[nlp_top["group"] == g_right].head(15)
            if len(t_right):
                fig_tr = go.Figure(go.Bar(
                    x=t_right["mean_tfidf"], y=t_right["term"], orientation="h",
                    marker=dict(color=GREEN, cornerradius=4),
                    text=t_right["mean_tfidf"].round(4), textposition="outside", cliponaxis=False,
                ))
                style_fig(fig_tr, title=f"Top 15 Terms — {lbl_right}", height=410, xaxis_title="Mean TF-IDF", yaxis=dict(autorange="reversed", showgrid=False), margin=dict(l=20, r=60, t=48, b=30))
                st.plotly_chart(fig_tr, width="stretch")

    with st.container(border=True):
        top_n = st.slider("Terms per side (Diverging TF-IDF Difference)", 8, 25, 12)
        neg_dist = nlp_dist.sort_values("difference", ascending=False).head(top_n)
        pos_dist = nlp_dist.sort_values("difference", ascending=True).head(top_n)
        div_df = pd.concat([neg_dist, pos_dist]).sort_values("difference", ascending=True)
        div_colors = [GREEN if d < 0 else RED for d in div_df["difference"]]

        fig_div = go.Figure(go.Bar(
            x=div_df["difference"],
            y=div_df["term"],
            orientation="h",
            marker=dict(color=div_colors, cornerradius=4),
            hovertemplate="<b>%{y}</b><br>1★ − 5★ TF-IDF Diff: %{x:.4f}<extra></extra>",
        ))
        fig_div.add_vline(x=0, line_color="#64748b", line_width=1.2)
        style_fig(
            fig_div,
            title="Most Distinctive Terms: Praise-Weighted (← Left) vs Complaint-Weighted (Right →)",
            height=max(420, top_n * 24),
            xaxis_title="Mean TF-IDF Difference (1★ minus 5★)",
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_div, width="stretch")

    insight_box(
        "Key Finding — 3-Day Patience Threshold & Negation-Driven Complaints",
        "Customer ratings drop from <b>4.29★</b> (early/on-time) to <b>3.77★</b> at 1–3 days late, then collapse to <b>2.32★</b> (4–7 days late) and <b>1.73★</b> (&gt;7 days late). "
        "Preserving Portuguese negation words (<i>não</i>, <i>nunca</i>, <i>sem</i>) reveals that <b>\"não recebi\"</b> (did not receive) and <b>\"não entregue\"</b> (not delivered) are the strongest drivers of 1★ complaints.",
        accent=RED,
    )

# ══════════════════════════════════════════════════════════════
# TAB 6 — SELLER SEGMENTATION
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    section_banner(
        "Lectures 13–21 • Dimensionality Reduction & Clustering",
        "Seller Performance Archetypes & Reliability Scorecard",
        "3,095 marketplace sellers reduced across 21 operational features and segmented into 4 actionable tiers (Notebook 05).",
    )

    arch_colors = {
        "Tier 1: Star High-Volume Seller": GREEN,
        "Tier 2: Premium / High-Ticket Seller": CYAN,
        "Tier 3: Standard Marketplace Seller": AMBER,
        "Tier 4: At-Risk / Slow Seller": RED,
    }

    col_s1, col_s2 = st.columns([2, 3], gap="medium")
    with col_s1:
        with st.container(border=True):
            arch_counts = sellers["seller_archetype"].value_counts()
            fig_arch = go.Figure(go.Pie(
                labels=arch_counts.index,
                values=arch_counts.values,
                hole=0.56,
                marker=dict(
                    colors=[arch_colors.get(a, INDIGO) for a in arch_counts.index],
                    line=dict(color="#0b0f17", width=2),
                ),
                textinfo="percent",
                textposition="inside",
                hovertemplate="<b>%{label}</b><br>Sellers: %{value:,} (%{percent})<extra></extra>",
            ))
            style_fig(
                fig_arch,
                title="Seller Archetype Breakdown (3,095 Sellers)",
                height=390,
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5, font=dict(size=10.5)),
                margin=dict(l=16, r=16, t=48, b=72),
            )
            st.plotly_chart(fig_arch, width="stretch")

    with col_s2:
        with st.container(border=True):
            fig_pca = px.scatter(
                sellers.dropna(subset=["pca_dim1", "pca_dim2"]),
                x="pca_dim1", y="pca_dim2",
                color="seller_archetype",
                color_discrete_map=arch_colors,
                opacity=0.65,
                hover_data=["seller_state", "total_orders", "total_revenue", "late_rate", "avg_review_score", "reliability_score"],
                labels={"pca_dim1": "PCA Component 1", "pca_dim2": "PCA Component 2", "seller_archetype": "Tier"},
            )
            style_fig(
                fig_pca,
                title="Seller Distribution in PCA Latent Space",
                height=390,
                legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5, font=dict(size=10)),
                margin=dict(l=20, r=20, t=48, b=76),
            )
            st.plotly_chart(fig_pca, width="stretch")

    col_s3, col_s4 = st.columns(2, gap="medium")
    with col_s3:
        with st.container(border=True):
            radar_metrics = ["avg_price", "avg_handling_days", "avg_transit_days", "late_rate", "avg_review_score", "avg_distance_km"]
            radar_labels = ["Avg Price", "Handling Days", "Transit Days", "Late Rate", "Avg Review", "Avg Distance"]
            radar_norm = sellers.groupby("seller_archetype")[radar_metrics].median().reset_index()
            for col in radar_metrics:
                cmin, cmax = radar_norm[col].min(), radar_norm[col].max()
                radar_norm[col] = (radar_norm[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.5

            fig_radar = go.Figure()
            for _, row in radar_norm.iterrows():
                vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
                labs = radar_labels + [radar_labels[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=labs, fill="toself",
                    name=row["seller_archetype"],
                    line=dict(color=arch_colors.get(row["seller_archetype"], INDIGO), width=2),
                    opacity=0.65,
                ))
            style_fig(
                fig_radar,
                title="Normalized Median Archetype Profiles",
                height=420,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(148,163,184,0.12)", tickfont=dict(size=9)),
                    angularaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
                ),
                legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5, font=dict(size=10)),
                margin=dict(l=40, r=40, t=50, b=85),
            )
            st.plotly_chart(fig_radar, width="stretch")

    with col_s4:
        with st.container(border=True):
            fig_rel = px.box(
                sellers.dropna(subset=["reliability_score"]),
                x="seller_archetype", y="reliability_score",
                color="seller_archetype",
                color_discrete_map=arch_colors,
            )
            style_fig(fig_rel, title="Reliability Score Distribution (0–100) by Tier", height=420, showlegend=False, xaxis_title="", yaxis_title="Reliability Score")
            fig_rel.update_xaxes(tickangle=-12)
            st.plotly_chart(fig_rel, width="stretch")

    with st.container(border=True):
        st.markdown("**Seller Tier Summary & Scorecard Lookup**")
        tier_summary = (
            sellers.groupby("seller_archetype")
            .agg(
                Sellers=("seller_id", "count"),
                Avg_Orders=("total_orders", "mean"),
                Avg_Revenue=("total_revenue", "mean"),
                Late_Rate=("late_rate", "mean"),
                Avg_Review=("avg_review_score", "mean"),
                Reliability=("reliability_score", "mean"),
            )
            .reset_index()
            .rename(columns={"seller_archetype": "Archetype"})
        )
        tier_summary["Late %"] = (tier_summary["Late_Rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            tier_summary[["Archetype", "Sellers", "Avg_Orders", "Avg_Revenue", "Late %", "Avg_Review", "Reliability"]].round(2),
            width="stretch",
            hide_index=True,
        )

    insight_box(
        "Key Finding — Actionable 4-Tier Seller Operations",
        "<b>Tier 1 (237 Star High-Volume Sellers)</b> average ~168 orders each with an 88.0 reliability score. "
        "Meanwhile, <b>Tier 4 (754 At-Risk / Slow Sellers)</b> drop to a 63.0 average reliability score—identifying a concrete operational cohort for tighter dispatch SLAs or coaching.",
        accent=TEAL,
    )

# ══════════════════════════════════════════════════════════════
# TAB 7 — RISK PATTERN MINING
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    section_banner(
        "Lecture 18 • FP-Growth & Apriori Association Rules",
        "Which Order Profiles Carry the Highest Risk of 1–2★ Reviews?",
        "Frequent-pattern rules mined across order attributes (baseline bad-review rate: 12.8%) and multi-category baskets (Notebook 04).",
    )

    col_rk1, col_rk2 = st.columns([3, 2], gap="medium")
    with col_rk1:
        with st.container(border=True):
            min_conf = st.slider("Minimum Confidence Threshold", 0.50, 0.90, 0.70, 0.05)
            risk_filt = rules_risk[rules_risk["confidence"] >= min_conf].sort_values("lift", ascending=False).head(15).copy()

            if len(risk_filt):
                risk_filt["conf_pct"] = risk_filt["confidence"] * 100
                risk_filt["short_label"] = risk_filt["antecedent"].str.replace(" + ", " · ", regex=False).str[:58]
                fig_risk = go.Figure(go.Bar(
                    y=risk_filt["short_label"],
                    x=risk_filt["lift"],
                    orientation="h",
                    marker=dict(
                        color=risk_filt["conf_pct"],
                        colorscale=[[0, AMBER], [0.5, ORANGE], [1, RED]],
                        colorbar=dict(title="Conf %"),
                        cornerradius=5,
                    ),
                    text=[f"{l:.2f}x ({c:.0f}%)" for l, c in zip(risk_filt["lift"], risk_filt["conf_pct"])],
                    textposition="outside",
                    cliponaxis=False,
                    customdata=np.stack([risk_filt["antecedent"], risk_filt["n_orders"], risk_filt["conf_pct"]], axis=-1),
                    hovertemplate="<b>%{customdata[0]}</b><br>Lift: %{x:.2f}x<br>Confidence: %{customdata[2]:.1f}%<br>Orders: %{customdata[1]:,}<extra></extra>",
                ))
                style_fig(
                    fig_risk,
                    title="Top 15 Bad-Review Risk Rules by Lift",
                    height=500,
                    xaxis_title="Lift (vs 12.8% Baseline)",
                    yaxis=dict(autorange="reversed", showgrid=False),
                    margin=dict(l=20, r=85, t=50, b=36),
                )
                st.plotly_chart(fig_risk, width="stretch")
            else:
                st.info("No rules meet the selected confidence threshold.")

    with col_rk2:
        with st.container(border=True):
            if len(rules_copurchase):
                cop = rules_copurchase.sort_values("lift", ascending=False).copy()
                cop["rule"] = cop["antecedent"] + " → " + cop["consequent"]
                fig_cop = go.Figure(go.Bar(
                    y=cop["rule"],
                    x=cop["lift"],
                    orientation="h",
                    marker=dict(color=INDIGO, cornerradius=5),
                    text=[f"{l:.2f}x (n={n})" for l, n in zip(cop["lift"], cop["n_baskets"])],
                    textposition="outside",
                    cliponaxis=False,
                    customdata=cop["confidence"] * 100,
                    hovertemplate="<b>%{y}</b><br>Lift: %{x:.2f}x<br>Confidence: %{customdata:.1f}%<extra></extra>",
                ))
                style_fig(
                    fig_cop,
                    title="Category Co-Purchase Rules (Multi-Category Orders)",
                    height=500,
                    xaxis_title="Lift",
                    yaxis=dict(autorange="reversed", showgrid=False),
                    margin=dict(l=20, r=75, t=50, b=36),
                )
                st.plotly_chart(fig_cop, width="stretch")

    with st.container(border=True):
        if len(rules_risk):
            rl_df = rules_risk.copy()
            rl_df["Confidence (%)"] = rl_df["confidence"] * 100
            fig_rl = px.scatter(
                rl_df,
                x="Confidence (%)",
                y="lift",
                size="n_orders",
                color="conviction",
                color_continuous_scale=[[0, CYAN], [0.5, AMBER], [1, RED]],
                hover_data=["antecedent", "n_orders", "support"],
                labels={"lift": "Lift", "n_orders": "Orders", "conviction": "Conviction"},
                opacity=0.72,
            )
            style_fig(fig_rl, title="Rule Quality Landscape: Confidence × Lift × Order Volume", height=400)
            st.plotly_chart(fig_rl, width="stretch")

            with st.expander("📋 Inspect Full Risk Rule Definitions"):
                disp_rules = rl_df[["antecedent", "n_orders", "support", "Confidence (%)", "lift", "conviction"]].sort_values("lift", ascending=False).head(50)
                st.dataframe(
                    disp_rules.style.format({"support": "{:.3f}", "Confidence (%)": "{:.1f}%", "lift": "{:.2f}x", "conviction": "{:.2f}"}),
                    width="stretch",
                    hide_index=True,
                )

    insight_box(
        "Key Finding — Combined Route & Delay Risk Multiplies Dissatisfaction 6.3×",
        "Cross-state orders delayed <b>&gt;7 days</b>—especially combined with slow seller handling or high freight—receive 1–2★ reviews roughly <b>80.6% of the time</b> "
        "(6.29× lift above the 12.8% baseline). Delivery reliability outweighs product category or price as a driver of bad reviews.",
        accent=AMBER,
    )

# ══════════════════════════════════════════════════════════════
# TAB 8 — DEEP DIVE EXPLORER
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    section_banner(
        "Interactive Sandbox • Custom Feature Explorer",
        "Explore Relationships, Correlations & Category Metrics",
        "Pick any two order variables, slice by category/status, and inspect category benchmarks dynamically.",
    )

    with st.container(border=True):
        numeric_cols = [
            "total_price", "total_freight", "freight_ratio", "total_weight_g",
            "distance_km", "approval_hours", "seller_handling_days",
            "carrier_transit_days", "total_delivery_days", "days_late",
            "review_score", "item_count", "max_installments",
        ]
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1:
            x_var = st.selectbox("X-axis", numeric_cols, index=4)
        with col_e2:
            y_var = st.selectbox("Y-axis", numeric_cols, index=8)
        with col_e3:
            color_var = st.selectbox("Color by", ["review_score", "is_late", "lateness_bucket", "customer_state", "primary_category", "is_cross_state"], index=0)
        with col_e4:
            sample_size = st.slider("Sample size", 500, 10000, 3000, 500)

        clean_exp = filt.dropna(subset=[x_var, y_var])
        if len(clean_exp):
            explore_data = clean_exp.sample(n=min(sample_size, len(clean_exp)), random_state=42)
            if color_var == "review_score":
                scatter_kwargs = dict(
                    x=x_var, y=y_var, color=color_var,
                    color_continuous_scale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                    opacity=0.55,
                )
                try:
                    import statsmodels.api  # noqa: F401
                    scatter_kwargs["trendline"] = "ols"
                except ImportError:
                    pass
                fig_explore = px.scatter(explore_data, **scatter_kwargs)
            else:
                fig_explore = px.scatter(
                    explore_data, x=x_var, y=y_var, color=color_var,
                    color_discrete_sequence=COLORS,
                    opacity=0.55,
                )
            style_fig(
                fig_explore,
                title=f"{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}",
                height=440,
            )
            st.plotly_chart(fig_explore, width="stretch")
        else:
            st.info("No non-null rows available for the selected axis pair.")

    col_ex1, col_ex2 = st.columns([3, 2], gap="medium")
    with col_ex1:
        with st.container(border=True):
            st.markdown("**Category Performance Summary**")
            cat_summary = (
                filt.groupby("primary_category")
                .agg(
                    orders=("order_id", "count"),
                    avg_price=("total_price", "mean"),
                    avg_freight=("total_freight", "mean"),
                    avg_delivery_days=("total_delivery_days", "mean"),
                    avg_days_late=("days_late", "mean"),
                    late_pct=("is_late", "mean"),
                    avg_review=("review_score", "mean"),
                    avg_distance=("distance_km", "mean"),
                )
                .reset_index()
                .sort_values("orders", ascending=False)
            )
            cat_summary["late_pct"] = (cat_summary["late_pct"] * 100).round(1)
            cat_summary = cat_summary.round(2)

            st.dataframe(
                cat_summary.style.background_gradient(subset=["late_pct"], cmap="RdYlGn_r")
                .background_gradient(subset=["avg_review"], cmap="RdYlGn")
                .format({"avg_price": "R$ {:.2f}", "avg_freight": "R$ {:.2f}", "avg_distance": "{:.0f} km", "late_pct": "{:.1f}%"}),
                width="stretch",
                height=440,
                hide_index=True,
            )

    with col_ex2:
        with st.container(border=True):
            corr_cols = [
                "total_price", "total_freight", "distance_km", "seller_handling_days",
                "carrier_transit_days", "total_delivery_days", "days_late", "review_score",
            ]
            corr_data = filt[corr_cols].dropna()
            if len(corr_data) > 30:
                corr_matrix = corr_data.corr()
                labels_clean = [c.replace("_", " ").replace("total ", "").title() for c in corr_matrix.columns]
                fig_corr = px.imshow(
                    corr_matrix.values,
                    x=labels_clean,
                    y=labels_clean,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    text_auto=".2f",
                    aspect="auto",
                )
                style_fig(fig_corr, title="Pearson Correlation Matrix", height=440, margin=dict(l=20, r=20, t=50, b=60))
                fig_corr.update_xaxes(tickangle=-35)
                st.plotly_chart(fig_corr, width="stretch")

    insight_box(
        "Key Finding — Delivery Lateness Outweighs Price in Customer Satisfaction",
        "Across all numeric variables, <b>days_late</b> exhibits the strongest negative correlation with <b>review_score</b> (~−0.34). "
        "Meanwhile, order price has virtually zero correlation with rating—proving Brazilian marketplace buyers care far more about promised-date reliability than item cost.",
        accent=CYAN,
    )

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:0.78rem; padding: 6px 0;'>"
    "📦 <b>DELTA</b> — Delivery Estimate, Lateness & Trust Analytics &nbsp;•&nbsp; DS3030 Data Analytics &nbsp;•&nbsp; "
    "Built with Streamlit & Plotly &nbsp;•&nbsp; Olist Brazilian E-Commerce (Kaggle)"
    "</div>",
    unsafe_allow_html=True,
)
