"""
DELTA - Delivery Estimate, Lateness & Trust Analytics
Streamlit dashboard for the DS3030 Data Analytics project (Olist Brazilian e-commerce).

Run from the project folder (the one that contains PROCESSED_DATA/):
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# Page setup and a deliberately plain look
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DELTA - Olist delivery analytics", layout="wide")

INK = "#1f2933"
MUTED = "#6b7580"
GRID = "#e4e7ea"
BLUE = "#2f6f9f"
ORANGE = "#d9822b"
RED = "#b8432f"
GREEN = "#4a8f5d"
GREY = "#9aa3ab"

BUCKET_ORDER = [
    "Early (>1d early)",
    "On Time (-1d to 0d)",
    "1-3 Days Late",
    "4-7 Days Late",
    ">7 Days Late",
]
BUCKET_COLORS = dict(zip(BUCKET_ORDER, [GREEN, "#8fbf9f", "#e3b23c", ORANGE, RED]))
BUCKET_SHORT = {
    "Early (>1d early)": "Early",
    "On Time (-1d to 0d)": "On time",
    "1-3 Days Late": "1-3 d late",
    "4-7 Days Late": "4-7 d late",
    ">7 Days Late": ">7 d late",
}

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }}
    h1 {{ font-size: 1.9rem !important; font-weight: 650 !important; }}
    h2 {{ font-size: 1.3rem !important; font-weight: 650 !important; margin-top: 1.6rem !important; }}
    h3 {{ font-size: 1.05rem !important; font-weight: 650 !important; }}
    [data-testid="stMetric"] {{
        background: #ffffff; border: 1px solid {GRID}; border-radius: 6px; padding: 10px 14px;
    }}
    [data-testid="stMetricLabel"] p {{ font-size: 0.82rem; color: {MUTED}; }}
    [data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
    .callout {{
        border-left: 3px solid {BLUE}; background: #eef3f7; padding: 0.75rem 1rem;
        border-radius: 3px; margin: 0.4rem 0 1.2rem 0; color: {INK}; line-height: 1.5;
    }}
    .callout b.t {{ display: block; margin-bottom: 2px; }}
    .chart-title {{ font-weight: 650; margin: 0.6rem 0 0 0; color: {INK}; }}
    .chart-sub {{ color: {MUTED}; font-size: 0.86rem; margin: 0 0 0.2rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def show(fig, **kw):
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def style(fig, height=380, legend=False, **layout):
    fig.update_layout(
        template="simple_white",
        height=height,
        margin=dict(l=10, r=20, t=20, b=10),
        font=dict(family="Source Sans Pro, Arial, sans-serif", size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, automargin=True, linecolor=GREY, ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_layout(**layout)
    return fig


def header(title, sub=""):
    st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='chart-sub'>{sub}</div>", unsafe_allow_html=True)


def callout(title, body):
    st.markdown(f"<div class='callout'><b class='t'>{title}</b>{body}</div>", unsafe_allow_html=True)


def page_intro(title, lab, text):
    st.title(title)
    st.caption(lab)
    st.write(text)


def fmt_p(p):
    if p is None or np.isnan(p):
        return "n/a"
    return "< 1e-300" if p == 0 else f"{p:.2e}"


def fmt_int(x):
    return f"{int(round(x)):,}"


def pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{x * 100:.{d}f}%"


def short_tier(name):
    return {
        "1": "Tier 1 - Star",
        "2": "Tier 2 - Premium",
        "3": "Tier 3 - Standard",
        "4": "Tier 4 - At-risk",
    }.get(str(name).split(":")[0].replace("Tier", "").strip(), str(name))


TIER_COLORS = {
    "Tier 1 - Star": GREEN,
    "Tier 2 - Premium": BLUE,
    "Tier 3 - Standard": GREY,
    "Tier 4 - At-risk": RED,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA = Path(__file__).resolve().parent / "PROCESSED_DATA"
if not DATA.exists():
    DATA = Path("PROCESSED_DATA")

ORDER_COLS = [
    "order_id", "order_status", "order_purchase_timestamp", "purchase_date", "purchase_year",
    "purchase_month", "purchase_dayofweek", "purchase_hour", "customer_state", "seller_state",
    "primary_category", "total_price", "total_freight", "total_weight_g", "freight_ratio",
    "item_count", "max_installments", "primary_payment_type", "review_score", "has_comment", "total_volume_cm3",
    "approval_hours", "seller_handling_days", "carrier_transit_days", "total_delivery_days",
    "estimated_delivery_days", "days_late", "seller_shipped_late", "is_late", "lateness_bucket",
    "distance_km", "is_cross_state",
]


@st.cache_data(show_spinner="Loading orders ...")
def load_orders():
    full = pd.read_csv(DATA / "master_orders.csv", low_memory=False)
    missing = full.isna().mean().sort_values(ascending=False)
    keep = [c for c in ORDER_COLS if c in full.columns]
    df = full[keep].copy()
    for c in ("purchase_date", "order_purchase_timestamp"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df, missing


@st.cache_data(show_spinner=False)
def load_csv(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else None


if not (DATA / "master_orders.csv").exists():
    st.error(
        "Could not find PROCESSED_DATA/master_orders.csv. Run the notebooks first, then start the app "
        "from the project folder that contains PROCESSED_DATA."
    )
    st.stop()

orders, missing_share = load_orders()
sellers = load_csv("seller_scorecard.csv")
daily = load_csv("daily_orders_smoothed.csv")
routes = load_csv("graph_route_performance.csv")
state_metrics = load_csv("graph_state_metrics.csv")
nlp_top = load_csv("nlp_top_terms.csv")
nlp_dist = load_csv("nlp_distinctive_terms.csv")
if daily is not None:
    daily["purchase_date"] = pd.to_datetime(daily["purchase_date"])
if sellers is not None:
    sellers["tier"] = sellers["seller_archetype"].map(short_tier)

# ---------------------------------------------------------------------------
# Sidebar: navigation + filters
# ---------------------------------------------------------------------------
PAGES = [
    "Overview",
    "1. Data processing",
    "2. Delivery insights",
    "3. Statistics & transformations",
    "4. Outliers & missing values",
    "5. Reviews (text mining)",
    "6. Geography & network",
    "7. Sellers: reduction & clustering",
    "8. Clustering algorithms",
    "Conclusions",
    "Challenges & decisions",
]
with st.sidebar:
    st.markdown("### DELTA")
    st.caption("Delivery Estimate, Lateness & Trust Analytics")
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Filters** (apply to order-level charts)")
    years = sorted(orders["purchase_year"].dropna().astype(int).unique().tolist())
    sel_years = st.multiselect("Purchase year", years, default=years)
    sel_states = st.multiselect("Customer state", sorted(orders["customer_state"].dropna().unique()))
    sel_cats = st.multiselect(
        "Product category", orders["primary_category"].value_counts().index.tolist()
    )

mask = orders["purchase_year"].isin(sel_years or years)
if sel_states:
    mask &= orders["customer_state"].isin(sel_states)
if sel_cats:
    mask &= orders["primary_category"].isin(sel_cats)
filt = orders[mask]
with st.sidebar:
    st.caption(f"{len(filt):,} of {len(orders):,} orders selected")

if filt.empty:
    st.warning("No orders match the filters. Widen the filters in the sidebar.")
    st.stop()

present_buckets = [b for b in BUCKET_ORDER if b in set(filt["lateness_bucket"].dropna())]


# ---------------------------------------------------------------------------
# Shared calculations
# ---------------------------------------------------------------------------
def skew(x):
    x = pd.Series(x).dropna()
    return float(stats.skew(x, bias=False)) if len(x) > 3 else np.nan


def low_review_rate(df):
    d = df["review_score"].dropna()
    return (d <= 2).mean() if len(d) else np.nan


# ===========================================================================
# OVERVIEW
# ===========================================================================
def page_overview():
    st.title("Why do orders arrive late, and what does it cost in ratings?")
    st.write(
        "An analysis of the Olist Brazilian marketplace (orders from 2016-2018). Starting from nine raw tables, "
        "we built clean order and seller tables, studied how delivery delays arise, and measured how much they "
        "hurt customer reviews. Use the sections on the left to follow the analysis in the order we did it."
    )

    delivered = filt.dropna(subset=["is_late"])
    c = st.columns(6)
    c[0].metric("Orders", fmt_int(len(filt)))
    c[1].metric("Revenue (R$)", f"{filt['total_price'].sum() / 1e6:.2f} M")
    c[2].metric("Late deliveries", pct(delivered["is_late"].mean()))
    c[3].metric("Average review", f"{filt['review_score'].mean():.2f} / 5")
    c[4].metric("Median delivery", f"{filt['total_delivery_days'].median():.1f} days")
    c[5].metric("Median distance", f"{filt['distance_km'].median():.0f} km")

    late = delivered[delivered["is_late"] == 1]
    ontime = delivered[delivered["is_late"] == 0]
    bad_4plus = filt[filt["lateness_bucket"].isin(["4-7 Days Late", ">7 Days Late"])]
    cross = delivered.groupby("is_cross_state")["is_late"].mean()
    shipped_late = late["seller_shipped_late"].mean() if "seller_shipped_late" in late else np.nan

    st.markdown("## What the data says")
    a, b = st.columns(2)
    with a:
        callout(
            "Lateness is the main driver of bad reviews",
            f"On-time orders average <b>{ontime['review_score'].mean():.2f}</b> stars, late orders "
            f"<b>{late['review_score'].mean():.2f}</b>. Once an order is 4+ days late, "
            f"<b>{pct(low_review_rate(bad_4plus), 0)}</b> of reviews are 1-2 stars "
            f"(baseline: {pct(low_review_rate(filt), 0)}).",
        )
        callout(
            "Most late orders are not the seller's fault at dispatch",
            f"Only <b>{pct(shipped_late, 0)}</b> of late orders were handed to the carrier after the seller's "
            "shipping deadline. The rest were late despite an on-time dispatch, so transit time and the promised "
            "date matter more than packing speed.",
        )
    with b:
        callout(
            "The promise was not shorter for orders that ended up late",
            f"Orders that arrived late had an estimated delivery window of "
            f"<b>{late['estimated_delivery_days'].mean():.1f}</b> days versus "
            f"<b>{ontime['estimated_delivery_days'].mean():.1f}</b> for on-time ones, but actually took "
            f"<b>{late['total_delivery_days'].mean():.1f}</b> vs <b>{ontime['total_delivery_days'].mean():.1f}</b> days. "
            "The estimate does not adapt to the risk of a route.",
        )
        if len(cross) == 2:
            callout(
                "Crossing a state border raises the risk",
                f"Late rate is <b>{pct(cross.get(1))}</b> for cross-state orders and "
                f"<b>{pct(cross.get(0))}</b> within a state.",
            )

    left, right = st.columns(2)
    with left:
        header("Average review by delivery outcome", "The rating drops sharply after the promised date.")
        g = filt.dropna(subset=["lateness_bucket", "review_score"]).groupby("lateness_bucket")["review_score"].mean()
        g = g.reindex(present_buckets)
        fig = go.Figure(
            go.Bar(
                x=[BUCKET_SHORT[b] for b in g.index], y=g.values,
                marker_color=[BUCKET_COLORS[b] for b in g.index],
                text=[f"{v:.2f}" for v in g.values], textposition="outside", cliponaxis=False,
            )
        )
        style(fig, 340, yaxis=dict(range=[0, 5.3], title="Average stars"))
        show(fig)
    with right:
        header("Orders and late rate by month", "Volume grew fast through 2017; late rate spiked around peak season.")
        m = (
            filt.dropna(subset=["purchase_date"])
            .assign(month=lambda d: d["purchase_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month")
            .agg(orders=("order_id", "count"), late=("is_late", "mean"))
            .reset_index()
        )
        m = m[m["orders"] >= 50]
        fig = go.Figure()
        fig.add_bar(x=m["month"], y=m["orders"], name="Orders", marker_color="#c9d6e2")
        fig.add_scatter(x=m["month"], y=m["late"] * 100, name="Late rate (%)", yaxis="y2",
                        line=dict(color=RED, width=2.5), mode="lines+markers")
        style(fig, 340, legend=True,
              yaxis=dict(title="Orders per month"),
              yaxis2=dict(title="Late rate (%)", overlaying="y", side="right", showgrid=False, rangemode="tozero"))
        show(fig)

    st.markdown("## Where each technique is used")
    table(pd.DataFrame([
        ["Data processing", "Joins without row multiplication, geo cleaning, derived delivery stages, smoothing", "1, 2"],
        ["Statistics of data", "Descriptives, skewness, hypothesis tests, effect sizes, correlation", "3"],
        ["Transformations", "Log, square root, Box-Cox, Yeo-Johnson, scaling, encodings", "3"],
        ["Outliers and missing values", "IQR vs Z-score, scatter plots, winsorizing, MNAR comments, comment model, imputation experiment", "4"],
        ["Text mining", "Portuguese TF-IDF that keeps negations", "5"],
        ["Graph analysis", "State shipping network, PageRank, betweenness, route risk", "6"],
        ["Dimensionality reduction", "PCA (and its SVD equivalence), t-SNE at five perplexities, autoencoder", "7"],
        ["Clustering", "K-Means, Ward, DBSCAN, Mean-Shift, OPTICS, seller tiers", "7, 8"],
    ], columns=["Topic", "What we did", "Section"]))


# ===========================================================================
# 1. DATA PROCESSING
# ===========================================================================
def page_processing():
    page_intro(
        "1. Data processing",
        "Notebook 01 - cleaning, master tables, time-series smoothing",
        "Nine raw CSV files were cleaned, joined into one row per order and one row per seller, and enriched "
        "with delivery stage durations and customer-to-seller distance.",
    )

    st.markdown("## The raw tables")
    table(pd.DataFrame({
        "Table": ["orders", "order_items", "order_payments", "order_reviews", "customers", "sellers", "products", "geolocation", "category_translation"],
        "Rows": [99441, 112650, 103886, 99224, 99441, 3095, 32951, 1000163, 71],
        "Columns": [8, 7, 5, 7, 5, 4, 9, 5, 2],
        "Exact duplicate rows": [0, 0, 0, 0, 0, 0, 0, 261831, 0],
        "Missing cells": [4908, 0, 0, 145903, 0, 0, 2448, 0, 0],
    }).style.format({"Rows": "{:,}", "Exact duplicate rows": "{:,}", "Missing cells": "{:,}"}))
    st.caption("Numbers are from the notebook's profiling step (full raw data, not affected by the sidebar filters).")

    st.markdown("## From raw tables to the master tables")
    header("Row counts before and after cleaning", "Aggregating child tables first keeps one row per order.")
    steps = pd.DataFrame({
        "Step": ["Order items", "Order reviews", "Geolocation"],
        "Raw rows": [112650, 99224, 1000163],
        "After cleaning": [98666, 98673, 19010],
        "How": ["Aggregated to 1 row per order", "De-duplicated per order", "Median lat/lng per zip prefix"],
    })
    fig = go.Figure()
    fig.add_bar(y=steps["Step"], x=steps["Raw rows"], orientation="h", name="Raw", marker_color="#c9d6e2")
    fig.add_bar(y=steps["Step"], x=steps["After cleaning"], orientation="h", name="After cleaning", marker_color=BLUE)
    style(fig, 250, legend=True, barmode="group", xaxis=dict(type="log", title="Rows (log scale)"),
          yaxis=dict(autorange="reversed"))
    show(fig)

    st.markdown("## Time-series noise removal")
    st.write(
        "Daily order counts swing between weekdays and weekends. We compared a simple moving average, an "
        "exponentially weighted average and a rolling median to expose the underlying trend."
    )
    if daily is None:
        st.info("daily_orders_smoothed.csv was not found.")
    else:
        labels = {"raw_orders": "Daily orders", "raw_revenue": "Daily revenue (R$)",
                  "late_rate": "Daily late rate (%)", "avg_review": "Daily average review"}
        avail = {k: v for k, v in labels.items() if k in daily.columns}
        c1, c2 = st.columns([2, 3])
        metric = c1.selectbox("Metric", list(avail), format_func=lambda k: avail[k])
        win = c2.radio("Window (days)", [7, 14, 30], index=1, horizontal=True)
        d = daily[["purchase_date", metric]].copy()
        d["raw"] = d[metric] * (100 if metric == "late_rate" else 1)
        d["SMA"] = d["raw"].rolling(win, min_periods=1).mean()
        d["EWMA"] = d["raw"].ewm(span=win, adjust=False).mean()
        d["Rolling median"] = d["raw"].rolling(win, min_periods=1).median()
        fig = go.Figure()
        fig.add_scatter(x=d["purchase_date"], y=d["raw"], name="Raw daily", line=dict(color="#c4cad0", width=1))
        for name, col in [("SMA", BLUE), ("EWMA", GREEN), ("Rolling median", ORANGE)]:
            fig.add_scatter(x=d["purchase_date"], y=d[name], name=f"{name} ({win}d)", line=dict(color=col, width=2.4))
        bf = pd.Timestamp("2017-11-24")
        fig.add_shape(type="line", x0=bf, x1=bf, y0=0, y1=1, yref="paper", line=dict(color=GREY, dash="dot"))
        fig.add_annotation(x=bf, y=1.0, yref="paper", text="Black Friday 2017", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=MUTED))
        style(fig, 380, legend=True, hovermode="x unified", yaxis=dict(title=avail[metric]))
        show(fig)

        bfr = daily[daily["purchase_date"] == bf]
        if len(bfr) and "raw_orders" in daily:
            s = daily["raw_orders"]
            raw_v = float(bfr["raw_orders"].iloc[0])
            idx = bfr.index[0]
            rows = [["Raw", raw_v]]
            for w in (7, 14, 30):
                rows.append([f"{w}-day moving average", float(s.rolling(w, min_periods=1).mean().iloc[idx])])
            rows.append(["14-day rolling median", float(s.rolling(14, min_periods=1).median().iloc[idx])])
            tbl = pd.DataFrame(rows, columns=["Filter", "Orders on 24 Nov 2017"])
            tbl["Orders on 24 Nov 2017"] = tbl["Orders on 24 Nov 2017"].round(0).astype(int)
            c1, c2 = st.columns([2, 3])
            with c1:
                table(tbl)
            with c2:
                callout(
                    "Wider windows flatten real events",
                    "Black Friday is a genuine spike, not noise. A 30-day average hides most of it, while a 7-day "
                    "window removes the weekday ripple and still shows the peak. We used 7-14 days for trend and "
                    "kept the raw series for event analysis.",
                )


# ===========================================================================
# 2. DELIVERY INSIGHTS
# ===========================================================================
def page_delivery():
    page_intro(
        "2. Delivery insights",
        "Research questions answered with the cleaned master table",
        "Where is time lost between purchase and delivery, who is responsible, and would a more honest delivery "
        "promise help?",
    )
    d = filt.dropna(subset=["is_late"])
    late, ontime = d[d["is_late"] == 1], d[d["is_late"] == 0]

    st.markdown("## Where is time lost?")
    stage_cols = ["approval_hours", "seller_handling_days", "carrier_transit_days"]
    s = d.dropna(subset=stage_cols)
    if len(s):
        def med(df):
            return [df["approval_hours"].median() / 24, df["seller_handling_days"].median(), df["carrier_transit_days"].median()]
        names = ["Payment approval", "Seller handling", "Carrier transit"]
        mo, ml = med(s[s["is_late"] == 0]), med(s[s["is_late"] == 1])
        c1, c2 = st.columns([3, 2])
        with c1:
            header("Median days per stage: on-time vs late orders")
            fig = go.Figure()
            fig.add_bar(x=names, y=mo, name="On time", marker_color=GREEN, text=[f"{v:.1f}" for v in mo], textposition="outside")
            fig.add_bar(x=names, y=ml, name="Late", marker_color=RED, text=[f"{v:.1f}" for v in ml], textposition="outside")
            style(fig, 340, legend=True, barmode="group", yaxis=dict(title="Median days"))
            show(fig)
        with c2:
            callout(
                "Transit is where late orders lose their time",
                f"Late orders spend a median of <b>{ml[2]:.1f}</b> days with the carrier against "
                f"<b>{mo[2]:.1f}</b> for on-time orders. Approval takes under a day in both groups and seller "
                f"handling changes by about <b>{ml[1] - mo[1]:.1f}</b> days, so the gap is almost entirely transit.",
            )

    st.markdown("## Who is responsible: seller or carrier?")
    if "seller_shipped_late" in late and late["seller_shipped_late"].notna().any():
        sl = late["seller_shipped_late"].mean()
        c1, c2 = st.columns([2, 3])
        with c1:
            fig = go.Figure(go.Pie(
                labels=["Seller shipped after deadline", "Seller shipped on time (carrier / promise)"],
                values=[sl, 1 - sl], hole=0.55, sort=False,
                marker=dict(colors=[ORANGE, BLUE], line=dict(color="white", width=2)),
                textinfo="percent", textfont=dict(size=14),
            ))
            style(fig, 300, legend=True, legend_orientation="h", legend_y=-0.15, legend_yanchor="top")
            show(fig)
        with c2:
            by_seller = d.groupby("seller_shipped_late")["is_late"].mean()
            callout(
                "Attributing lateness",
                f"Of all late orders, <b>{pct(sl, 0)}</b> involved a seller who missed the dispatch deadline. "
                f"But sellers who shipped late still delivered late only <b>{pct(by_seller.get(1.0), 0)}</b> of the time, "
                f"against <b>{pct(by_seller.get(0.0), 0)}</b> when they shipped on time. Dispatch speed matters, but it "
                "is not the main lever.",
            )

    st.markdown("## The delivery promise vs reality")
    c1, c2 = st.columns([3, 2])
    with c1:
        header("Estimated vs actual delivery time (mean days)")
        fig = go.Figure()
        fig.add_bar(x=["On time", "Late"], y=[ontime["estimated_delivery_days"].mean(), late["estimated_delivery_days"].mean()],
                    name="Estimated by Olist", marker_color="#c9d6e2")
        fig.add_bar(x=["On time", "Late"], y=[ontime["total_delivery_days"].mean(), late["total_delivery_days"].mean()],
                    name="Actual", marker_color=BLUE)
        style(fig, 320, legend=True, barmode="group", yaxis=dict(title="Days"))
        show(fig)
    with c2:
        callout(
            "The estimate ignores route risk",
            "Late orders were promised about the same window as on-time ones, yet took roughly three times longer. "
            "A risk-aware estimate (longer for far or slow routes) would convert many late orders into on-time ones.",
        )

    st.markdown("## What if the promise were padded?")
    st.write(
        "Suppose every estimated delivery date were pushed back by a number of days. Orders that were late by less "
        "than that become on-time. We assume those orders would then be reviewed like other on-time orders."
    )
    pad = st.slider("Days added to the promised date", 0, 14, 3)
    dd = d.dropna(subset=["days_late"])
    base_bad = low_review_rate(dd[dd["days_late"] <= 0])

    def sim(p):
        resc = dd[(dd["days_late"] > 0) & (dd["days_late"] <= p)]
        rs = resc["review_score"].dropna()
        avoided = (rs <= 2).sum() - base_bad * len(rs)
        return (dd["days_late"] - p > 0).mean(), len(resc), avoided

    new_rate, n_resc, avoided = sim(pad)
    m = st.columns(4)
    m[0].metric("Late rate now", pct((dd["days_late"] > 0).mean()))
    m[1].metric(f"Late rate with +{pad} d", pct(new_rate))
    m[2].metric("Orders turned on-time", fmt_int(n_resc))
    m[3].metric("1-2 star reviews avoided (est.)", fmt_int(max(avoided, 0)))
    xs = list(range(0, 15))
    ys = [sim(p) for p in xs]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=[v[0] * 100 for v in ys], mode="lines+markers", name="Late rate (%)", line=dict(color=BLUE, width=2.5))
    fig.add_scatter(x=xs, y=[max(v[2], 0) for v in ys], mode="lines+markers", name="Bad reviews avoided", yaxis="y2",
                    line=dict(color=ORANGE, width=2.5))
    fig.add_vline(x=pad, line_dash="dot", line_color=GREY)
    style(fig, 320, legend=True, xaxis=dict(title="Days added to estimate"),
          yaxis=dict(title="Late rate (%)", rangemode="tozero"),
          yaxis2=dict(title="1-2 star reviews avoided", overlaying="y", side="right", showgrid=False, rangemode="tozero"))
    show(fig)
    st.caption(
        "This is a what-if on historical data, not a forecast. A longer promise can also lower conversion, "
        "which this simple model does not capture."
    )


# ===========================================================================
# 3. STATISTICS & TRANSFORMATIONS
# ===========================================================================
NUM_LABELS = {
    "total_price": "Order price (R$)", "total_freight": "Freight (R$)", "total_weight_g": "Weight (g)",
    "approval_hours": "Approval time (h)", "seller_handling_days": "Seller handling (d)",
    "carrier_transit_days": "Carrier transit (d)", "total_delivery_days": "Delivery time (d)",
    "days_late": "Days late", "distance_km": "Distance (km)", "review_score": "Review score",
}


TRANSFORM_METHODS = ["Log (log1p)", "Square root", "Box-Cox", "Yeo-Johnson"]
SCALING_FEATURES = [
    "total_orders", "total_revenue", "avg_handling_days", "avg_transit_days",
    "avg_delivery_days", "late_rate", "avg_review_score", "avg_distance_km",
]
CYCLICAL_SPECS = {
    "Hour of day (0-23)": ("purchase_hour", 24, 0, None),
    "Day of week (0-6)": ("purchase_dayofweek", 7, 0, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    "Month (1-12)": ("purchase_month", 12, 1, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
}


def _apply_transform(vals, method):
    """Apply a named skewness-reducing transform and return (transformed, fitted_lambda)."""
    arr = np.asarray(vals, dtype=float)
    if method == "Square root":
        return np.sqrt(arr), None
    if method == "Log (log1p)":
        return np.log1p(arr), None
    if method == "Box-Cox":
        out, lam = stats.boxcox(arr)
        return out, float(lam)
    out, lam = stats.yeojohnson(arr)
    return out, float(lam)


def _valid_transforms(arr):
    mn = float(np.min(arr))
    methods = []
    if mn >= 0:
        methods.extend(["Log (log1p)", "Square root"])
    if mn > 0:
        methods.append("Box-Cox")
    methods.append("Yeo-Johnson")
    return methods


@st.cache_data(show_spinner=False)
def transform_skews(x_values):
    x = pd.Series(x_values).dropna()
    if len(x) > 40000:
        x = x.sample(40000, random_state=0)
    vals = x.values.astype(float)
    out = {"Original": skew(vals)}
    for method in _valid_transforms(vals):
        tr, _ = _apply_transform(vals, method)
        out[method] = skew(tr)
    return out


@st.cache_data(show_spinner=False)
def seller_scaling_bundle(df):
    """Apply Standard, Robust, and Min-Max scalers on the 8 seller features from NB 02."""
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    cols = [c for c in SCALING_FEATURES if c in df.columns]
    raw = df[cols].astype(float).copy()
    raw = raw.fillna(raw.median())
    scalers = {
        "Original": None,
        "Standard (z-score)": StandardScaler(),
        "Robust (median / IQR)": RobustScaler(),
        "Min-Max (0-1)": MinMaxScaler(),
    }
    scaled = {}
    for name, sc in scalers.items():
        scaled[name] = raw.copy() if sc is None else pd.DataFrame(sc.fit_transform(raw), columns=cols, index=raw.index)
    return cols, scaled


@st.cache_data(show_spinner=False)
def oof_target_encode(cat_series, target_series, n_splits=5, smooth_m=20):
    """5-fold stratified out-of-fold smoothed target encoding (NB 02 s2.7)."""
    from sklearn.model_selection import StratifiedKFold
    valid = cat_series.notna() & target_series.notna()
    cats = cat_series[valid].astype(str).reset_index(drop=True)
    y = target_series[valid].astype(int).reset_index(drop=True)
    if len(y) < n_splits * 2 or y.value_counts().min() < n_splits:
        return None
    oof = np.empty(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(cats, y):
        tr_df = pd.DataFrame({"cat": cats.iloc[tr_idx], "y": y.iloc[tr_idx]})
        gmean = float(tr_df["y"].mean())
        agg = tr_df.groupby("cat")["y"].agg(["count", "mean"])
        enc_map = (agg["count"] * agg["mean"] + smooth_m * gmean) / (agg["count"] + smooth_m)
        oof[va_idx] = cats.iloc[va_idx].map(enc_map).fillna(gmean).values
    summary = (
        pd.DataFrame({"Category": cats, "y": y, "oof_enc": oof})
        .groupby("Category")
        .agg(Orders=("y", "count"), Raw_late_rate=("y", "mean"), OOF_encoded=("oof_enc", "mean"))
        .reset_index()
    )
    return summary


def run_tests(df):
    d = df.dropna(subset=["days_late", "review_score", "lateness_bucket"])
    late = d.loc[d["days_late"] > 0, "review_score"]
    ok = d.loc[d["days_late"] <= 0, "review_score"]
    if len(late) < 5 or len(ok) < 5:
        return None
    u = stats.mannwhitneyu(late, ok, alternative="two-sided")
    delta = 2 * u.statistic / (len(late) * len(ok)) - 1
    groups = [g["review_score"].values for _, g in d.groupby("lateness_bucket") if len(g) > 1]
    kw = stats.kruskal(*groups) if len(groups) > 1 else None
    ct = pd.crosstab(d["lateness_bucket"], d["review_score"] <= 2)
    chi = stats.chi2_contingency(ct) if ct.shape[0] > 1 and ct.shape[1] > 1 else None
    v = np.sqrt(chi[0] / (ct.values.sum() * (min(ct.shape) - 1))) if chi else np.nan
    return dict(u=u, delta=delta, kw=kw, chi=chi, v=v, n_late=len(late), n_ok=len(ok),
                med_late=late.median(), med_ok=ok.median())


def page_statistics():
    page_intro(
        "3. Statistics & transformations",
        "Notebook 02 - descriptive statistics, hypothesis tests, effect sizes, transformations",
        "Almost every business variable in this data is right-skewed, so the choice of statistic and test matters.",
    )
    cols = [c for c in NUM_LABELS if c in filt.columns]

    st.markdown("## Descriptive statistics")
    rows = []
    for c in cols:
        x = filt[c].dropna()
        rows.append([NUM_LABELS[c], len(x), x.mean(), x.median(), x.std(), skew(x)])
    desc = pd.DataFrame(rows, columns=["Variable", "Count", "Mean", "Median", "Std dev", "Skewness"])
    table(desc.style.format({"Count": "{:,.0f}", "Mean": "{:,.2f}", "Median": "{:,.2f}", "Std dev": "{:,.2f}",
                              "Skewness": "{:.2f}"}).background_gradient(subset=["Skewness"], cmap="Oranges"))
    callout(
        "Mean and median disagree for money and time",
        "Order price has a mean far above its median and a skewness near 10, because a few expensive orders pull the "
        "average up. Approval time is the extreme case: a median of about 20 minutes but a mean of ten hours. For "
        "these variables we report medians and use rank-based tests.",
    )

    st.markdown("## Relationships between variables")
    c1, c2 = st.columns([3, 2])
    method = c2.radio("Correlation type", ["Spearman", "Pearson"], horizontal=True,
                      help="Spearman uses ranks and is robust to the heavy skew seen above.")
    corr_cols = [c for c in ["total_price", "total_freight", "distance_km", "seller_handling_days", "carrier_transit_days",
                             "total_delivery_days", "days_late", "review_score"] if c in filt.columns]
    cm = filt[corr_cols].dropna().corr(method=method.lower())
    labs = [NUM_LABELS[c] for c in corr_cols]
    with c1:
        fig = go.Figure(go.Heatmap(z=cm.values, x=labs, y=labs, zmin=-1, zmax=1, colorscale="RdBu_r",
                                   text=np.round(cm.values, 2), texttemplate="%{text}", textfont=dict(size=11),
                                   colorbar=dict(thickness=12)))
        style(fig, 470, xaxis=dict(tickangle=-40), yaxis=dict(autorange="reversed", showgrid=False),
              margin=dict(l=10, r=10, t=10, b=10))
        show(fig)
    with c2:
        callout(
            "Distance explains time, not ratings",
            "Distance and delivery time are moderately related (Spearman about 0.54). Review score is tied to "
            "delivery time and lateness, but order price has almost no relationship with it: customers judge the "
            "promise, not the price.",
        )

    st.markdown("## Taming the skew: transformations")
    st.write(
        "Skewness near 0 means a symmetric distribution. Each transformation is applied to the selected data and "
        "the skewness is recomputed. Log and Box-Cox need positive values, so days late (which is negative for early "
        "deliveries) can only use Yeo-Johnson."
    )
    var = st.selectbox("Variable", cols[:-1] if "review_score" in cols else cols, format_func=lambda c: NUM_LABELS[c], index=0)
    raw_vals = filt[var].dropna().values
    sk = transform_skews(raw_vals)
    sk_df = pd.DataFrame({"Method": list(sk), "Skewness": list(sk.values())})
    c1, c2 = st.columns([2, 3])
    with c1:
        table(sk_df.style.format({"Skewness": "{:.3f}"}))
        st.caption("Methods that need positive values are left out when the variable has zeros or negatives.")
    with c2:
        fig = go.Figure(go.Bar(
            x=sk_df["Method"], y=sk_df["Skewness"],
            marker_color=[RED if m == "Original" else BLUE for m in sk_df["Method"]],
            text=[f"{v:.2f}" for v in sk_df["Skewness"]], textposition="outside", cliponaxis=False))
        style(fig, 300, yaxis=dict(title="Skewness"))
        show(fig)

    xs = filt[var].dropna().sample(min(int(filt[var].notna().sum()), 20000), random_state=0).values
    valid_m = _valid_transforms(xs)
    tr_choice = st.radio("Transformation to inspect", valid_m, horizontal=True, key="stat_tr_choice")
    tr_vals, tr_lam = _apply_transform(xs, tr_choice)
    c1, c2 = st.columns(2)
    with c1:
        header(f"{NUM_LABELS[var]} - original")
        f = go.Figure(go.Histogram(x=xs, nbinsx=60, marker_color=RED))
        style(f, 260, xaxis=dict(title=NUM_LABELS[var]), yaxis=dict(title="Orders"))
        show(f)
    with c2:
        lam_note = f" (lambda = {tr_lam:.3f})" if tr_lam is not None else ""
        header(f"After {tr_choice}{lam_note}")
        f = go.Figure(go.Histogram(x=tr_vals, nbinsx=60, marker_color=BLUE))
        style(f, 260, xaxis=dict(title=f"{tr_choice} of {NUM_LABELS[var]}"), yaxis=dict(title="Orders"))
        show(f)

    st.markdown("## Feature scaling: Standard vs Robust vs Min-Max")
    st.write(
        "From Notebook 02 (section 2.6.7), the eight continuous seller metrics span vastly different units (orders in 1-2,000 vs "
        "revenue in R\\$10-240k). Before clustering or neural models, features must share a common scale without letting outliers "
        "crush normal sellers into a narrow band."
    )
    if sellers is not None:
        sc_cols, sc_maps = seller_scaling_bundle(sellers)
        c1, c2 = st.columns([1, 2])
        sc_feat = c1.selectbox("Seller metric", sc_cols, format_func=lambda c: c.replace("_", " ").replace("avg ", "average ").capitalize())
        sc_choice = c2.radio("Scaler to summarize", list(sc_maps)[1:], horizontal=True, key="scaler_tbl_choice")
        s_df = sc_maps[sc_choice]
        sum_rows = []
        for col_s in sc_cols:
            v = s_df[col_s]
            sum_rows.append([
                col_s.replace("_", " ").replace("avg ", "average "),
                float(v.median()), float(v.quantile(0.75) - v.quantile(0.25)),
                float(v.mean()), float(v.std()), float(v.min()), float(v.max()),
            ])
        c1, c2 = st.columns([3, 2])
        with c1:
            header(f"Scaled seller distributions ({sc_feat.replace('_', ' ')})")
            f = go.Figure()
            for col_name, color in [("Standard (z-score)", BLUE), ("Robust (median / IQR)", GREEN), ("Min-Max (0-1)", ORANGE)]:
                f.add_box(y=sc_maps[col_name][sc_feat], name=col_name.split(" (")[0], marker_color=color, boxpoints="outliers")
            style(f, 300, yaxis=dict(title="Scaled value"), xaxis=dict(showgrid=False))
            show(f)
        with c2:
            callout(
                "Why Robust or Standard beats Min-Max",
                "Min-Max compresses 95% of sellers near 0 whenever a top seller hits R\\$240k revenue or 2,000 orders. "
                "RobustScaler centers at 0 with IQR = 1; in Notebook 05 we log-transform skewed counts first and then apply "
                "StandardScaler.",
            )
        with st.expander(f"See all 8 features under {sc_choice}"):
            table(pd.DataFrame(sum_rows, columns=["Feature", "Median", "IQR", "Mean", "Std dev", "Min", "Max"]).style.format(
                {"Median": "{:.2f}", "IQR": "{:.2f}", "Mean": "{:.2f}", "Std dev": "{:.2f}", "Min": "{:.2f}", "Max": "{:.2f}"}
            ))


# ===========================================================================
# 4. OUTLIERS & MISSING VALUES
# ===========================================================================
OUT_VARS = ["total_price", "total_freight", "total_weight_g", "total_delivery_days", "days_late", "distance_km", "approval_hours"]


def treatment_table(x, k):
    x = x.dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    p1, p99 = x.quantile([0.01, 0.99])
    rows = {"Original": x, "Winsorized (1-99%)": x.clip(p1, p99), "IQR-trimmed": x[(x >= lo) & (x <= hi)]}
    if x.min() >= 0:
        rows["log1p"] = np.log1p(x)
    return pd.DataFrame([[n, len(v), v.mean(), v.median(), v.std(), skew(v)] for n, v in rows.items()],
                        columns=["Treatment", "Count", "Mean", "Median", "Std dev", "Skewness"])


SCATTER_PAIRS = {
    "Orders": [("total_price", "total_freight"), ("total_weight_g", "total_volume_cm3"), ("total_delivery_days", "days_late")],
    "Items": [("price", "freight_value"), ("product_weight_g", "product_volume_cm3")],
    "Products": [("product_weight_g", "product_volume_cm3"), ("product_length_cm", "product_width_cm")],
}
SCATTER_LABELS = {
    "total_price": "Order price (R$)", "total_freight": "Freight (R$)", "total_weight_g": "Weight (g)",
    "total_volume_cm3": "Volume (cm3)", "total_delivery_days": "Delivery time (d)", "days_late": "Days late",
    "price": "Item price (R$)", "freight_value": "Item freight (R$)", "product_weight_g": "Product weight (g)",
    "product_volume_cm3": "Product volume (cm3)", "product_length_cm": "Product length (cm)",
    "product_width_cm": "Product width (cm)",
}


def scatter_frame(dataset):
    """Orders follow the sidebar filters; items and products are the cleaned tables from Notebook 03."""
    if dataset == "Orders":
        return filt
    return load_csv({"Items": "cleaned_items.csv", "Products": "cleaned_products.csv"}[dataset])


def iqr_flags(s, k):
    q1, q3 = s.quantile([0.25, 0.75])
    return (s < q1 - k * (q3 - q1)) | (s > q3 + k * (q3 - q1))


@st.cache_data(show_spinner=False)
def comment_model(df):
    """Notebook 03, section 4.6: does review score + days late predict that a comment exists?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = df.dropna(subset=["review_score", "days_late", "has_comment"])
    y = d["has_comment"].astype(int)
    if len(d) < 200 or y.nunique() < 2 or y.value_counts().min() < 20:
        return None
    X = d[["review_score", "days_late"]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    coef = model.named_steps["logisticregression"].coef_[0]
    return dict(
        n=len(d), n_test=len(yte),
        accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred, zero_division=0),
        recall=recall_score(yte, pred, zero_division=0), f1=f1_score(yte, pred, zero_division=0),
        cm=confusion_matrix(yte, pred), coef=dict(zip(["review_score", "days_late"], coef)),
        baseline=max(yte.mean(), 1 - yte.mean()),
    )


def page_outliers():
    page_intro(
        "4. Outliers & missing values",
        "Notebook 03 - outlier detection and handling, missing-value analysis, imputation experiment",
        "Extreme values are not automatically errors. We first detect them with two rules, then compare ways of "
        "handling them before deciding what to keep.",
    )
    st.markdown("## Outlier explorer")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    var = c1.selectbox("Variable", OUT_VARS, format_func=lambda c: NUM_LABELS[c])
    k = c2.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.25)
    z = c3.slider("Z-score cutoff", 2.0, 4.0, 3.0, 0.25)
    x = filt[var].dropna()
    can_log = bool((x > 0).all())
    use_log = c4.checkbox("Log x-axis", value=can_log, disabled=not can_log,
                          help="Only available for strictly positive variables.")
    q1, q3 = x.quantile([0.25, 0.75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    iqr_m = (x < lo) | (x > hi)
    z_m = ((x - x.mean()) / x.std()).abs() > z
    m = st.columns(4)
    m[0].metric("Values analysed", fmt_int(len(x)))
    m[1].metric("Flagged by IQR", f"{iqr_m.sum():,}", f"{iqr_m.mean() * 100:.1f}%", delta_color="off")
    m[2].metric("Flagged by Z-score", f"{z_m.sum():,}", f"{z_m.mean() * 100:.1f}%", delta_color="off")
    m[3].metric("Flagged by both", f"{(iqr_m & z_m).sum():,}", f"{(iqr_m & z_m).mean() * 100:.1f}%", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        header("Distribution with fences", "Dashed: IQR upper fence. Dotted: Z-score cutoff.")
        xs = x.sample(min(len(x), 30000), random_state=0)
        zc = x.mean() + z * x.std()
        # Plotly histograms mis-bin on log axes, so bin in log10 space ourselves.
        tf = (lambda v: np.log10(v)) if use_log else (lambda v: v)
        fig = go.Figure(go.Histogram(x=tf(xs), nbinsx=70, marker_color="#8fb3cf"))
        fig.add_vline(x=float(tf(hi)), line_dash="dash", line_color=RED)
        if zc > 0 or not use_log:
            fig.add_vline(x=float(tf(zc)), line_dash="dot", line_color=ORANGE)
        if lo > x.min() and not (use_log and lo <= 0):
            fig.add_vline(x=float(tf(lo)), line_dash="dash", line_color=RED)
        xaxis = dict(title=NUM_LABELS[var])
        if use_log:
            a, b = int(np.floor(np.log10(xs.min()))), int(np.ceil(np.log10(xs.max())))
            xaxis.update(tickmode="array", tickvals=list(range(a, b + 1)), ticktext=[f"{10.0 ** i:g}" for i in range(a, b + 1)])
        style(fig, 360, xaxis=xaxis, yaxis=dict(title="Orders"))
        show(fig)
    with c2:
        header("Box plot by review score", "One box per star rating; points beyond the whiskers are potential outliers.")
        bp = filt.dropna(subset=[var, "review_score"])
        bp = bp.sample(min(len(bp), 15000), random_state=0)
        bp["Stars"] = bp["review_score"].astype(int).astype(str) + " star"
        fig = px.box(bp.sort_values("review_score"), x=var, y="Stars", orientation="h", points="outliers",
                     color_discrete_sequence=[BLUE], category_orders={"Stars": [f"{i} star" for i in range(5, 0, -1)]})
        fig.update_traces(marker=dict(size=3, opacity=0.35, color=ORANGE), line=dict(width=1.5), fillcolor="rgba(47,111,159,0.25)")
        style(fig, 360, xaxis=dict(title=NUM_LABELS[var], type="log" if use_log else "linear"), yaxis=dict(title="", showgrid=False))
        show(fig)

    st.markdown("## Scatter plots: unusual combinations")
    st.write(
        "A value can look normal on each axis and still be odd in combination, such as a light parcel with a very "
        "high freight bill. Pick a pair of variables: points beyond the IQR fence (multiplier from the slider above) "
        "on one or both axes are coloured."
    )
    c1, c2, c3, c4 = st.columns([1, 2.4, 1, 1.2])
    ds = c1.selectbox("Table", list(SCATTER_PAIRS))
    frame = scatter_frame(ds)
    pairs = [p for p in SCATTER_PAIRS[ds] if frame is not None and all(c in frame.columns for c in p)]
    if not pairs:
        st.info(f"No usable columns for {ds.lower()} were found in PROCESSED_DATA.")
    else:
        def lab(c):
            return SCATTER_LABELS.get(c, c)

        pi = c2.selectbox("Pair of variables", list(range(len(pairs))),
                          format_func=lambda i: f"{lab(pairs[i][0])} vs {lab(pairs[i][1])}")
        a, b = pairs[pi]
        d = frame[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        can_log = bool(len(d)) and bool((d >= 0).all().all())
        log_axes = c3.checkbox("Log axes", value=can_log, disabled=not can_log,
                               help="Only for non-negative variables; zeros cannot be drawn on a log axis.")
        colour = c4.checkbox("Colour IQR outliers", value=True)
        if d.empty:
            st.info("No complete rows for this pair.")
        else:
            fx, fy = iqr_flags(d[a], k), iqr_flags(d[b], k)
            both, one = fx & fy, fx ^ fy
            m = st.columns(4)
            m[0].metric("Points analysed", fmt_int(len(d)))
            m[1].metric(f"Outside fence on {lab(a)} only", f"{(fx & ~fy).sum():,}")
            m[2].metric(f"Outside fence on {lab(b)} only", f"{(fy & ~fx).sum():,}")
            m[3].metric("Outside on both", f"{both.sum():,}")

            plot = d.sample(min(len(d), 5000), random_state=42)
            cat = pd.Series("Within fences", index=d.index)
            cat[one] = "Outside on one variable"
            cat[both] = "Outside on both"
            cat = cat.loc[plot.index]
            if log_axes:
                keep = (plot[a] > 0) & (plot[b] > 0)
                dropped = int((~keep).sum())
                plot, cat = plot[keep], cat[keep]
            else:
                dropped = 0
            fig = go.Figure()
            groups = [("Within fences", "#8fb3cf", 0.35), ("Outside on one variable", ORANGE, 0.6), ("Outside on both", RED, 0.8)]
            for name, colr, op in groups:
                sel = (cat == name) if colour else (cat == cat)
                if not colour and name != "Within fences":
                    continue
                fig.add_scatter(x=plot.loc[sel, a], y=plot.loc[sel, b], mode="markers", name=name if colour else "Orders",
                                marker=dict(size=5, color=colr, opacity=op), hoverinfo="skip" if len(plot) > 4000 else None)
            style(fig, 420, legend=colour,
                  xaxis=dict(title=lab(a), type="log" if log_axes else "linear", showgrid=True, gridcolor=GRID),
                  yaxis=dict(title=lab(b), type="log" if log_axes else "linear"))
            show(fig)
            st.caption(
                f"A random sample of up to 5,000 points is drawn; fences and counts use all {len(d):,} rows. "
                + (f"{dropped} zero values are left off the log axes. " if dropped else "")
                + ("Orders follow the sidebar filters; items and products do not." if ds == "Orders" else
                   "Items and products come from the cleaned tables and are not affected by the sidebar filters.")
            )
    callout(
        "A visual check, not an automatic count",
        "Scatter plots show which extreme points are expected (big orders are heavy and bulky) and which are odd "
        "(a heavy, cheap-to-ship parcel). Points outside the fence on both axes are usually consistent with each "
        "other; points outside on one axis only deserve a closer look. They are candidates to investigate, not errors.",
    )

    st.markdown("## What happens if we treat the outliers?")
    tt = treatment_table(filt[var], k)
    c1, c2 = st.columns([3, 2])
    with c1:
        table(tt.style.format({"Count": "{:,.0f}", "Mean": "{:,.2f}", "Median": "{:,.2f}", "Std dev": "{:,.2f}", "Skewness": "{:.2f}"}))
    with c2:
        fig = go.Figure(go.Bar(x=tt["Treatment"], y=tt["Skewness"], marker_color=BLUE,
                               text=[f"{v:.2f}" for v in tt["Skewness"]], textposition="outside", cliponaxis=False))
        style(fig, 280, yaxis=dict(title="Skewness"), xaxis=dict(tickangle=-15))
        show(fig)
    callout(
        "Why we did not simply delete outliers",
        "Winsorizing keeps every row and still lowers the standard deviation. Trimming removes thousands of real, "
        "expensive or very slow orders and changes the mean a lot. The median barely moves under capping, which is "
        "why it is the safer summary here.",
    )

    st.markdown("## IQR and Z-score disagree (full dataset, all tables)")
    bench = pd.DataFrame([
        ["Orders", "total_price", 7913, 1709], ["Orders", "total_freight", 9941, 1581], ["Orders", "total_weight_g", 13804, 1962],
        ["Orders", "total_volume_cm3", 9164, 1706], ["Orders", "approval_hours", 9141, 1095], ["Orders", "seller_handling_days", 5660, 1871],
        ["Orders", "carrier_transit_days", 5071, 1581], ["Orders", "total_delivery_days", 4899, 1571], ["Orders", "days_late", 4875, 1333],
        ["Orders", "distance_km", 7622, 1564], ["Items", "price", 8427, 1966], ["Items", "freight_value", 12134, 2041],
        ["Items", "product_weight_g", 15807, 2955], ["Items", "product_volume_cm3", 10004, 2498],
    ], columns=["Dataset", "Variable", "IQR", "Z-score"])
    bench["Label"] = bench["Dataset"] + " / " + bench["Variable"]
    bench = bench.sort_values("IQR")
    fig = go.Figure()
    fig.add_bar(y=bench["Label"], x=bench["IQR"], orientation="h", name="IQR rule", marker_color=BLUE)
    fig.add_bar(y=bench["Label"], x=bench["Z-score"], orientation="h", name="Z-score > 3", marker_color=ORANGE)
    style(fig, 520, legend=True, barmode="group", xaxis=dict(title="Flagged values"), yaxis=dict(showgrid=False))
    show(fig)
    st.caption("From Notebook 03 (full data, not filtered). The mean and standard deviation that the Z-score uses are themselves distorted by the skew, so it flags far fewer values than the quartile-based rule.")

    st.markdown("## Missing values")
    c1, c2 = st.columns([2, 3])
    with c1:
        header("Share of missing values by column (master_orders)")
        ms = missing_share[missing_share > 0].head(12).sort_values()
        fig = go.Figure(go.Bar(x=ms.values * 100, y=ms.index, orientation="h", marker_color=BLUE,
                               text=[f"{v * 100:.1f}%" for v in ms.values], textposition="outside", cliponaxis=False))
        style(fig, 400, xaxis=dict(title="% missing", range=[0, max(ms.max() * 100 * 1.2, 5)]), yaxis=dict(showgrid=False),
              margin=dict(l=10, r=40, t=10, b=10))
        show(fig)
    with c2:
        callout(
            "Missing does not always mean random",
            "Most gaps are structural: no delivery date for orders that were cancelled, no comment text for reviewers "
            "who only left a star rating. Review comments are missing for about 59% of reviews, which is a large "
            "share, so we checked whether that missingness carries information.",
        )
        st.write("")

    c1, c2 = st.columns(2)
    d = filt.dropna(subset=["review_score", "has_comment"])
    with c1:
        header("Share of reviews with a written comment, by star rating")
        g = d.groupby("review_score")["has_comment"].mean() * 100
        fig = go.Figure(go.Bar(x=[f"{int(i)} star" for i in g.index], y=g.values, marker_color=[RED, ORANGE, GREY, "#8fb3cf", BLUE],
                               text=[f"{v:.0f}%" for v in g.values], textposition="outside", cliponaxis=False))
        style(fig, 320, yaxis=dict(title="% with comment", range=[0, 100]))
        show(fig)
    with c2:
        header("... and by delivery outcome")
        g2 = filt.dropna(subset=["lateness_bucket", "has_comment"]).groupby("lateness_bucket")["has_comment"].mean().reindex(present_buckets) * 100
        fig = go.Figure(go.Bar(x=[BUCKET_SHORT[b] for b in g2.index], y=g2.values, marker_color=[BUCKET_COLORS[b] for b in g2.index],
                               text=[f"{v:.0f}%" for v in g2.values], textposition="outside", cliponaxis=False))
        style(fig, 320, yaxis=dict(title="% with comment", range=[0, 100]))
        show(fig)
    ct = pd.crosstab(filt["lateness_bucket"], filt["has_comment"])
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        callout(
            "Unhappy customers write more (missing not at random)",
            f"Chi-square test of comment vs lateness bucket: chi2 = {chi2:,.0f}, dof = {dof}, p = {fmt_p(p)}. "
            "1-star reviewers comment far more often than 4-5 star reviewers, so a text analysis of comments "
            "over-represents complaints. We keep this in mind when reading the text-mining results.",
        )

    st.markdown("## Can we predict who writes a comment?")
    st.write(
        "A logistic regression asks whether review score and days late carry enough information to predict that a "
        "comment exists (standardized inputs, balanced class weights, 80/20 split, as in Notebook 03)."
    )
    cmres = comment_model(filt[["review_score", "days_late", "has_comment"]])
    if cmres is None:
        st.info("Not enough complete reviews in the current selection to fit the model.")
    else:
        m = st.columns(5)
        m[0].metric("Accuracy", f"{cmres['accuracy']:.3f}")
        m[1].metric("Precision", f"{cmres['precision']:.3f}")
        m[2].metric("Recall", f"{cmres['recall']:.3f}")
        m[3].metric("F1-score", f"{cmres['f1']:.3f}")
        m[4].metric("Always 'no comment'", f"{cmres['baseline']:.3f}", help="Accuracy of the lazy rule that predicts the majority class for everyone.")
        c1, c2 = st.columns(2)
        with c1:
            header("Confusion matrix (test set)", f"{cmres['n_test']:,} held-out reviews.")
            cm = cmres["cm"]
            rows_ = ["Actual: no comment", "Actual: has comment"]
            cols_ = ["Predicted: no comment", "Predicted: has comment"]
            fig = go.Figure(go.Heatmap(z=cm, x=cols_, y=rows_, text=[[f"{v:,}" for v in r] for r in cm], texttemplate="%{text}",
                                       colorscale="Blues", showscale=False, textfont=dict(size=15)))
            style(fig, 280, yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False), margin=dict(l=10, r=10, t=10, b=10))
            show(fig)
        with c2:
            header("What the model learned", "Change in log-odds of a comment per one standard deviation.")
            cf = cmres["coef"]
            names = ["Review score", "Days late"]
            vals_ = [cf["review_score"], cf["days_late"]]
            fig = go.Figure(go.Bar(x=vals_, y=names, orientation="h", marker_color=[BLUE if v < 0 else ORANGE for v in vals_],
                                   text=[f"{v:+.2f}" for v in vals_], textposition="outside", cliponaxis=False))
            lim = max(abs(v) for v in vals_) * 1.5 + 0.05
            style(fig, 280, xaxis=dict(title="Standardized coefficient", range=[-lim, lim], zeroline=True, zerolinecolor=GREY),
                  yaxis=dict(autorange="reversed", showgrid=False), margin=dict(l=10, r=40, t=10, b=10))
            show(fig)
        signs = cmres["coef"]["review_score"] < 0 and cmres["coef"]["days_late"] > 0
        beat = cmres["accuracy"] > cmres["baseline"]
        callout(
            "The direction is clear, the prediction is weak",
            ("Lower scores and longer delays both raise the odds of a comment, matching the chi-square result above. "
             if signs else "The coefficient signs are shown above. ")
            + f"But accuracy is {cmres['accuracy']:.2f} and F1 only {cmres['f1']:.2f}; always guessing 'no comment' "
            f"would be right {cmres['baseline']:.2f} of the time, so the model "
            + ("is only slightly better than that. " if beat else "does not beat that simple rule on accuracy (balanced weights trade accuracy for catching commenters). ")
            + "Whether someone writes a comment depends on much more than score and lateness, which is consistent "
            "with missing-not-at-random: association, not an easy rule to predict.",
        )
        st.caption("With no sidebar filters this reproduces Notebook 03 (accuracy 0.58, F1 0.47, 19,166 test reviews).")

    st.markdown("## Imputation experiment")
    st.write(
        "To compare imputation methods fairly, 15% of known product dimensions were hidden (4,942 values per feature), "
        "filled in with each method, and compared with the truth. Errors are only comparable within a feature."
    )
    imp = pd.DataFrame([
        ["Weight (g)", "Mean", 2550.90, 4368.12], ["Weight (g)", "Median", 1964.98, 4649.87], ["Weight (g)", "kNN", 2130.60, 3944.67], ["Weight (g)", "MICE", 2300.23, 3523.95],
        ["Length (cm)", "Mean", 12.65, 16.77], ["Length (cm)", "Median", 11.92, 17.74], ["Length (cm)", "kNN", 11.02, 15.91], ["Length (cm)", "MICE", 10.68, 15.06],
        ["Height (cm)", "Mean", 9.68, 13.89], ["Height (cm)", "Median", 9.29, 14.52], ["Height (cm)", "kNN", 9.51, 13.77], ["Height (cm)", "MICE", 9.25, 12.76],
        ["Width (cm)", "Mean", 9.40, 12.04], ["Width (cm)", "Median", 8.77, 12.45], ["Width (cm)", "kNN", 7.43, 10.71], ["Width (cm)", "MICE", 7.75, 10.62],
    ], columns=["Feature", "Method", "MAE", "RMSE"])
    c1, c2 = st.columns([1, 1])
    feat = c1.selectbox("Feature", imp["Feature"].unique())
    metric = c2.radio("Error", ["RMSE", "MAE"], horizontal=True)
    sub = imp[imp["Feature"] == feat]
    best = sub[metric].idxmin()
    fig = go.Figure(go.Bar(x=sub["Method"], y=sub[metric], marker_color=[GREEN if i == best else "#b9c6d2" for i in sub.index],
                           text=[f"{v:,.2f}" for v in sub[metric]], textposition="outside", cliponaxis=False))
    style(fig, 320, yaxis=dict(title=f"{metric} (lower is better)", rangemode="tozero"))
    show(fig)
    callout(
        "The best method depends on the error you care about",
        "MICE has the lowest RMSE for all four features because it uses the other dimensions to predict the "
        "missing one and avoids large misses. For the very skewed weight, the median wins on MAE: it is a safe "
        "typical value but misses badly on heavy products. For categorical <code>primary_category</code> (no real gaps), "
        "15% random masking recovered <b>9.45%</b> of 4,942 masked labels using the global mode (<code>bed_bath_table</code>) - "
        "why predictive or item-level imputation would be needed if categories were missing.",
    )


# ===========================================================================
# 5. REVIEWS (TEXT MINING)
# ===========================================================================
def terms_chart(df, color, title):
    d = df.sort_values("mean_tfidf", ascending=False).head(15).sort_values("mean_tfidf")
    header(title)
    fig = go.Figure(go.Bar(x=d["mean_tfidf"], y=d["term"], orientation="h", marker_color=color))
    style(fig, 430, xaxis=dict(title="Mean TF-IDF"), yaxis=dict(showgrid=False), margin=dict(l=10, r=10, t=10, b=10))
    show(fig)


def page_text():
    page_intro(
        "5. Reviews (text mining)",
        "Notebook 04, section A - Portuguese TF-IDF with negation handling",
        "41% of reviews contain text (40,774 of 98,673). We looked at which words separate complaints from praise.",
    )
    c1, c2 = st.columns(2)
    with c1:
        header("Star rating distribution")
        v = filt["review_score"].dropna().astype(int).value_counts().sort_index()
        fig = go.Figure(go.Bar(x=[f"{i} star" for i in v.index], y=v.values,
                               marker_color=[RED, ORANGE, GREY, "#8fb3cf", BLUE], text=[f"{n:,}" for n in v.values],
                               textposition="outside", cliponaxis=False))
        style(fig, 320, yaxis=dict(title="Orders"))
        show(fig)
    with c2:
        header("Share of low reviews (1-2 stars) by delivery outcome")
        g = filt.dropna(subset=["lateness_bucket", "review_score"]).groupby("lateness_bucket")["review_score"].apply(lambda s: (s <= 2).mean() * 100)
        g = g.reindex(present_buckets)
        fig = go.Figure(go.Bar(x=[BUCKET_SHORT[b] for b in g.index], y=g.values, marker_color=[BUCKET_COLORS[b] for b in g.index],
                               text=[f"{v:.0f}%" for v in g.values], textposition="outside", cliponaxis=False))
        style(fig, 320, yaxis=dict(title="% of reviews", range=[0, 100]))
        show(fig)

    st.markdown("## Cleaning Portuguese text without losing the complaint")
    st.write(
        "Standard stopword lists delete words such as *nao* (not), *nunca* (never) and *sem* (without). Doing that "
        "turns \"nao recebi\" (I did not receive) into \"recebi\" (I received) and makes complaints look like praise. "
        "We removed accents, punctuation and neutral stopwords but kept the negations, and used unigrams plus bigrams "
        "(40,556 reviews x 4,385 terms)."
    )
    if nlp_top is None or nlp_dist is None:
        st.info("NLP result files were not found.")
        return
    groups = list(nlp_top["group"].unique())
    pairs = [("1-star", "5-star"), ("late", "on-time")]
    pairs = [p for p in pairs if p[0] in groups and p[1] in groups]
    if pairs:
        pick = st.radio("Compare", [f"{a} vs {b}" for a, b in pairs], horizontal=True)
        a, b = pairs[[f"{x} vs {y}" for x, y in pairs].index(pick)]
        c1, c2 = st.columns(2)
        with c1:
            terms_chart(nlp_top[nlp_top["group"] == a], RED, f"Top terms in {a} reviews")
        with c2:
            terms_chart(nlp_top[nlp_top["group"] == b], GREEN, f"Top terms in {b} reviews")

    st.markdown("## Words that separate complaints from praise")
    n = st.slider("Terms per side", 6, 20, 12)
    dd = pd.concat([nlp_dist.nlargest(n, "difference"), nlp_dist.nsmallest(n, "difference")]).sort_values("difference")
    header("TF-IDF in 1-star minus TF-IDF in 5-star reviews", "Right: used more by unhappy customers. Left: used more by happy customers.")
    fig = go.Figure(go.Bar(x=dd["difference"], y=dd["term"], orientation="h",
                           marker_color=[RED if v > 0 else GREEN for v in dd["difference"]]))
    fig.add_vline(x=0, line_color=GREY)
    style(fig, max(380, n * 26), xaxis=dict(title="Difference in mean TF-IDF"), yaxis=dict(showgrid=False))
    show(fig)
    callout(
        "Complaints are about not receiving the order",
        "The strongest complaint terms are <i>nao recebi</i> (did not receive), <i>nao entregue</i> (not delivered) and "
        "<i>nao chegou</i> (did not arrive). Praise is dominated by <i>otimo</i>, <i>recomendo</i> and <i>prazo</i> "
        "(on time). Words about the product itself appear far less, which supports the numeric finding that delivery "
        "drives ratings. Remember that comments over-represent unhappy customers (section 4).",
    )


# ===========================================================================
# 6. GEOGRAPHY & NETWORK
# ===========================================================================
STATE_XY = {
    "AC": (-70.5, -9.0), "AL": (-36.6, -9.6), "AM": (-64.6, -4.1), "AP": (-51.9, 1.4), "BA": (-41.7, -12.9),
    "CE": (-39.3, -5.2), "DF": (-47.8, -15.8), "ES": (-40.7, -19.6), "GO": (-49.6, -15.9), "MA": (-45.3, -5.1),
    "MG": (-44.7, -18.5), "MS": (-54.8, -20.5), "MT": (-55.9, -12.9), "PA": (-52.3, -3.9), "PB": (-36.8, -7.1),
    "PE": (-37.9, -8.4), "PI": (-42.8, -7.7), "PR": (-51.6, -24.6), "RJ": (-42.7, -22.2), "RN": (-36.5, -5.8),
    "RO": (-62.8, -10.9), "RR": (-61.3, 2.1), "RS": (-53.5, -29.8), "SC": (-50.5, -27.3), "SE": (-37.4, -10.6),
    "SP": (-48.7, -22.2), "TO": (-48.3, -10.2),
}


def page_geo():
    page_intro(
        "6. Geography & network",
        "Notebook 04, section B - shipping network graph, route risk",
        "Every delivery is an edge from a seller state to a customer state. The resulting directed network shows "
        "which states act as hubs and which corridors are slow.",
    )
    if state_metrics is None or routes is None:
        st.info("Graph result files were not found.")
        return

    st.markdown("## The shipping network")
    c1, c2 = st.columns([3, 2])
    with c1:
        top = routes[routes["seller_state"] != routes["customer_state"]].nlargest(40, "order_count")
        inbound = routes.groupby("customer_state")["order_count"].sum()
        late_in = state_metrics.set_index("state")["dest_late_rate"] if "dest_late_rate" in state_metrics else pd.Series(dtype=float)
        fig = go.Figure()
        for _, r in top.iterrows():
            if r["seller_state"] in STATE_XY and r["customer_state"] in STATE_XY:
                (x0, y0), (x1, y1) = STATE_XY[r["seller_state"]], STATE_XY[r["customer_state"]]
                fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                line=dict(color="rgba(47,111,159,0.25)", width=max(0.6, np.sqrt(r["order_count"]) / 12)))
        st_list = [s for s in STATE_XY if s in inbound.index]
        fig.add_scatter(
            x=[STATE_XY[s][0] for s in st_list], y=[STATE_XY[s][1] for s in st_list], mode="markers+text", text=st_list,
            textposition="middle center", textfont=dict(size=10, color="white"),
            marker=dict(size=[14 + np.sqrt(inbound[s]) / 5 for s in st_list], color=[late_in.get(s, np.nan) * 100 for s in st_list],
                        colorscale="YlOrRd", colorbar=dict(title="Late %", thickness=12), line=dict(color="white", width=1)),
            customdata=[[int(inbound[s])] for s in st_list], hovertemplate="%{text}<br>Inbound orders: %{customdata[0]:,}<extra></extra>",
            showlegend=False)
        style(fig, 480, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), margin=dict(l=0, r=0, t=0, b=0))
        show(fig)
        st.caption("Bubble size: orders received. Colour: late rate for orders delivered to that state. Lines: 40 busiest inter-state routes.")
    with c2:
        sm = state_metrics.sort_values("pagerank", ascending=False).head(6)
        header("Most central states (PageRank)")
        table(sm[["state", "pagerank", "betweenness", "in_degree", "out_degree", "intra_state_share"]].rename(columns={
            "state": "State", "pagerank": "PageRank", "betweenness": "Betweenness",
            "in_degree": "In deg", "out_degree": "Out deg", "intra_state_share": "Intra-state"}).style.format(
            {"PageRank": "{:.3f}", "Betweenness": "{:.3f}", "In deg": "{:.0f}", "Out deg": "{:.0f}", "Intra-state": "{:.1%}"}))
        with st.expander("All 27 states (degrees, strength & intra-state share)"):
            full_sm = state_metrics.sort_values("pagerank", ascending=False).copy()
            cols_sm = [c for c in ["state", "pagerank", "betweenness", "in_degree", "out_degree",
                                   "in_strength", "out_strength", "intra_state_share", "dest_late_rate", "dest_avg_review"] if c in full_sm]
            table(full_sm[cols_sm].style.format({
                "pagerank": "{:.3f}", "betweenness": "{:.3f}", "in_degree": "{:.0f}", "out_degree": "{:.0f}",
                "in_strength": "{:,.0f}", "out_strength": "{:,.0f}", "intra_state_share": "{:.1%}",
                "dest_late_rate": "{:.1%}", "dest_avg_review": "{:.2f}",
            }))
        callout(
            "One hub, no communities",
            "Sao Paulo ships about 61% of all inter-state orders and holds almost all the betweenness. The 27-state "
            "graph is so densely connected through SP that community detection finds a single group (modularity 0), "
            "so we judge states by centrality and by route performance instead.",
        )

    st.markdown("## Which destinations are slowest?")
    sa = (filt.groupby("customer_state").agg(orders=("order_id", "count"), late=("is_late", "mean"), review=("review_score", "mean"))
          .reset_index().query("orders >= 100").sort_values("late", ascending=False).head(15).sort_values("late"))
    header("Late rate by destination state", "States with at least 100 orders in the current selection.")
    fig = go.Figure(go.Bar(x=sa["late"] * 100, y=sa["customer_state"], orientation="h",
                           marker_color=sa["late"] * 100, marker_colorscale="YlOrRd",
                           text=[f"{l * 100:.1f}%  ({n:,} orders)" for l, n in zip(sa["late"], sa["orders"])],
                           textposition="outside", cliponaxis=False))
    style(fig, 460, xaxis=dict(title="Late rate (%)", range=[0, max(sa["late"].max() * 100 * 1.35, 5)]), yaxis=dict(showgrid=False),
          margin=dict(l=10, r=40, t=10, b=10))
    show(fig)

    st.markdown("## Distance and delivery time")
    dist = filt.dropna(subset=["distance_km", "total_delivery_days"]).copy()
    dist["bin"] = pd.cut(dist["distance_km"], [0, 100, 250, 500, 750, 1000, 1500, 2000, 4000], include_lowest=True)
    gb = dist.groupby("bin", observed=True)["total_delivery_days"]
    qq = pd.DataFrame({"q25": gb.quantile(0.25), "med": gb.median(), "q75": gb.quantile(0.75), "n": gb.size()}).reset_index()
    qq["label"] = [f"{int(max(i.left, 0))}-{int(i.right)}" for i in qq["bin"]]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=qq["label"], y=qq["q75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=qq["label"], y=qq["q25"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(47,111,159,0.18)", name="Middle 50% of orders")
        fig.add_scatter(x=qq["label"], y=qq["med"], mode="lines+markers", line=dict(color=BLUE, width=2.5), name="Median")
        style(fig, 340, legend=True, xaxis=dict(title="Seller-to-customer distance (km)"), yaxis=dict(title="Delivery days", rangemode="tozero"))
        show(fig)
    with c2:
        rho = stats.spearmanr(dist["distance_km"], dist["total_delivery_days"])[0]
        callout(
            "Distance matters, but spread grows with it",
            f"Spearman correlation between distance and delivery time is <b>{rho:.2f}</b>. Beyond ~1,000 km the "
            "middle half of orders already spans many days, so far destinations are both slower and less predictable.",
        )
        cross = filt.groupby("is_cross_state").agg(late=("is_late", "mean"), days=("total_delivery_days", "mean"), review=("review_score", "mean"))
        if len(cross) == 2:
            table(pd.DataFrame({
                "": ["Same state", "Cross state"],
                "Late rate": [pct(cross.loc[0, "late"]), pct(cross.loc[1, "late"])],
                "Avg days": [f"{cross.loc[0, 'days']:.1f}", f"{cross.loc[1, 'days']:.1f}"],
                "Avg review": [f"{cross.loc[0, 'review']:.2f}", f"{cross.loc[1, 'review']:.2f}"],
            }))

    st.markdown("## Origin-destination late-rate heatmap")
    hm_min = st.slider("Minimum orders to show route cell", 10, 200, 30, 10, key="hm_min_orders",
                       help="Matches Notebook 04 (default: 30 orders) so tiny cells do not distort the colour scale.")
    hm_df = routes[routes["order_count"] >= hm_min].copy()
    if hm_df.empty:
        st.info("No routes meet that order threshold.")
    else:
        piv_rate = hm_df.pivot(index="seller_state", columns="customer_state", values="late_rate") * 100
        piv_cnt = hm_df.pivot(index="seller_state", columns="customer_state", values="order_count")
        piv_rate = piv_rate.dropna(how="all").dropna(axis=1, how="all")
        piv_cnt = piv_cnt.reindex(index=piv_rate.index, columns=piv_rate.columns)
        fig = go.Figure(go.Heatmap(
            z=piv_rate.values, x=piv_rate.columns.tolist(), y=piv_rate.index.tolist(),
            customdata=piv_cnt.values, colorscale="YlOrRd", zmin=0, zmax=min(40.0, float(np.nanmax(piv_rate.values))),
            colorbar=dict(title="Late %", thickness=12),
            hovertemplate="Seller: %{y} -> Customer: %{x}<br>Late rate: %{z:.1f}%<br>Orders: %{customdata:,.0f}<extra></extra>",
        ))
        style(fig, max(340, 22 * len(piv_rate)), xaxis=dict(title="Customer state (destination)"),
              yaxis=dict(title="Seller state (origin)", autorange="reversed", showgrid=False), margin=dict(l=10, r=10, t=10, b=10))
        show(fig)

    st.markdown("## Worst routes (seller state to customer state)")
    c1, c2 = st.columns([1, 2])
    min_o = c1.slider("Minimum orders per route", 20, 500, 100, 20,
                      help="Rare routes with a handful of orders can look extreme by chance.")
    key = c2.radio("Rank by", ["Late rate", "Late order count", "Lowest review"], horizontal=True)
    rf = routes[routes["order_count"] >= min_o].copy()
    rf["route"] = rf["seller_state"] + " to " + rf["customer_state"]
    rf["late_orders"] = rf["late_rate"] * rf["order_count"]
    if key == "Late rate":
        w = rf.nlargest(12, "late_rate").sort_values("late_rate")
        xv, label = w["late_rate"] * 100, [f"{v * 100:.1f}%" for v in w["late_rate"]]
        xtitle = "Late rate (%)"
    elif key == "Late order count":
        w = rf.nlargest(12, "late_orders").sort_values("late_orders")
        xv, label = w["late_orders"], [f"{int(round(v)):,} late" for v in w["late_orders"]]
        xtitle = "Late orders"
    else:
        w = rf.nsmallest(12, "avg_review").sort_values("avg_review", ascending=False)
        xv, label = w["avg_review"], [f"{v:.2f}" for v in w["avg_review"]]
        xtitle = "Average review"
    fig = go.Figure(go.Bar(x=xv, y=w["route"], orientation="h", marker_color=RED,
                           text=[f"{t}  ({n:,} orders)" for t, n in zip(label, w["order_count"])], textposition="outside", cliponaxis=False))
    style(fig, 420, xaxis=dict(title=xtitle, range=[0, float(xv.max()) * 1.4]),
          yaxis=dict(showgrid=False), margin=dict(l=10, r=40, t=10, b=10))
    show(fig)
    st.caption(f"{len(rf)} of {len(routes)} routes have at least {min_o} orders.")
    callout(
        "Long corridors from Sao Paulo to the north-east are the problem",
        "In the full data SP to AL (26% late) and SP to MA (21%) are the worst well-populated routes, while SP to RJ "
        "(15.5%) produces the largest number of late orders because it carries so much traffic. Within a state the late "
        "rate is only about 6%.",
    )


# ===========================================================================
# 7. SELLERS: REDUCTION & CLUSTERING
# ===========================================================================
FEATURES = [
    "total_orders", "total_items", "total_revenue", "total_freight", "avg_price", "avg_freight", "avg_weight_g",
    "avg_handling_days", "median_handling_days", "avg_transit_days", "avg_delivery_days", "avg_days_late", "late_rate",
    "seller_shipped_late_rate", "avg_review_score", "low_review_rate", "distinct_categories", "states_served",
    "avg_distance_km", "freight_ratio", "cross_state_share",
]
SKEWED = ["total_orders", "total_items", "total_revenue", "total_freight", "avg_price", "avg_freight", "avg_weight_g",
          "distinct_categories", "states_served", "freight_ratio"]


@st.cache_data(show_spinner=False)
def seller_matrix(df):
    X = df[FEATURES].astype(float)
    X = X.fillna(X.median())
    for c in SKEWED:
        X[c] = np.log1p(X[c].clip(lower=0))
    return ((X - X.mean()) / X.std(ddof=0)).values


@st.cache_data(show_spinner=False)
def pca_summary(X):
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
    o = np.argsort(vals)[::-1]
    return vals[o], vecs[:, o]


@st.cache_data(show_spinner="Running K-Means for k = 2 ... 10")
def kmeans_sweep(X):
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500, random_state=42).fit(X)
        rows.append([k, km.inertia_, silhouette_score(X, km.labels_),
                     calinski_harabasz_score(X, km.labels_), davies_bouldin_score(X, km.labels_)])
    return pd.DataFrame(rows, columns=["k", "inertia", "silhouette", "ch", "db"])


def nice(c):
    return c.replace("_", " ").replace("avg ", "average ")


@st.cache_data(show_spinner=False)
def svd_check(X):
    """PCA by covariance eigen-decomposition vs PCA by SVD of the (already centred) matrix."""
    n = X.shape[0]
    _, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    vals, vecs = pca_summary(X)
    cos = np.abs(np.sum(vecs * Vt.T, axis=0))
    return vals, S ** 2 / (n - 1), cos


TSNE_SCORES = {
    5: (0.9708, 0.9295, 1.5840),
    15: (0.9798, 0.9476, 1.6970),
    30: (0.9794, 0.9524, 1.6369),
    50: (0.9772, 0.9530, 1.5485),
    100: (0.9754, 0.9553, 1.3633),
}


def _fit_single_ae(X_tr, X_te, X_all, k):
    """Train symmetric 21->16->k->16->21 MLP autoencoder (NB 05 Cell 5)."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error
    from sklearn.neural_network import MLPRegressor
    pca_k = PCA(n_components=k, random_state=42).fit(X_tr)
    pca_tr = mean_squared_error(X_tr, pca_k.inverse_transform(pca_k.transform(X_tr)))
    pca_te = mean_squared_error(X_te, pca_k.inverse_transform(pca_k.transform(X_te)))
    ae = MLPRegressor(
        hidden_layer_sizes=(16, k, 16), activation="relu", solver="adam", alpha=1e-4,
        batch_size=64, learning_rate_init=0.003, max_iter=400, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25, random_state=42,
    ).fit(X_tr, X_tr)
    ae_tr = mean_squared_error(X_tr, ae.predict(X_tr))
    ae_te = mean_squared_error(X_te, ae.predict(X_te))
    a = X_all
    for idx in range(2):
        a = np.maximum(0, a @ ae.coefs_[idx] + ae.intercepts_[idx])
    metrics = {
        "k": str(k), "PCA Train MSE": pca_tr, "AE Train MSE": ae_tr,
        "PCA Test MSE": pca_te, "AE Test MSE": ae_te,
        "Gain %": (pca_te - ae_te) / pca_te * 100,
    }
    return metrics, ae.loss_curve_, a


@st.cache_data(show_spinner="Training Autoencoder latent space (cached afterwards) ...")
def autoencoder_latent(X, k=6):
    from sklearn.model_selection import train_test_split
    X_tr, X_te = train_test_split(X, test_size=0.20, random_state=42)
    _, _, Z = _fit_single_ae(X_tr, X_te, X, k)
    return Z


@st.cache_data(show_spinner="Training Autoencoder suite for k = 2, 3, 6, 9 (cached afterwards) ...")
def autoencoder_suite(X):
    from sklearn.model_selection import train_test_split
    X_tr, X_te = train_test_split(X, test_size=0.20, random_state=42)
    rows, losses = [], {}
    for k in (2, 3, 6, 9):
        m, lc, _ = _fit_single_ae(X_tr, X_te, X, k)
        rows.append(m)
        losses[k] = lc
    return pd.DataFrame(rows), losses


@st.cache_data(show_spinner="Running t-SNE for this perplexity (several seconds, cached afterwards) ...")
def tsne_embed(X, perp):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", max_iter=1000,
                random_state=42).fit_transform(X)


def get_tsne(X, perp):
    # Perplexity 15 is saved in the scorecard; otherwise fall back to cached t-SNE computation.
    if perp == 15 and {"tsne_dim1", "tsne_dim2"} <= set(sellers.columns):
        return sellers[["tsne_dim1", "tsne_dim2"]].values
    return tsne_embed(X, perp)


def get_ae_2d(X):
    if {"ae_dim1", "ae_dim2"} <= set(sellers.columns):
        return sellers[["ae_dim1", "ae_dim2"]].values
    return autoencoder_latent(X, k=2)



def page_sellers():
    page_intro(
        "7. Sellers: dimensionality reduction & clustering",
        "Notebook 05 - PCA, t-SNE, autoencoder, K-Means, Ward, DBSCAN, Mean-Shift, OPTICS",
        "3,095 sellers described by 21 behaviour features: volume, price, handling and transit time, lateness, "
        "reviews and reach. Features were median-imputed, the 10 skewed ones were log-transformed, then all were standardized.",
    )
    if sellers is None or not all(c in sellers.columns for c in FEATURES):
        st.info("seller_scorecard.csv was not found or is missing seller feature columns.")
        return
    X = seller_matrix(sellers)

    st.markdown("## How many dimensions do we really need? (PCA)")
    vals, vecs = pca_summary(X)
    ratio = vals / vals.sum()
    cum = np.cumsum(ratio)
    k80, k90, k95 = [int(np.argmax(cum >= t) + 1) for t in (0.80, 0.90, 0.95)]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=[f"PC{i + 1}" for i in range(len(ratio))], y=ratio * 100, name="Variance per component", marker_color="#8fb3cf")
        fig.add_scatter(x=[f"PC{i + 1}" for i in range(len(ratio))], y=cum * 100, name="Cumulative", yaxis="y2",
                        mode="lines+markers", line=dict(color=BLUE, width=2.5))
        for t in (80, 90, 95):
            fig.add_hline(y=t, yref="y2", line_dash="dot", line_color=GREY)
        style(fig, 360, legend=True, yaxis=dict(title="% of variance"),
              yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 102], showgrid=False),
              xaxis=dict(tickangle=-45))
        show(fig)
    with c2:
        callout(
            "Six components keep 80% of the information",
            f"The first two components hold {cum[1] * 100:.0f}% of the variance; <b>{k80}</b> components are needed for 80%, "
            f"<b>{k90}</b> for 90% and <b>{k95}</b> for 95%. The 21 features are strongly redundant (for example, orders, "
            "items and revenue move together), so a 2-D map is a useful summary but loses detail.",
        )
    header("What each component means (loadings)", "Red: feature pushes the component up. Blue: pushes it down.")
    L = pd.DataFrame(vecs[:, :3], index=[nice(f) for f in FEATURES], columns=["PC1", "PC2", "PC3"])
    fig = go.Figure(go.Heatmap(z=L.values, x=L.columns, y=L.index, zmin=-0.5, zmax=0.5, colorscale="RdBu_r",
                               text=np.round(L.values, 2), texttemplate="%{text}", textfont=dict(size=10), colorbar=dict(thickness=12)))
    style(fig, 560, yaxis=dict(autorange="reversed", showgrid=False), margin=dict(l=10, r=10, t=10, b=10))
    show(fig)

    header("PCA and SVD give the same answer", "Truncated SVD of the standardized matrix is PCA without ever forming the covariance matrix.")
    ev, sv, cosv = svd_check(X)
    c1, c2 = st.columns([2, 3])
    with c1:
        m1, m2 = st.columns(2)
        m1.metric("Largest eigenvalue gap", f"{np.max(np.abs(ev - sv)):.1e}", help="max |covariance eigenvalue - sigma^2 / (n-1)| over all 21 components")
        m2.metric("Smallest |cos| of angle", f"{cosv.min():.6f}", help="1.0 means the PCA direction and the SVD direction are identical (the sign is arbitrary).")
        st.caption("Computed live from the seller matrix. Differences at the 1e-15 level are floating-point noise.")
    with c2:
        table(pd.DataFrame({
            "Component": [f"PC{i + 1}" for i in range(6)],
            "Eigenvalue (covariance)": ev[:6], "sigma^2 / (n-1) (SVD)": sv[:6], "|cos| of directions": cosv[:6],
        }).style.format({"Eigenvalue (covariance)": "{:.4f}", "sigma^2 / (n-1) (SVD)": "{:.4f}", "|cos| of directions": "{:.6f}"}))

    st.markdown("## Linear vs non-linear reduction")
    c1, c2 = st.columns(2)
    with c1:
        header("t-SNE: neighbourhood preservation & KL divergence", "Trustworthiness (1 = perfect) and KL divergence by perplexity.")
        perp_vals = list(TSNE_SCORES)
        fig = go.Figure()
        fig.add_scatter(x=perp_vals, y=[TSNE_SCORES[p][0] for p in perp_vals], mode="lines+markers", name="Trustworthiness (k=10)", line=dict(color=BLUE, width=2.5))
        fig.add_scatter(x=perp_vals, y=[TSNE_SCORES[p][1] for p in perp_vals], mode="lines+markers", name="Trustworthiness (k=30)", line=dict(color=GREEN, width=2.5))
        fig.add_scatter(x=perp_vals, y=[TSNE_SCORES[p][2] for p in perp_vals], mode="lines+markers", name="KL divergence", yaxis="y2", line=dict(color=ORANGE, width=2, dash="dash"))
        fig.add_hline(y=0.8058, line_dash="dot", line_color=RED, annotation_text="2-D PCA baseline (0.81)", annotation_position="bottom right")
        style(fig, 340, legend=True, xaxis=dict(title="Perplexity", type="category"),
              yaxis=dict(title="Trustworthiness", range=[0.78, 1.0]),
              yaxis2=dict(title="KL divergence", overlaying="y", side="right", range=[1.2, 1.9], showgrid=False))
        show(fig)
    with c2:
        header("Autoencoder vs PCA: reconstruction error", "Test-set MSE by bottleneck size (lower is better).")
        ks = ["2", "3", "6", "9"]
        fig = go.Figure()
        fig.add_bar(x=ks, y=[0.5255, 0.4052, 0.1808, 0.0816], name="PCA", marker_color="#b9c6d2")
        fig.add_bar(x=ks, y=[0.4044, 0.3001, 0.1116, 0.0684], name="Autoencoder", marker_color=BLUE,
                    text=["-23%", "-26%", "-38%", "-16%"], textposition="outside")
        style(fig, 340, legend=True, barmode="group", xaxis=dict(title="Latent dimensions"), yaxis=dict(title="Test MSE"))
        show(fig)
    if st.checkbox("Inspect live Autoencoder training curves & Train vs Test MSE (~7s first run)", value=False, key="ae_live_check"):
        ae_df, ae_losses = autoencoder_suite(X)
        c1, c2 = st.columns([3, 2])
        with c1:
            header("Autoencoder training convergence (loss curve)", "Adam with early stopping on a 15% validation split.")
            fig = go.Figure()
            for (k_ae, curve), col_ae in zip(ae_losses.items(), [GREY, ORANGE, BLUE, GREEN]):
                fig.add_scatter(x=list(range(1, len(curve) + 1)), y=curve, mode="lines",
                                name=f"k = {k_ae} ({len(curve)} epochs)", line=dict(color=col_ae, width=2))
            style(fig, 300, legend=True, xaxis=dict(title="Epoch"), yaxis=dict(title="Training MSE loss"))
            show(fig)
        with c2:
            header("Train vs unseen test MSE")
            table(ae_df.style.format({"PCA Train MSE": "{:.4f}", "AE Train MSE": "{:.4f}",
                                      "PCA Test MSE": "{:.4f}", "AE Test MSE": "{:.4f}", "Gain %": "{:+.1f}%"}))
    else:
        st.caption("Summary scores are from Notebook 05; toggle above to run the live 4-bottleneck MLP autoencoder comparison and view convergence curves.")
    callout(
        "Non-linear methods keep more structure",
        "A 2-D PCA preserves 81% of local neighbourhoods; t-SNE preserves about 98%. The autoencoder reconstructs the "
        "held-out sellers 16-38% better than PCA at the same size, with its biggest gain at 6 latent dimensions. "
        "Perplexity 15 gave the best local score, so that map is used below.",
    )

    header("t-SNE maps by perplexity", "Low perplexity splits the cloud into small islands, high perplexity smooths it out.")
    c1, c2 = st.columns([2, 1])
    perp = c1.radio("Perplexity", list(TSNE_SCORES), index=1, horizontal=True)
    allfive = c2.checkbox("Show all five side by side", help="Computes the four maps that are not saved (about a minute the first time).")
    tier_list = [t for t in TIER_COLORS if t in set(sellers["tier"])]
    if allfive:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=3, subplot_titles=[f"Perplexity {p}" for p in TSNE_SCORES], horizontal_spacing=0.05, vertical_spacing=0.12)
        for i, p in enumerate(TSNE_SCORES):
            Y = get_tsne(X, p)
            for t in tier_list:
                sel = (sellers["tier"] == t).values
                fig.add_scatter(x=Y[sel, 0], y=Y[sel, 1], mode="markers", name=t, marker=dict(size=3, color=TIER_COLORS[t], opacity=0.6),
                                showlegend=(i == 0), hoverinfo="skip", row=i // 3 + 1, col=i % 3 + 1)
        style(fig, 640, legend=True)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        show(fig)
    else:
        Y = get_tsne(X, perp)
        tm = pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1], "tier": sellers["tier"].values})
        fig = px.scatter(tm, x="x", y="y", color="tier", opacity=0.6, color_discrete_map=TIER_COLORS, category_orders={"tier": tier_list})
        fig.update_traces(marker=dict(size=4), hoverinfo="skip")
        style(fig, 420, legend=True, xaxis=dict(title="t-SNE 1", showticklabels=False, showgrid=False), yaxis=dict(title="t-SNE 2", showticklabels=False))
        show(fig)
    t10, t30, kl = TSNE_SCORES[perp]
    m = st.columns(4)
    m[0].metric("Trustworthiness (k=10)", f"{t10:.3f}", "best of the five" if perp == 15 else None, delta_color="off")
    m[1].metric("Trustworthiness (k=30)", f"{t30:.3f}")
    m[2].metric("Final KL divergence", f"{kl:.3f}")
    m[3].metric("2-D PCA baseline (k=10 / 30)", "0.806 / 0.806")
    st.caption("Scores are from Notebook 05. Only perplexity 15 is saved in the scorecard; the other maps are re-run here with the notebook's settings "
               "(PCA initialisation, 1,000 iterations, seed 42). Axes of a t-SNE map have no meaning; only which sellers sit close together does.")

    header("Seller map", "Each dot is a seller.")
    c1, c2 = st.columns([2, 3])
    emb = c1.radio("Embedding", ["PCA", "t-SNE", "Autoencoder"], horizontal=True)
    color_opts = ["Seller tier", "K-Means cluster", "Ward cluster", "DBSCAN", "Mean-Shift", "OPTICS"]
    by = c2.selectbox("Colour by", color_opts, key="seller_map_color")
    sc = sellers.copy()
    if emb == "PCA" and {"pca_dim1", "pca_dim2"} <= set(sc.columns):
        xcol, ycol = "pca_dim1", "pca_dim2"
    elif emb == "t-SNE":
        coords = get_tsne(X, 15)
        sc["_map_x"], sc["_map_y"] = coords[:, 0], coords[:, 1]
        xcol, ycol = "_map_x", "_map_y"
    elif emb == "Autoencoder":
        coords = get_ae_2d(X)
        sc["_map_x"], sc["_map_y"] = coords[:, 0], coords[:, 1]
        xcol, ycol = "_map_x", "_map_y"
    else:
        sc["_map_x"], sc["_map_y"] = (X @ vecs[:, 0]), (X @ vecs[:, 1])
        xcol, ycol = "_map_x", "_map_y"

    if by == "Seller tier":
        col = "tier"
    elif by == "K-Means cluster" and "cluster_kmeans" in sc:
        sc["_grp"] = "Cluster " + sc["cluster_kmeans"].astype(str)
        col = "_grp"
    elif by == "Ward cluster" and "cluster_ward" in sc:
        sc["_grp"] = "Ward " + sc["cluster_ward"].astype(str)
        col = "_grp"
    elif by == "DBSCAN" and "cluster_dbscan" in sc:
        sc["_grp"] = np.where(sc["cluster_dbscan"] == -1, "Noise (-1)", "Cluster " + sc["cluster_dbscan"].astype(str))
        col = "_grp"
    elif by == "Mean-Shift" and "cluster_meanshift" in sc:
        top_modes = set(sc["cluster_meanshift"].value_counts().head(4).index)
        sc["_grp"] = sc["cluster_meanshift"].apply(lambda v: f"Mode {v}" if v in top_modes else "Minor mode")
        col = "_grp"
    elif by == "OPTICS" and "cluster_optics" in sc:
        sc["_grp"] = np.where(sc["cluster_optics"] == -1, "Noise (-1)", "Cluster " + sc["cluster_optics"].astype(str))
        col = "_grp"
    else:
        col = "tier"

    fig = px.scatter(sc, x=xcol, y=ycol, color=col, opacity=0.6, color_discrete_map=TIER_COLORS if col == "tier" else None,
                     color_discrete_sequence=[BLUE, ORANGE, GREEN, RED, GREY, "#7a5fa0"],
                     hover_data={"total_orders": True, "late_rate": ":.2f", "avg_review_score": ":.2f", xcol: False, ycol: False},
                     category_orders={"tier": list(TIER_COLORS)} if col == "tier" else None)
    fig.update_traces(marker=dict(size=5))
    style(fig, 460, legend=True, xaxis=dict(title=f"{emb} 1", showgrid=True, gridcolor=GRID), yaxis=dict(title=f"{emb} 2"))
    show(fig)

    st.markdown("## Clustering: does the data contain natural groups?")
    sw = kmeans_sweep(X)
    c1, c2 = st.columns(2)
    with c1:
        header("Elbow: within-cluster variance", "No sharp elbow, a sign of a continuum rather than separate groups.")
        fig = go.Figure(go.Scatter(x=sw["k"], y=sw["inertia"], mode="lines+markers", line=dict(color=BLUE, width=2.5)))
        style(fig, 300, xaxis=dict(title="Number of clusters k", dtick=1), yaxis=dict(title="Inertia"))
        show(fig)
    with c2:
        header("Silhouette score by k", "Closer to 1 means clearer clusters.")
        bestk = int(sw.loc[sw["silhouette"].idxmax(), "k"])
        fig = go.Figure(go.Bar(x=sw["k"], y=sw["silhouette"], marker_color=[GREEN if k == bestk else "#b9c6d2" for k in sw["k"]],
                               text=[f"{v:.2f}" for v in sw["silhouette"]], textposition="outside", cliponaxis=False))
        style(fig, 300, xaxis=dict(title="Number of clusters k", dtick=1), yaxis=dict(title="Silhouette", range=[0, max(sw["silhouette"]) * 1.3]))
        show(fig)

    header("Four algorithms compared", "Results from Notebook 05 on the same standardized 21-feature matrix.")
    table(pd.DataFrame([
        ["K-Means (k=3)", 3, "0.190", "5% / 40% / 55% of sellers", "Best silhouette among k-means runs"],
        ["DBSCAN (eps 4.37, minPts 42)", 1, "n/a", "1 cluster, 3.2% noise", "Same single cluster for every eps tried"],
        ["Mean-Shift (q = 0.25)", 7, "0.333", "96% in one mode", "High silhouette is misleading: 6 tiny outlier groups"],
        ["OPTICS (xi 0.04)", 1, "n/a", "1 cluster, 0% noise", "No density valleys, also in the 6-D latent space"],
    ], columns=["Algorithm", "Clusters", "Silhouette", "Sizes", "Reading"]))
    st.caption("Each algorithm is examined in detail (validity scores, dendrogram, k-distance plot, bandwidth sweep, reachability plot) on page 8.")
    callout(
        "Sellers form a continuum, so we segment by business rules",
        "Silhouettes never exceed 0.19 for balanced clusterings, and the density methods (DBSCAN, OPTICS) find a single "
        "blob. Rather than forcing arbitrary clusters, we used K-Means as supporting evidence and defined four "
        "interpretable seller tiers from volume, review score, late rate and price, plus a 0-100 reliability score.",
    )

    st.markdown("## Seller tiers")
    tiers = [t for t in TIER_COLORS if t in set(sellers["tier"])]
    cnt = sellers["tier"].value_counts().reindex(tiers)
    c1, c2 = st.columns([2, 3])
    with c1:
        header("Sellers per tier")
        fig = go.Figure(go.Bar(x=cnt.values, y=cnt.index, orientation="h", marker_color=[TIER_COLORS[t] for t in cnt.index],
                               text=[f"{v:,}  ({v / cnt.sum() * 100:.0f}%)" for v in cnt.values], textposition="outside", cliponaxis=False))
        style(fig, 280, xaxis=dict(title="Sellers", range=[0, cnt.max() * 1.35]), yaxis=dict(autorange="reversed", showgrid=False),
              margin=dict(l=10, r=40, t=10, b=10))
        show(fig)
    with c2:
        header("Reliability score by tier (0-100)")
        fig = go.Figure()
        for t in tiers:
            fig.add_box(y=sellers.loc[sellers["tier"] == t, "reliability_score"].dropna(), name=t, marker_color=TIER_COLORS[t], boxpoints=False)
        style(fig, 280, yaxis=dict(title="Reliability score"), xaxis=dict(showgrid=False))
        show(fig)

    prof_cols = ["total_orders", "avg_price", "avg_handling_days", "avg_transit_days", "late_rate", "avg_review_score", "avg_distance_km"]
    prof = sellers.groupby("tier")[prof_cols].median().reindex(tiers)
    norm = (prof - prof.min()) / (prof.max() - prof.min()).replace(0, 1)
    header("Typical seller in each tier (median)", "Shading is scaled within each column; numbers are the actual medians.")
    txt = prof.copy()
    fmt = {"total_orders": "{:.0f}", "avg_price": "{:.0f}", "avg_handling_days": "{:.1f}", "avg_transit_days": "{:.1f}",
           "late_rate": "{:.0%}", "avg_review_score": "{:.2f}", "avg_distance_km": "{:.0f}"}
    text = [[fmt[c].format(txt.loc[t, c]) for c in prof_cols] for t in tiers]
    fig = go.Figure(go.Heatmap(z=norm.values, x=[nice(c) for c in prof_cols], y=tiers, text=text, texttemplate="%{text}",
                               colorscale="Blues", showscale=False, textfont=dict(size=13)))
    style(fig, 260, yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(side="top", showgrid=False))
    show(fig)

    st.markdown("## Sellers to coach first")
    st.write("At-risk sellers with the most orders: the largest effect on customers for the least effort.")
    ar = sellers[(sellers["tier"] == "Tier 4 - At-risk")].nlargest(15, "total_orders")
    show_cols = ["seller_id", "seller_state", "total_orders", "late_rate", "avg_review_score", "avg_handling_days", "reliability_score"]
    show_cols = [c for c in show_cols if c in ar]
    tbl = ar[show_cols].copy()
    tbl["seller_id"] = tbl["seller_id"].str[:8] + "..."
    table(tbl.rename(columns={c: nice(c) for c in show_cols}).style.format(
        {"late rate": "{:.0%}", "average review score": "{:.2f}", "average handling days": "{:.1f}", "reliability score": "{:.0f}", "total orders": "{:,.0f}"}))


# ===========================================================================
# 8. CLUSTERING ALGORITHMS
# ===========================================================================
MS_SWEEP = pd.DataFrame(
    [[0.10, 3.931, 7, 90.8, 0.2408], [0.15, 4.231, 9, 93.6, 0.2571], [0.20, 4.479, 9, 93.5, 0.2546],
     [0.25, 4.701, 7, 96.1, 0.3328], [0.30, 4.909, 8, 95.6, 0.3194]],
    columns=["Quantile", "Bandwidth", "Modes found", "Largest mode (%)", "Silhouette"],
)
MS_SIZES = [2975, 86, 14, 9, 7, 3, 1]  # q = 0.25; the last mode is what remains of 3,095 after the six listed in the notebook
CLUSTER_METRICS = [("Silhouette", "silhouette", "higher"), ("Calinski-Harabasz", "ch", "higher"), ("Davies-Bouldin", "db", "lower")]


@st.cache_data(show_spinner=False)
def ward_linkage(X):
    from scipy.cluster.hierarchy import linkage
    return linkage(X, method="ward", metric="euclidean")


@st.cache_data(show_spinner="Scoring Ward cuts for k = 2 ... 10")
def ward_sweep(X):
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    Z = ward_linkage(X)
    rows = []
    for k in range(2, 11):
        lab = fcluster(Z, k, criterion="maxclust")
        rows.append([k, silhouette_score(X, lab), calinski_harabasz_score(X, lab), davies_bouldin_score(X, lab)])
    return pd.DataFrame(rows, columns=["k", "silhouette", "ch", "db"])


@st.cache_data(show_spinner=False)
def kdistance(X, m):
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=m).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


def find_knee(sorted_d):
    """Point of maximum perpendicular distance from the chord (same rule as Notebook 05)."""
    pts = np.vstack((np.arange(len(sorted_d)), sorted_d)).T
    v = pts[-1] - pts[0]
    v = v / np.linalg.norm(v)
    rel = pts - pts[0]
    dist = np.linalg.norm(rel - np.outer(rel @ v, v), axis=1)
    i = int(np.argmax(dist))
    return i, float(sorted_d[i])


@st.cache_data(show_spinner=False)
def dbscan_run(X, eps, m):
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=eps, min_samples=m).fit(X)
    core = np.zeros(len(db.labels_), dtype=bool)
    core[db.core_sample_indices_] = True
    return db.labels_, core


@st.cache_data(show_spinner=False)
def kmeans_sample_silhouette(X, k=3):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500, random_state=42).fit(X)
    return km.labels_, silhouette_samples(X, km.labels_), float(silhouette_score(X, km.labels_))


@st.cache_data(show_spinner="Fitting OPTICS on all 3,095 sellers (cached afterwards) ...")
def optics_run(X, min_samples=25):
    from sklearn.cluster import OPTICS
    op = OPTICS(min_samples=min_samples, max_eps=np.inf, metric="euclidean", cluster_method="xi", xi=0.04,
                min_cluster_size=0.015, n_jobs=1).fit(X)
    return op.reachability_, op.core_distances_, op.ordering_, op.labels_


def page_clustering():
    page_intro(
        "8. Clustering algorithms",
        "Notebook 05 - K-Means, Ward, DBSCAN, Mean-Shift, OPTICS under the microscope",
        "Page 8 gives the verdict (sellers form a continuum). This page shows the evidence for each algorithm on the same "
        "standardized 21-feature seller matrix (and the 6-D autoencoder latent space): validity scores, per-sample silhouettes, "
        "the dendrogram & merge jumps, DBSCAN k-distance curves, Mean-Shift sweep and OPTICS reachability plots.",
    )
    if sellers is None or not all(c in sellers.columns for c in FEATURES):
        st.info("seller_scorecard.csv was not found or is missing seller feature columns.")
        return
    X = seller_matrix(sellers)
    n = len(X)
    vals, vecs = pca_summary(X)
    pcs = X @ vecs[:, :2]

    # ------------------------------------------------------------------ validity scores
    st.markdown("## Three scores for choosing k")
    st.write(
        "Silhouette and Calinski-Harabasz are better when higher, Davies-Bouldin when lower. "
        "If the three scores recommend different k, the data has no single natural number of groups."
    )
    km = kmeans_sweep(X)
    wd = ward_sweep(X)
    fig = make_subplots_scores(km, wd)
    show(fig)
    picks = {}
    for algo, df in (("K-Means", km), ("Ward", wd)):
        picks[algo] = {
            name: int(df.loc[df[col].idxmax() if better == "higher" else df[col].idxmin(), "k"])
            for name, col, better in CLUSTER_METRICS
        }
    pk = pd.DataFrame(picks).T.reset_index().rename(columns={"index": "Algorithm"})
    pk.columns = ["Algorithm"] + [f"Best k by {c}" for c in pk.columns[1:]]
    table(pk)
    disagree = any(len(set(v.values())) > 1 for v in picks.values())
    callout(
        "The scores do not agree on one k" if disagree else "The scores agree on k",
        ("The scores recommend different values of k (table above), and even the best silhouette stays low: "
         f"{km['silhouette'].max():.2f} for K-Means and {wd['silhouette'].max():.2f} for Ward, where 0.5 or more would mean clear structure. "
         "That is another sign of a continuum rather than separate seller groups.")
        if disagree else
        (f"All three scores point at the same k for each algorithm, but the best silhouette is only {km['silhouette'].max():.2f} (K-Means) "
         f"and {wd['silhouette'].max():.2f} (Ward), which is weak structure."),
    )
    with st.expander("See the numbers"):
        t = km.rename(columns={"inertia": "K-Means inertia", "silhouette": "K-Means silhouette", "ch": "K-Means CH", "db": "K-Means DB"}).merge(
            wd.rename(columns={"silhouette": "Ward silhouette", "ch": "Ward CH", "db": "Ward DB"}), on="k")
        table(t.style.format({c: "{:.3f}" for c in t.columns if c != "k"} | {"K-Means inertia": "{:,.0f}", "K-Means CH": "{:.1f}", "Ward CH": "{:.1f}"}))

    st.markdown("## K-Means per-sample silhouette profile")
    k_sil = st.slider("Inspect K-Means clusters k", 2, 8, 3, key="km_sil_k")
    km_lab_s, sample_sils, mean_sil = kmeans_sample_silhouette(X, k_sil)
    c1, c2 = st.columns([3, 2])
    with c1:
        header(f"Silhouette coefficient per seller (k = {k_sil})", "Sorted within each cluster; values below 0 sit closer to a neighbouring cluster.")
        fig = go.Figure()
        y_low = 0
        pal = [BLUE, ORANGE, GREEN, RED, GREY, "#7a5fa0", "#2b8cbe", "#a63603"]
        for cid in range(k_sil):
            s_vals = np.sort(sample_sils[km_lab_s == cid])
            yc = np.arange(y_low, y_low + len(s_vals))
            fig.add_scatter(x=s_vals, y=yc, mode="lines", line=dict(width=0.6, color=pal[cid % len(pal)]),
                            fill="tozerox", name=f"C{cid} (n={len(s_vals):,}, mean={s_vals.mean():.2f})")
            y_low += len(s_vals) + 45
        fig.add_vline(x=mean_sil, line_dash="dash", line_color=RED, annotation_text=f"Mean = {mean_sil:.3f}", annotation_position="top right")
        fig.add_vline(x=0, line_color=GREY, line_width=1)
        style(fig, 360, legend=True, xaxis=dict(title="Silhouette coefficient s_i", range=[min(-0.2, float(sample_sils.min()) - 0.05), 0.7]),
              yaxis=dict(title="Sellers grouped by cluster", showticklabels=False, showgrid=False))
        show(fig)
    with c2:
        neg_share = float((sample_sils < 0).mean())
        st.metric("Mean silhouette score", f"{mean_sil:.3f}")
        st.metric("Sellers with negative silhouette", f"{int((sample_sils < 0).sum()):,}", f"{neg_share * 100:.1f}%", delta_color="off")
        callout(
            "Overlapping cluster borders",
            f"At k = {k_sil}, <b>{neg_share * 100:.1f}%</b> of sellers have a negative silhouette coefficient, confirming that the clusters "
            "touch and overlap along a continuous cloud rather than forming separated islands.",
        )

    # ------------------------------------------------------------------ Ward
    st.markdown("## Ward hierarchical clustering")
    st.write(
        "Ward merges, at each step, the two groups whose union increases the total within-group variance the least. "
        "The dendrogram shows the last 30 merges (the number in brackets is how many sellers a branch holds); "
        "the red line is where the tree is cut."
    )
    k_w = st.slider("Cut the tree into k clusters", 2, 10, 3, key="ward_k")
    from scipy.cluster.hierarchy import dendrogram, fcluster
    Z = ward_linkage(X)
    dd = dendrogram(Z, truncate_mode="lastp", p=30, no_plot=True)
    xs, ys = [], []
    for ic, dc in zip(dd["icoord"], dd["dcoord"]):
        xs += list(ic) + [None]
        ys += list(dc) + [None]
    cut = float((Z[-k_w, 2] + Z[-(k_w - 1), 2]) / 2)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BLUE, width=1.6), hoverinfo="skip"))
    fig.add_hline(y=cut, line_dash="dash", line_color=RED, annotation_text=f"cut for k = {k_w}", annotation_position="top right")
    style(fig, 400, xaxis=dict(title="Branches (sellers per branch)", tickmode="array", tickvals=[5 + 10 * i for i in range(len(dd["ivl"]))],
                               ticktext=dd["ivl"], tickangle=-60, showgrid=False),
          yaxis=dict(title="Ward merge distance"))
    show(fig)
    lab = fcluster(Z, k_w, criterion="maxclust")
    sizes = pd.Series(lab).value_counts().sort_values(ascending=False)
    c1, c2 = st.columns([3, 2])
    with c1:
        header("Merge-height jump plot (last 15 splits)", "A sharp elbow in merge distance indicates natural clusters.")
        last_merges = Z[-15:, 2][::-1]
        merge_k = list(range(2, 17))
        fig = go.Figure(go.Scatter(x=merge_k, y=last_merges, mode="lines+markers", line=dict(color="#7a5fa0", width=2.5), marker=dict(size=7)))
        fig.add_vline(x=k_w, line_dash="dash", line_color=RED, annotation_text=f"k = {k_w}", annotation_position="top right")
        style(fig, 300, xaxis=dict(title="Number of clusters k", dtick=1), yaxis=dict(title="Merge distance"))
        show(fig)
        header(f"Cluster sizes at k = {k_w}")
        fig = go.Figure(go.Bar(x=[f"Cluster {i + 1}" for i in range(len(sizes))], y=sizes.values / n * 100, marker_color=BLUE,
                               text=[f"{v:,}  ({v / n * 100:.0f}%)" for v in sizes.values], textposition="outside", cliponaxis=False))
        style(fig, 280, yaxis=dict(title="% of sellers", range=[0, sizes.max() / n * 100 * 1.25]))
        show(fig)
    with c2:
        row = wd[wd["k"] == k_w].iloc[0]
        m1, m2 = st.columns(2)
        m1.metric("Silhouette", f"{row['silhouette']:.3f}")
        m2.metric("Davies-Bouldin", f"{row['db']:.2f}", help="Lower is better.")
        st.metric("Calinski-Harabasz", f"{row['ch']:.0f}")
        top = ", ".join(f"{Z[-i, 2]:.1f}" for i in range(1, 6))
        callout("Reading the tree",
                "A natural number of groups would show as one very long vertical stretch with no merges. Distances of the last five "
                f"merges (top first): {top}. At k = 3 the split is about 51% / 40% / 9% in the notebook, very different from "
                "K-Means (5% / 40% / 55%), so the two methods do not agree on what the groups are.")

    # ------------------------------------------------------------------ DBSCAN
    st.markdown("## DBSCAN: density-based clusters")
    st.write(
        "DBSCAN needs a radius (eps) and a minimum neighbour count (minPts = 2 x dimensions). We check both the full "
        "21-D standardized matrix (minPts = 42) and the 6-D autoencoder latent space (minPts = 12) to verify whether "
        "density valleys appear once distance concentration is reduced."
    )
    db_space = st.radio("Feature space", ["21D Standardized (minPts = 42)", "6D Autoencoder latent (minPts = 12)"],
                        horizontal=True, key="db_space")
    is_db_6d = db_space.startswith("6D")
    X_db = autoencoder_latent(X, k=6) if is_db_6d else X
    c1, c2 = st.columns([1, 1])
    minpts = int(c1.number_input("minPts", 5, 100, 12 if is_db_6d else 42, 1, key=f"db_minpts_{is_db_6d}"))
    kd = kdistance(X_db, minpts)
    knee_i, knee_eps = find_knee(kd)
    default_eps = round(float(np.percentile(kd, 92)), 2) if is_db_6d else 4.37
    max_eps = max(5.0, round(float(kd.max()) * 1.05, 1)) if is_db_6d else 8.0
    min_eps = 0.2 if is_db_6d else 2.0
    eps = float(c2.slider(
        "eps", min_eps, max_eps, min(max(default_eps, min_eps), max_eps), 0.01, key=f"db_eps_{is_db_6d}",
        help="Notebook 05 used eps = 4.37 in 21D (80% of knee) and the 92nd percentile (~2.20) in the 6D autoencoder space.",
    ))
    c1, c2 = st.columns([3, 2])
    with c1:
        header("k-distance curve", f"{'6D Autoencoder' if is_db_6d else '21D Standardized'} space: sellers sorted by distance to neighbour {minpts}.")
        fig = go.Figure(go.Scatter(x=np.arange(n), y=kd, mode="lines", line=dict(color=BLUE, width=2.5), hoverinfo="skip"))
        fig.add_scatter(x=[knee_i], y=[knee_eps], mode="markers", marker=dict(size=10, color=ORANGE), name="Knee")
        fig.add_hline(y=eps, line_dash="dash", line_color=RED, annotation_text=f"eps = {eps:.2f}", annotation_position="top left")
        style(fig, 340, xaxis=dict(title="Sellers (sorted)"), yaxis=dict(title=f"Distance to neighbour {minpts}"))
        show(fig)
        st.caption(f"Detected knee: eps = {knee_eps:.2f} (orange dot).")
    lab_db, core = dbscan_run(X_db, eps, minpts)
    noise = lab_db == -1
    border = ~core & ~noise
    n_clu = len(set(lab_db)) - (1 if -1 in lab_db else 0)
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("Clusters found", n_clu)
        m2.metric("Noise", f"{noise.sum():,}", f"{noise.mean() * 100:.1f}%", delta_color="off")
        m3, m4 = st.columns(2)
        m3.metric("Core points", f"{core.sum():,}", f"{core.mean() * 100:.1f}%", delta_color="off")
        m4.metric("Border points", f"{border.sum():,}", f"{border.mean() * 100:.1f}%", delta_color="off")
        st.caption("Core: at least minPts neighbours within eps. Border: within eps of a core point but not core itself. Noise: neither.")

    header("Where the core, border and noise points sit", "Projected onto the first two principal components.")
    fig = go.Figure()
    for name, mask_, colr, sz in (("Core", core, "#8fb3cf", 4), ("Border", border, ORANGE, 5), ("Noise", noise, RED, 6)):
        fig.add_scatter(x=pcs[mask_, 0], y=pcs[mask_, 1], mode="markers", name=f"{name} ({mask_.sum():,})",
                        marker=dict(size=sz, color=colr, opacity=0.65), hoverinfo="skip")
    style(fig, 400, legend=True, xaxis=dict(title="PC1", showgrid=True, gridcolor=GRID), yaxis=dict(title="PC2"))
    show(fig)

    header("Is the single cluster an accident of eps?", "The same minPts at five radii around the knee.")
    rows = []
    for f in (0.65, 0.80, 0.90, 1.00, 1.15):
        e = round(knee_eps * f, 2)
        lb, _ = dbscan_run(X_db, e, minpts)
        rows.append([e, len(set(lb)) - (1 if -1 in lb else 0), int((lb == -1).sum()), (lb == -1).mean()])
    sweep = pd.DataFrame(rows, columns=["eps", "Clusters", "Noise sellers", "Noise share"])
    table(sweep.style.format({"eps": "{:.2f}", "Noise sellers": "{:,}", "Noise share": "{:.1%}"}))
    one = bool((sweep["Clusters"] <= 1).all())
    callout(
        "One blob at every radius" if one else "The number of clusters changes with eps",
        ("Every eps gives a single cluster; widening it only absorbs more of the noise points into that one cluster. "
         "Even in the 6-D autoencoder space (eps 2.20, minPts 12: 1 cluster, 4.4% noise), there is no density valley "
         "separating distinct seller groups.")
        if one else
        "More than one cluster appears at some radii. Check the noise share: clusters that only exist at one eps are fragile.",
    )

    # ------------------------------------------------------------------ Mean-Shift
    st.markdown("## Mean-Shift: modes of the density")
    st.write(
        "Mean-Shift slides a window of radius h (the bandwidth) uphill until it reaches a density peak; every seller that "
        "ends at the same peak forms one cluster. The bandwidth is set from a quantile of pairwise distances."
    )
    c1, c2 = st.columns(2)
    with c1:
        header("Modes found and size of the biggest one", "Each bar is the share of sellers in the largest mode.")
        fig = go.Figure(go.Bar(x=[f"q = {q:.2f}" for q in MS_SWEEP["Quantile"]], y=MS_SWEEP["Largest mode (%)"],
                               marker_color=[RED if q == 0.25 else "#b9c6d2" for q in MS_SWEEP["Quantile"]],
                               text=[f"{m} modes, {p:.0f}%" for m, p in zip(MS_SWEEP["Modes found"], MS_SWEEP["Largest mode (%)"])],
                               textposition="outside", cliponaxis=False))
        style(fig, 320, yaxis=dict(title="% of sellers in largest mode", range=[0, 112]))
        show(fig)
    with c2:
        header("Silhouette by bandwidth", "The best-looking score comes from the most lopsided split.")
        fig = go.Figure(go.Bar(x=[f"h = {b:.2f}" for b in MS_SWEEP["Bandwidth"]], y=MS_SWEEP["Silhouette"],
                               marker_color=[RED if q == 0.25 else "#b9c6d2" for q in MS_SWEEP["Quantile"]],
                               text=[f"{v:.2f}" for v in MS_SWEEP["Silhouette"]], textposition="outside", cliponaxis=False))
        style(fig, 320, yaxis=dict(title="Silhouette", range=[0, 0.42]))
        show(fig)
    table(MS_SWEEP.style.format({"Quantile": "{:.2f}", "Bandwidth": "{:.3f}", "Largest mode (%)": "{:.1f}", "Silhouette": "{:.4f}"}))
    header("Mode sizes at q = 0.25 (the notebook's choice)", "Log scale: one mode holds almost everyone.")
    fig = go.Figure(go.Bar(y=[f"Mode {i + 1}" for i in range(len(MS_SIZES))], x=MS_SIZES, orientation="h", marker_color=BLUE,
                           text=[f"{v:,}" for v in MS_SIZES], textposition="outside", cliponaxis=False))
    style(fig, 280, xaxis=dict(title="Sellers (log scale)", type="log", range=[-0.1, 3.9]), yaxis=dict(autorange="reversed", showgrid=False))
    show(fig)
    st.caption("Saved numbers from Notebook 05 (re-running Mean-Shift for five bandwidths is slow). The notebook lists the six biggest modes; "
               "the seventh holds the one seller left over from 3,095.")
    callout(
        "A high silhouette that means little",
        "q = 0.25 has the best silhouette (0.33) because 96% of sellers sit in one mode and the other six modes are tiny groups of extreme "
        "sellers (86, 14, 9, 7, 3 and 1). That finds outliers, not segments, which is why K-Means and business-rule tiers are used instead.",
    )

    # ------------------------------------------------------------------ OPTICS
    st.markdown("## OPTICS: reachability plot")
    st.write(
        "OPTICS orders the sellers so that neighbours are next to each other and records how far each is from the previous group "
        "(its reachability distance). A cluster is a valley in this plot; a cluster boundary is a tall wall (xi = 0.04, min cluster size = 1.5%)."
    )
    opt_space = st.radio("Feature space", ["21D Standardized (minPts = 25)", "6D Autoencoder latent (minPts = 20)"],
                         horizontal=True, key="opt_space")
    is_op_6d = opt_space.startswith("6D")
    X_op = autoencoder_latent(X, k=6) if is_op_6d else X
    reach, core_d, order, lab_op = optics_run(X_op, 20 if is_op_6d else 25)
    r = reach[order]
    fin = r[np.isfinite(r)]
    cap = float(np.percentile(fin, 99.5))
    p90 = float(np.percentile(fin, 90))
    cut_e = st.slider("Horizontal cut height (eps')", round(float(np.percentile(fin, 50)), 2), round(cap, 2), round(p90, 2), 0.05,
                      key=f"opt_cut_{is_op_6d}",
                      help="Cutting the plot at one height gives the same result as DBSCAN with that eps. Default: the 90th percentile.")
    fig = go.Figure(go.Bar(x=np.arange(n), y=np.where(np.isfinite(r), np.minimum(r, cap), np.nan), marker_color=GREEN if is_op_6d else BLUE,
                           marker_line_width=0, hoverinfo="skip"))
    fig.add_hline(y=cut_e, line_dash="dash", line_color=RED, annotation_text=f"eps' = {cut_e:.2f}", annotation_position="top right")
    style(fig, 360, bargap=0, xaxis=dict(title=f"Sellers in {'6D Autoencoder' if is_op_6d else '21D'} OPTICS order", showticklabels=False),
          yaxis=dict(title="Reachability distance"))
    show(fig)
    st.caption(f"The axis is cut at the 99.5th percentile ({cap:.1f}) for readability; the largest reachability distance is {fin.max():.1f}.")
    from sklearn.cluster import cluster_optics_dbscan
    cut_lab = cluster_optics_dbscan(reachability=reach, core_distances=core_d, ordering=order, eps=cut_e)
    n_xi = len(set(lab_op)) - (1 if -1 in lab_op else 0)
    n_cut = len(set(cut_lab)) - (1 if -1 in cut_lab else 0)
    m = st.columns(4)
    m[0].metric("Clusters (xi method)", n_xi)
    m[1].metric("Noise (xi method)", f"{(lab_op == -1).sum():,}")
    m[2].metric("Clusters (horizontal cut)", n_cut)
    m[3].metric("Noise (horizontal cut)", f"{(cut_lab == -1).sum():,}", f"{(cut_lab == -1).mean() * 100:.1f}%", delta_color="off")
    callout(
        "One long, shallow valley" if n_xi <= 1 else "Several valleys found",
        ("The plot is one broad valley with spikes for isolated sellers; there are no deep, separated valleys. "
         "Both the 21-D standardized matrix (minPts = 25) and 6-D autoencoder space (minPts = 20) collapse into one cluster, "
         "showing the continuum is not just an artifact of high dimensionality.")
        if n_xi <= 1 else
        "The xi method separated more than one valley. Compare with the horizontal cut: valleys that disappear when the cut moves are not robust.",
    )

    callout(
        "What the five algorithms add up to",
        "K-Means and Ward can always cut a continuum into pieces, but their scores stay low and disagree. DBSCAN and OPTICS see one dense "
        "cloud in both 21D and 6D. Mean-Shift only isolates a handful of extreme sellers. Taken together this supports the choice on page 8: "
        "interpretable, rule-based seller tiers instead of algorithmic clusters.",
    )


def make_subplots_scores(km, wd):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{n} ({b} is better)" for n, _, b in CLUSTER_METRICS], horizontal_spacing=0.08)
    for j, (name, col, _) in enumerate(CLUSTER_METRICS, start=1):
        for algo, df, colr in (("K-Means", km, BLUE), ("Ward", wd, ORANGE)):
            fig.add_scatter(x=df["k"], y=df[col], mode="lines+markers", name=algo, line=dict(color=colr, width=2.5),
                            showlegend=(j == 1), row=1, col=j)
    style(fig, 340, legend=True)
    fig.update_xaxes(dtick=1, title_text="k")
    return fig


# ===========================================================================
# CHALLENGES & DECISIONS
# ===========================================================================
CHALLENGES = {
    "Data processing": [
        ("Joining child tables multiplied rows", "Orders with several items or payments would be duplicated by a plain join.",
         "Aggregated items, payments and reviews to one row per order first, then joined.", "112,650 item rows became 98,666 order rows; 99,441 orders kept one row each."),
        ("Geolocation table was dirty", "1,000,163 rows, 261,831 exact duplicates, and some coordinates in the Atlantic Ocean.",
         "Kept only points inside Brazil's bounding box and took the median point per zip prefix.", "19,010 clean zip centroids used for Haversine distances."),
        ("Misspelled and inconsistent columns", "`product_name_lenght`, a UTF-8 BOM, and Portuguese category names.",
         "Renamed columns, stripped the BOM, merged the English translation table.", "Joins work without silent mismatches; 100% referential integrity checked."),
        ("Noisy daily series", "Weekday/weekend swings hide the trend, but Black Friday is a real spike.",
         "Compared SMA, EWMA and rolling median at 7, 14 and 30 days.", "Short windows for trend; raw series kept for events."),
    ],
    "Statistics & transformations": [
        ("Heavy right skew", "Price skewness 9.7, freight 12.1, approval time 57.",
         "Used medians and rank-based tests; compared log, square root, Box-Cox, Yeo-Johnson.", "Log cut price skewness from 9.7 to 0.3."),
        ("Negative values in days late", "Early deliveries make days late negative, so log and Box-Cox fail.",
         "Used Yeo-Johnson for that variable.", "Skewness 2.0 reduced to -0.8."),
        ("Significance is trivial with 100k rows", "p-values were essentially 0 for every test.",
         "Added effect sizes (Cliff's delta, Cramer's V).", "Delta -0.55 (large), V 0.43 (moderate to strong)."),
        ("Target leakage when encoding", "Target encoding can leak the label into training rows.",
         "Out-of-fold target encoding with 5-fold stratified splits; test categories use training statistics only.", "No row sees its own label."),
    ],
    "Outliers & missing values": [
        ("IQR and Z-score disagree", "IQR flagged 15,807 item weights, Z-score only 2,955.",
         "Treated them as candidates, not errors; compared winsorizing, log and trimming.", "Winsorized data keeps all rows and lowers the spread."),
        ("58.7% of reviews have no text", "Dropping them would bias the review analysis.",
         "Tested whether missingness is related to rating and lateness.", "Comment rate 78% for 1 star vs 36% for 5 stars; chi2 = 1110, p ~ 1e-239. Missing not at random."),
        ("Which imputation to trust", "Weight is in grams, dimensions in cm, so errors are not comparable across features.",
         "Hid 15% of known values and scored mean, median, kNN and MICE within each feature.", "MICE best on RMSE for all four; median best on MAE for weight."),
        ("Categorical column had no gaps", "Mode imputation could not be tested on real missing values.",
         "Masked 15% of known categories and filled them with the mode.", "Demonstrated the method honestly on masked data."),
    ],
    "Text mining": [
        ("Stopword lists delete negations", "Removing *nao*, *nunca*, *sem* turns \"did not receive\" into \"received\".",
         "Custom Portuguese stopword list that keeps nine negation words; added bigrams.", "Complaints such as \"nao recebi\" and \"nao entregue\" surface at the top."),
        ("Frequent words are not distinctive", "Common words rank high in both groups.",
         "Ranked by the TF-IDF difference between 1-star and 5-star reviews.", "Clear complaint vs praise vocabularies."),
    ],
    "Graph analysis": [
        ("Community detection found one group", "The network is dense (density 0.56) and routed through SP.",
         "Used PageRank, betweenness and route-level metrics instead.", "SP is the hub (61% of inter-state outflow)."),
        ("Rare routes look extreme by chance", "Routes with a handful of orders can show 100% late.",
         "Only ranked routes with at least 100 orders (adjustable).", "70 of 409 routes qualify."),
    ],
    "Dimensionality reduction & clustering": [
        ("Missing values and scale in seller features", "118-125 sellers lack timing features; counts and revenue are skewed.",
         "Median imputation, log1p for 10 skewed features, standardization.", "No NaNs; features comparable."),
        ("Choosing t-SNE perplexity", "Plots look different for every setting.",
         "Scored perplexity 5 to 100 with trustworthiness.", "Best local score at perplexity 15 (0.98 vs 0.81 for PCA)."),
        ("Is non-linear worth it?", "PCA is simpler and interpretable.",
         "Trained an autoencoder and compared test MSE with PCA at equal size.", "16-38% lower error, so the non-linear map is used for display."),
        ("Density methods found one cluster", "DBSCAN and OPTICS returned a single cluster; Mean-Shift put 96% in one mode.",
         "Treated this as a finding (a continuum) and did not force clusters; K-Means k=3 used as evidence, business-rule tiers as the output.", "Four tiers and a 0-100 reliability score."),
    ],
}


def page_challenges():
    page_intro(
        "Challenges & decisions",
        "How problems were handled at each step of the analysis",
        "Real data rarely behaves. This page lists what went wrong at each stage, what we did about it, and the evidence that the choice worked.",
    )
    for stage, items in CHALLENGES.items():
        st.markdown(f"## {stage}")
        for title, problem, action, result in items:
            with st.expander(title):
                st.markdown(f"**Problem.** {problem}")
                st.markdown(f"**What we did.** {action}")
                st.markdown(f"**Result.** {result}")


# ===========================================================================
# CONCLUSIONS
# ===========================================================================
def page_conclusions():
    """Render a synthesis of the dashboard's verified findings and takeaways."""
    page_intro(
        "Conclusions",
        "What we found across orders, delivery stages, review text, geography, and sellers",
        "Bringing together the empirical results from Sections 1-8: why Olist deliveries arrive late, how delays "
        "shape customer ratings, and what the data supports for operations and seller management.",
    )

    delivered = filt.dropna(subset=["is_late"])
    late = delivered[delivered["is_late"] == 1]
    ontime = delivered[delivered["is_late"] == 0]
    bad_4plus = filt[filt["lateness_bucket"].isin(["4-7 Days Late", ">7 Days Late"])]
    cross = delivered.groupby("is_cross_state")["is_late"].mean()
    shipped_late = late["seller_shipped_late"].mean() if "seller_shipped_late" in late and len(late) else np.nan
    on_time_dispatch = (1 - shipped_late) if pd.notna(shipped_late) else np.nan

    c = st.columns(4)
    c[0].metric("Late deliveries", pct(delivered["is_late"].mean()) if len(delivered) else "n/a")
    c[1].metric(
        "Rating drop when late",
        f"{ontime['review_score'].mean():.2f} -> {late['review_score'].mean():.2f}"
        if len(ontime) and len(late) else "n/a",
        help="Mean review score for on-time vs late delivered orders in the current selection.",
    )
    c[2].metric(
        "Late despite on-time dispatch",
        pct(on_time_dispatch, 0),
        help="Share of late orders handed to the carrier before the seller's shipping deadline.",
    )
    c[3].metric(
        "4+ days late low reviews",
        pct(low_review_rate(bad_4plus), 0),
        help=f"Share of 1-2 star reviews when 4+ days late (overall selection baseline: {pct(low_review_rate(filt), 0)}).",
    )

    st.markdown("## Core findings")
    left, right = st.columns(2)
    with left:
        callout(
            "1. Transit time and flat promises drive lateness, not seller packing",
            f"Only <b>{pct(shipped_late, 0)}</b> of late orders missed the seller dispatch deadline. Carrier transit "
            f"accounts for the gap (median <b>{late['carrier_transit_days'].median():.1f}</b> days vs "
            f"<b>{ontime['carrier_transit_days'].median():.1f}</b> for on-time orders), while Olist's estimated delivery "
            f"window is nearly identical for orders that arrive late (<b>{late['estimated_delivery_days'].mean():.1f}</b> d) "
            f"and on time (<b>{ontime['estimated_delivery_days'].mean():.1f}</b> d)."
            if len(late) and len(ontime) else
            "Select a filter range with both late and on-time delivered orders to compare dispatch and transit timings.",
        )
        callout(
            "2. Delivery delays are the primary driver of 1-2 star reviews",
            f"On-time orders average <b>{ontime['review_score'].mean():.2f}</b> stars vs "
            f"<b>{late['review_score'].mean():.2f}</b> for late orders, and orders 4+ days late receive 1-2 star "
            f"reviews <b>{pct(low_review_rate(bad_4plus), 0)}</b> of the time (baseline: "
            f"<b>{pct(low_review_rate(filt), 0)}</b>). Portuguese TF-IDF with negations preserved confirms that "
            "complaints center on non-delivery (<i>nao recebi</i>, <i>nao entregue</i>, <i>nao chegou</i>), noting that "
            "comments are missing not at random because unhappy buyers write far more often."
            if len(late) and len(ontime) else
            "Portuguese TF-IDF with negations preserved shows complaints center on non-delivery rather than product defects.",
        )
        callout(
            "3. Skewed distributions require rank tests and careful imputation",
            "Order price, freight, weight, and approval time are heavily right-skewed, so means diverge sharply from "
            "medians and log or Yeo-Johnson transforms are needed before parametric modeling. On masked product dimensions, "
            "MICE achieved the lowest RMSE across all four features while median gave lower MAE on heavily skewed weight.",
        )
    with right:
        cross_txt = (
            f"Cross-state orders arrive late <b>{pct(cross.get(1))}</b> of the time vs <b>{pct(cross.get(0))}</b> "
            "within the same state. "
            if len(cross) == 2 else ""
        )
        callout(
            "4. Sao Paulo anchors a dense network with risky long corridors",
            f"{cross_txt}Sao Paulo ships ~61% of inter-state volume and dominates betweenness centrality, so the 27-state "
            "graph forms one connected hub rather than separate communities. The highest late rates appear on long "
            "corridors from SP to the Northeast (e.g., SP to AL at 26% and SP to MA at 21%), while SP to RJ (15.5%) "
            "generates the largest count of late deliveries.",
        )
        seller_txt = ""
        if sellers is not None and len(sellers):
            n_risk = int((sellers["tier"] == "Tier 4 - At-risk").sum())
            n_star = int((sellers["tier"] == "Tier 1 - Star").sum())
            seller_txt = f" Of {len(sellers):,} sellers, <b>{n_star:,}</b> sit in Tier 1 (Star) and <b>{n_risk:,}</b> in Tier 4 (At-risk)."
        callout(
            "5. Sellers form a continuum, making rule-based tiers more useful than clusters",
            "Across K-Means, DBSCAN, Mean-Shift, and OPTICS (in both 21-D standardized and 6-D autoencoder spaces), "
            "density methods collapse sellers into one main cloud and K-Means silhouettes stay low (0.19 at k=3). "
            f"Business-rule tiers paired with a 0-100 reliability score separate high performers from at-risk sellers cleanly.{seller_txt}",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTES = dict(zip(PAGES, [
    page_overview, page_processing, page_delivery, page_statistics, page_outliers, page_text, page_geo,
    page_sellers, page_clustering, page_conclusions, page_challenges,
]))
ROUTES[page]()

st.markdown("---")
st.caption("DS3030 Data Analytics - Olist Brazilian E-Commerce (Kaggle). Built with Streamlit and Plotly.")