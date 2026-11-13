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
    "item_count", "max_installments", "primary_payment_type", "review_score", "has_comment",
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
rules_risk = load_csv("rules_bad_review_risk.csv")
rules_cop = load_csv("rules_category_copurchase.csv")
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
    "7. Association rules",
    "8. Sellers: reduction & clustering",
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
        ["Outliers and missing values", "IQR vs Z-score, winsorizing, MNAR comments, imputation experiment", "4"],
        ["Text mining", "Portuguese TF-IDF that keeps negations", "5"],
        ["Graph analysis", "State shipping network, PageRank, betweenness, route risk", "6"],
        ["Association rules", "FP-Growth / Apriori on order risk profiles and category baskets", "7"],
        ["Dimensionality reduction", "PCA, t-SNE, autoencoder", "8"],
        ["Clustering", "K-Means, Ward, DBSCAN, Mean-Shift, OPTICS, seller tiers", "8"],
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
    c1, c2 = st.columns([3, 2])
    with c1:
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
    with c2:
        header("What the master table adds")
        st.markdown(
            "- `approval_hours`: purchase to payment approval\n"
            "- `seller_handling_days`: approval to carrier pickup\n"
            "- `carrier_transit_days`: pickup to customer\n"
            "- `total_delivery_days`, `days_late` (actual minus estimate)\n"
            "- `distance_km`: Haversine distance seller to customer\n"
            "- `lateness_bucket`, `is_late`, `is_cross_state`"
        )
        st.caption("master_orders: 99,441 rows x 59 columns. master_sellers: 3,095 rows x 25 columns.")

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

    st.markdown("## When do customers buy? (why time is encoded as sine/cosine)")
    st.write(
        "Hour of day, weekday and month are cyclic: 23:00 is next to 00:00 and December is next to January. "
        "Plain integers break that, so the notebook encodes them as sin and cos pairs."
    )
    c1, c2 = st.columns([3, 2])
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    with c1:
        header("Orders by weekday and hour")
        hm = filt.groupby(["purchase_dayofweek", "purchase_hour"]).size().unstack(fill_value=0)
        hm = hm.reindex(index=range(7), columns=range(24), fill_value=0)
        fig = go.Figure(go.Heatmap(z=hm.values, x=list(range(24)), y=dow_names, colorscale="Blues",
                                   colorbar=dict(title="Orders", thickness=12)))
        style(fig, 320, xaxis=dict(title="Hour of day", dtick=2), yaxis=dict(autorange="reversed", showgrid=False))
        show(fig)
    with c2:
        header("Late rate by purchase weekday")
        wd = filt.groupby("purchase_dayofweek")["is_late"].mean().reindex(range(7)) * 100
        fig = go.Figure(go.Bar(x=dow_names, y=wd.values, marker_color=BLUE,
                               text=[f"{v:.1f}%" for v in wd.values], textposition="outside", cliponaxis=False))
        style(fig, 320, yaxis=dict(title="Late rate (%)", rangemode="tozero"))
        show(fig)


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


@st.cache_data(show_spinner=False)
def transform_skews(x_values):
    x = pd.Series(x_values).dropna()
    if len(x) > 40000:
        x = x.sample(40000, random_state=0)
    out = {"Original": skew(x)}
    if x.min() >= 0:
        out["Square root"] = skew(np.sqrt(x))
        out["Log (log1p)"] = skew(np.log1p(x))
    if x.min() > 0:
        out["Box-Cox"] = skew(stats.boxcox(x.values)[0])
    out["Yeo-Johnson"] = skew(stats.yeojohnson(x.values)[0])
    return out


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
        rows.append([NUM_LABELS[c], len(x), x.mean(), x.median(), x.std(), skew(x), float(stats.kurtosis(x, bias=False))])
    desc = pd.DataFrame(rows, columns=["Variable", "Count", "Mean", "Median", "Std dev", "Skewness", "Kurtosis"])
    table(desc.style.format({"Count": "{:,.0f}", "Mean": "{:,.2f}", "Median": "{:,.2f}", "Std dev": "{:,.2f}",
                              "Skewness": "{:.2f}", "Kurtosis": "{:.1f}"}).background_gradient(subset=["Skewness"], cmap="Oranges"))
    callout(
        "Mean and median disagree for money and time",
        "Order price has a mean far above its median and a skewness near 10, because a few expensive orders pull the "
        "average up. Approval time is the extreme case: a median of about 20 minutes but a mean of ten hours. For "
        "these variables we report medians and use rank-based tests.",
    )

    st.markdown("## Does lateness change the review? Three tests and two effect sizes")
    res = run_tests(filt)
    if res is None:
        st.info("Not enough late and on-time orders in the current selection to run the tests.")
    else:
        t = pd.DataFrame([
            ["Mann-Whitney U", "Review scores, late vs not late", f"U = {res['u'].statistic:,.0f}", fmt_p(res["u"].pvalue)],
            ["Kruskal-Wallis", "Review scores across 5 lateness buckets",
             f"H = {res['kw'].statistic:,.0f}" if res["kw"] else "n/a", fmt_p(res["kw"].pvalue) if res["kw"] else "n/a"],
            ["Chi-square", "Lateness bucket vs low review (1-2 stars)",
             f"chi2 = {res['chi'][0]:,.0f}" if res["chi"] else "n/a", fmt_p(res["chi"][1]) if res["chi"] else "n/a"],
        ], columns=["Test", "Question", "Statistic", "p-value"])
        c1, c2 = st.columns([3, 2])
        with c1:
            table(t)
        with c2:
            m1, m2 = st.columns(2)
            m1.metric("Cliff's delta", f"{res['delta']:.2f}", help="-1 to +1. Negative: late orders get lower scores.")
            m2.metric("Cramer's V", f"{res['v']:.2f}", help="0 to 1. Strength of the lateness / low-review association.")
        size = "large" if abs(res["delta"]) >= 0.474 else "medium" if abs(res["delta"]) >= 0.33 else "small"
        callout(
            "Significant and large",
            f"All three tests reject independence. With ~100k orders, p-values are always tiny, so the effect sizes "
            f"tell the real story: Cliff's delta of {res['delta']:.2f} is a <b>{size}</b> effect. The median review is "
            f"<b>{res['med_ok']:.0f}</b> star(s) for on-time orders and <b>{res['med_late']:.0f}</b> for late orders.",
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
    sk = transform_skews(filt[var].dropna().values)
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
    x = filt[var].dropna()
    xs = x.sample(min(len(x), 20000), random_state=0)
    if xs.min() >= 0:
        tr, tr_name = np.log1p(xs), "log1p"
    else:
        tr, tr_name = pd.Series(stats.yeojohnson(xs.values)[0]), "Yeo-Johnson"
    c1, c2 = st.columns(2)
    with c1:
        header(f"{NUM_LABELS[var]} - original")
        f = go.Figure(go.Histogram(x=xs, nbinsx=60, marker_color=RED))
        style(f, 260, xaxis=dict(title=NUM_LABELS[var]), yaxis=dict(title="Orders"))
        show(f)
    with c2:
        header(f"After {tr_name}")
        f = go.Figure(go.Histogram(x=tr, nbinsx=60, marker_color=BLUE))
        style(f, 260, xaxis=dict(title=f"{tr_name} of {NUM_LABELS[var]}"), yaxis=dict(title="Orders"))
        show(f)

    st.markdown("## Encoding categorical and time variables")
    table(pd.DataFrame([
        ["Payment type", "One-hot", "Nominal, no order, only a few levels"],
        ["State, category", "Frequency + target encoding", "Many levels; target encoding uses 5-fold out-of-fold means so a row never sees its own label"],
        ["Review score", "Ordinal (kept as 1-5)", "Natural order already present"],
        ["Hour, weekday, month", "Sine / cosine pairs", "Keeps 23h next to 0h and December next to January"],
    ], columns=["Variable", "Encoding", "Why"]))


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
        "typical value but misses badly on heavy products. The categorical column had no real gaps, so mode "
        "imputation was demonstrated on masked labels only.",
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
        table(sm[["state", "pagerank", "betweenness", "in_strength", "out_strength"]].rename(columns={
            "state": "State", "pagerank": "PageRank", "betweenness": "Betweenness", "in_strength": "Orders in", "out_strength": "Orders out"}).style.format(
            {"PageRank": "{:.3f}", "Betweenness": "{:.3f}", "Orders in": "{:,.0f}", "Orders out": "{:,.0f}"}))
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

    st.markdown("## Worst routes (seller state to customer state)")
    c1, c2 = st.columns([1, 2])
    min_o = c1.slider("Minimum orders per route", 20, 500, 100, 20,
                      help="Rare routes with a handful of orders can look extreme by chance.")
    key = c2.radio("Rank by", ["Late rate", "Lowest review"], horizontal=True)
    rf = routes[routes["order_count"] >= min_o].copy()
    rf["route"] = rf["seller_state"] + " to " + rf["customer_state"]
    if key == "Late rate":
        w = rf.nlargest(12, "late_rate").sort_values("late_rate")
        xv, label = w["late_rate"] * 100, [f"{v * 100:.1f}%" for v in w["late_rate"]]
    else:
        w = rf.nsmallest(12, "avg_review").sort_values("avg_review", ascending=False)
        xv, label = w["avg_review"], [f"{v:.2f}" for v in w["avg_review"]]
    fig = go.Figure(go.Bar(x=xv, y=w["route"], orientation="h", marker_color=RED,
                           text=[f"{t}  ({n:,} orders)" for t, n in zip(label, w["order_count"])], textposition="outside", cliponaxis=False))
    style(fig, 420, xaxis=dict(title="Late rate (%)" if key == "Late rate" else "Average review", range=[0, float(xv.max()) * 1.4]),
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
# 7. ASSOCIATION RULES
# ===========================================================================
def clean_rule(s, width=34):
    def one(p):
        k, _, v = p.partition("=")
        return f"{k.replace('_', ' ').title()}: {v.replace('_', ' ')}" if v else p.replace("_", " ")
    parts = [one(p) for p in s.split(" + ")]
    out, line = [], ""
    for p in parts:
        if line and len(line) + len(p) > width:
            out.append(line)
            line = p
        else:
            line = f"{line} + {p}" if line else p
    out.append(line)
    return "<br>".join(out)


def page_rules():
    page_intro(
        "7. Association rules",
        "Notebook 04, section C - FP-Growth and Apriori",
        "We looked for combinations of order characteristics that go together with a 1-2 star review.",
    )
    st.markdown("## First attempt: which categories are bought together?")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric("Orders with 2+ categories", "786", "0.8% of orders", delta_color="off")
        callout(
            "Too few baskets",
            "97.9% of orders contain a single category, so only 786 baskets can form category rules. The few rules found "
            "are real but cover a tiny share of sales, so we did not stop there.",
        )
    with c2:
        if rules_cop is not None and len(rules_cop):
            cop = rules_cop.sort_values("lift").copy()
            cop["rule"] = (cop["antecedent"] + " -> " + cop["consequent"]).str.replace("_", " ")
            fig = go.Figure(go.Bar(x=cop["lift"], y=cop["rule"], orientation="h", marker_color=BLUE,
                                   text=[f"lift {l:.1f}  (n={n})" for l, n in zip(cop["lift"], cop["n_baskets"])],
                                   textposition="outside", cliponaxis=False))
            style(fig, 330, xaxis=dict(title="Lift", range=[0, cop["lift"].max() * 1.45]), yaxis=dict(showgrid=False),
                  margin=dict(l=10, r=40, t=10, b=10))
            show(fig)

    st.markdown("## Second attempt: risk profiles that lead to a bad review")
    st.write(
        "Each order became a basket of tags: seller state, customer region, same/cross-state route, lateness status, "
        "freight level, distance band, handling speed and category. The target was a 1-2 star review (12.8% of orders). "
        "FP-Growth and Apriori found the same 4,272 itemsets (1.5 s vs 3.1 s), so we kept FP-Growth."
    )
    if rules_risk is None:
        st.info("rules_bad_review_risk.csv was not found.")
        return
    c1, c2 = st.columns([1, 1])
    conf = c1.slider("Minimum confidence", 0.50, 0.90, 0.70, 0.05)
    nshow = c2.slider("Rules to show", 5, 15, 10)
    rk = rules_risk[rules_risk["confidence"] >= conf].sort_values("lift", ascending=False).head(nshow).sort_values("lift")
    if rk.empty:
        st.info("No rules meet this confidence threshold.")
    else:
        header("Top rules by lift", "Lift 6 means a 1-2 star review is six times more likely than average.")
        fig = go.Figure(go.Bar(
            x=rk["lift"], y=[clean_rule(a) for a in rk["antecedent"]], orientation="h", marker_color=RED,
            text=[f"lift {l:.2f}  |  {c * 100:.0f}% conf  |  {n:,} orders" for l, c, n in zip(rk["lift"], rk["confidence"], rk["n_orders"])],
            textposition="outside", cliponaxis=False))
        style(fig, max(380, 58 * len(rk)), xaxis=dict(title="Lift vs 12.8% baseline", range=[0, rk["lift"].max() * 1.7]),
              yaxis=dict(showgrid=False, tickfont=dict(size=11)), margin=dict(l=10, r=40, t=10, b=10))
        show(fig)

    c1, c2 = st.columns(2)
    with c1:
        header("Low-review rate by lateness status", "Computed from the current selection.")
        g = filt.dropna(subset=["lateness_bucket", "review_score"]).groupby("lateness_bucket")["review_score"].apply(lambda s: (s <= 2).mean())
        g = g.reindex(present_buckets)
        base = low_review_rate(filt)
        fig = go.Figure(go.Bar(x=[BUCKET_SHORT[b] for b in g.index], y=g.values * 100, marker_color=[BUCKET_COLORS[b] for b in g.index],
                               text=[f"{v * 100:.0f}%  ({v / base:.1f}x)" for v in g.values], textposition="outside", cliponaxis=False))
        style(fig, 330, yaxis=dict(title="% 1-2 star reviews", range=[0, 100]))
        show(fig)
    with c2:
        header("Rule quality: confidence vs lift", "Bubble size: number of orders.")
        fig = px.scatter(rules_risk, x=rules_risk["confidence"] * 100, y="lift", size="n_orders", size_max=22,
                         color_discrete_sequence=[BLUE], opacity=0.55)
        style(fig, 330, xaxis=dict(title="Confidence (%)"), yaxis=dict(title="Lift"))
        show(fig)

    callout(
        "Delay explains the risk; what is left is handling speed and region",
        "The strongest rule, a cross-state order to the Southeast delayed more than 7 days, gives a bad review 80.6% of "
        "the time (lift 6.3). When we removed every rule containing a delay tag, the best remaining profiles (slow seller "
        "handling combined with Northeast customers or long distance) only reach lift of about 1.6. So fixing delays "
        "removes most of the risk; slow handling and the far Northeast are the next place to look.",
    )
    with st.expander("See all mined risk rules"):
        t = rules_risk.sort_values("lift", ascending=False).head(100).copy()
        t["confidence"] = t["confidence"] * 100
        cols = [c for c in ["antecedent", "n_orders", "support", "confidence", "lift", "conviction"] if c in t]
        table(t[cols].rename(columns={"confidence": "confidence %"}))


# ===========================================================================
# 8. SELLERS: REDUCTION & CLUSTERING
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
    from sklearn.metrics import silhouette_score
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append([k, km.inertia_, silhouette_score(X, km.labels_)])
    return pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])


def nice(c):
    return c.replace("_", " ").replace("avg ", "average ")


def page_sellers():
    page_intro(
        "8. Sellers: dimensionality reduction & clustering",
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

    st.markdown("## Linear vs non-linear reduction")
    c1, c2 = st.columns(2)
    with c1:
        header("t-SNE: neighbourhood preservation", "Trustworthiness of the 2-D map (1 = perfect), by perplexity.")
        perp = [5, 15, 30, 50, 100]
        fig = go.Figure()
        fig.add_scatter(x=perp, y=[0.9708, 0.9798, 0.9794, 0.9772, 0.9754], mode="lines+markers", name="k = 10 neighbours", line=dict(color=BLUE, width=2.5))
        fig.add_scatter(x=perp, y=[0.9295, 0.9476, 0.9524, 0.9530, 0.9553], mode="lines+markers", name="k = 30 neighbours", line=dict(color=GREEN, width=2.5))
        fig.add_hline(y=0.8058, line_dash="dot", line_color=RED, annotation_text="2-D PCA baseline (0.81)", annotation_position="bottom right")
        style(fig, 330, legend=True, xaxis=dict(title="Perplexity", type="category"), yaxis=dict(title="Trustworthiness", range=[0.78, 1.0]))
        show(fig)
    with c2:
        header("Autoencoder vs PCA: reconstruction error", "Test-set MSE by bottleneck size (lower is better).")
        ks = ["2", "3", "6", "9"]
        fig = go.Figure()
        fig.add_bar(x=ks, y=[0.5255, 0.4052, 0.1808, 0.0816], name="PCA", marker_color="#b9c6d2")
        fig.add_bar(x=ks, y=[0.4044, 0.3001, 0.1116, 0.0684], name="Autoencoder", marker_color=BLUE,
                    text=["-23%", "-26%", "-38%", "-16%"], textposition="outside")
        style(fig, 330, legend=True, barmode="group", xaxis=dict(title="Latent dimensions"), yaxis=dict(title="Test MSE"))
        show(fig)
    st.caption("t-SNE and autoencoder results are from Notebook 05 (these are slow to retrain, so the dashboard reads the saved numbers).")
    callout(
        "Non-linear methods keep more structure",
        "A 2-D PCA preserves 81% of local neighbourhoods; t-SNE preserves about 98%. The autoencoder reconstructs the "
        "held-out sellers 16-38% better than PCA at the same size, with its biggest gain at 6 latent dimensions. "
        "Perplexity 15 gave the best local score, so that map is used below.",
    )

    header("Seller map", "Each dot is a seller.")
    c1, c2 = st.columns([1, 1])
    methods = {"PCA": ("pca_dim1", "pca_dim2"), "t-SNE": ("tsne_dim1", "tsne_dim2"), "Autoencoder": ("ae_dim1", "ae_dim2")}
    methods = {k: v for k, v in methods.items() if all(c in sellers.columns for c in v)}
    emb = c1.radio("Embedding", list(methods), horizontal=True)
    by = c2.radio("Colour by", ["Seller tier", "K-Means cluster"], horizontal=True)
    sc = sellers.copy()
    sc["K-Means cluster"] = "Cluster " + sc["cluster_kmeans"].astype(str) if "cluster_kmeans" in sc else "n/a"
    xcol, ycol = methods[emb]
    col = "tier" if by == "Seller tier" else "K-Means cluster"
    fig = px.scatter(sc, x=xcol, y=ycol, color=col, opacity=0.6, color_discrete_map=TIER_COLORS if col == "tier" else None,
                     color_discrete_sequence=[BLUE, ORANGE, GREEN, RED, GREY, "#7a5fa0"],
                     hover_data={"total_orders": True, "late_rate": ":.2f", "avg_review_score": ":.2f", xcol: False, ycol: False},
                     category_orders={"tier": list(TIER_COLORS)})
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

    header("Five algorithms compared", "Results from Notebook 05 on the same standardized 21-feature matrix.")
    table(pd.DataFrame([
        ["K-Means (k=3)", 3, "0.190", "5% / 40% / 55% of sellers", "Best silhouette among k-means runs"],
        ["Ward hierarchical (k=3)", 3, "0.157", "51% / 40% / 9%", "Similar split, lower silhouette"],
        ["DBSCAN (eps 4.37, minPts 42)", 1, "n/a", "1 cluster, 3.2% noise", "Same single cluster for every eps tried"],
        ["Mean-Shift (q = 0.25)", 7, "0.333", "96% in one mode", "High silhouette is misleading: 6 tiny outlier groups"],
        ["OPTICS (xi 0.04)", 1, "n/a", "1 cluster, 0% noise", "No density valleys, also in the 6-D latent space"],
    ], columns=["Algorithm", "Clusters", "Silhouette", "Sizes", "Reading"]))
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
    "Graph & association rules": [
        ("Community detection found one group", "The network is dense (density 0.56) and routed through SP.",
         "Used PageRank, betweenness and route-level metrics instead.", "SP is the hub (61% of inter-state outflow)."),
        ("Rare routes look extreme by chance", "Routes with a handful of orders can show 100% late.",
         "Only ranked routes with at least 100 orders (adjustable).", "70 of 409 routes qualify."),
        ("Almost no multi-category baskets", "97.9% of orders have one category; only 786 baskets.",
         "Changed the question: tagged each order with route, delay, freight, distance, handling and mined rules for a bad review.", "95,830 transactions x 55 items, 4,272 itemsets."),
        ("Delay tags dominate every rule", "The obvious cause hides everything else.",
         "Re-ran the ranking excluding rules with a delay tag.", "Slow handling plus Northeast reaches lift about 1.6 even without delay."),
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


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ROUTES = {
    PAGES[0]: page_overview, PAGES[1]: page_processing, PAGES[2]: page_delivery, PAGES[3]: page_statistics,
    PAGES[4]: page_outliers, PAGES[5]: page_text, PAGES[6]: page_geo, PAGES[7]: page_rules,
    PAGES[8]: page_sellers, PAGES[9]: page_challenges,
}
ROUTES[page]()

st.markdown("---")
st.caption("DS3030 Data Analytics - Olist Brazilian E-Commerce (Kaggle). Built with Streamlit and Plotly.")
