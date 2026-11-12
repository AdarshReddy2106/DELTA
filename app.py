"""
DELTA — Delivery Estimate, Lateness & Trust Analytics
Streamlit Dashboard  •  DS3030 Data Analytics Final Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 0. PAGE CONFIG & GLOBAL STYLE
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DELTA Analytics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Main background & text */
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #232a3b 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #8b949e !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background-color: #161b22; border-radius: 10px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 20px; font-weight: 600;
        color: #8b949e; background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #238636 0%, transparent 100%);
        padding: 4px 16px; border-radius: 6px; margin-bottom: 12px;
        font-size: 1.1rem; font-weight: 600; color: #ffffff;
    }

    /* Hide default hamburger & footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* divider */
    hr { border: 1px solid #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING (cached)
# ──────────────────────────────────────────────────────────────
DATA = Path("PROCESSED_DATA")

@st.cache_data(show_spinner="Loading order data …")
def load_orders():
    df = pd.read_csv(DATA / "master_orders.csv", parse_dates=[
        "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date", "purchase_date",
    ])
    return df

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

# Load all data
orders = load_orders()
sellers = load_sellers()
ts = load_timeseries()
routes = load_routes()
state_metrics = load_state_metrics()
nlp_top, nlp_dist = load_nlp()
rules_risk, rules_copurchase = load_rules()

# ──────────────────────────────────────────────────────────────
# 2. SIDEBAR — GLOBAL FILTERS
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 DELTA")
    st.markdown("**Delivery Estimate, Lateness & Trust Analytics**")
    st.markdown("---")

    st.markdown("### 🔍 Global Filters")

    # Year filter
    years = sorted(orders["purchase_year"].dropna().unique().astype(int))
    sel_years = st.multiselect("Purchase Year", years, default=years, key="year_filter")

    # State filter
    all_states = sorted(orders["customer_state"].dropna().unique())
    sel_states = st.multiselect("Customer State", all_states, default=all_states, key="state_filter")

    # Category filter (top 20 + all)
    top_cats = orders["primary_category"].value_counts().head(20).index.tolist()
    sel_cats = st.multiselect(
        "Product Category (top 20 shown)",
        top_cats, default=top_cats, key="cat_filter"
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#8b949e;font-size:0.75rem;'>"
        "DS3030 • Data Analytics<br>Brazilian E-Commerce Dataset (Olist)"
        "</div>",
        unsafe_allow_html=True,
    )

# Apply global filters
mask = (
    orders["purchase_year"].isin(sel_years)
    & orders["customer_state"].isin(sel_states)
    & orders["primary_category"].isin(sel_cats)
)
filt = orders[mask].copy()

# ──────────────────────────────────────────────────────────────
# 3. HEADER & KPI CARDS
# ──────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; background: linear-gradient(90deg, #58a6ff, #3fb950, #f0883e); "
    "-webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.4rem;'>"
    "📦 DELTA — Delivery Estimate, Lateness & Trust Analytics</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#8b949e; margin-top:-8px;'>"
    "Interactive exploration of ~100 000 Brazilian e-commerce orders  •  DS3030 Final Dashboard</p>",
    unsafe_allow_html=True,
)
st.markdown("")

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Total Orders", f"{len(filt):,}")
with k2:
    rev = filt["total_price"].sum()
    st.metric("Total Revenue", f"R$ {rev:,.0f}")
with k3:
    late_pct = filt["is_late"].mean() * 100 if len(filt) else 0
    st.metric("Late Delivery %", f"{late_pct:.1f}%")
with k4:
    avg_rev = filt["review_score"].mean() if len(filt) else 0
    st.metric("Avg Review Score", f"{avg_rev:.2f} ★")
with k5:
    avg_del = filt["total_delivery_days"].mean() if len(filt) else 0
    st.metric("Avg Delivery Days", f"{avg_del:.1f}")
with k6:
    avg_dist = filt["distance_km"].mean() if len(filt) else 0
    st.metric("Avg Distance", f"{avg_dist:.0f} km")

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# 4. TABBED SECTIONS — 7 ANALYTICAL FUNCTIONALITIES
# ──────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Delivery Pipeline",
    "⏱️ Time-Series Trends",
    "🗺️ Geographic Intelligence",
    "🛡️ Outliers & Imputation",
    "⭐ Review & Sentiment Analysis",
    "🏪 Seller Segmentation",
    "⚠️ Risk Pattern Mining",
    "🔎 Deep Dive Explorer",
])

# ——— Plotly theme helper ———
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)

GREEN  = "#3fb950"
BLUE   = "#58a6ff"
ORANGE = "#f0883e"
RED    = "#f85149"
PURPLE = "#bc8cff"
YELLOW = "#d29922"
COLORS = [GREEN, BLUE, ORANGE, RED, PURPLE, YELLOW, "#79c0ff", "#a5d6ff"]

# ══════════════════════════════════════════════════════════════
# TAB 1 — DELIVERY PIPELINE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">🔬 Where Do Delivery Delays Actually Come From?</div>', unsafe_allow_html=True)
    st.caption("Decomposes the end-to-end delivery process into 3 stages: **approval → seller handling → carrier transit**. Identifies which stage causes the most delay.")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        # Compute median hours for each stage
        stage_data = filt[["approval_hours", "seller_handling_days", "carrier_transit_days"]].dropna()
        if len(stage_data):
            medians = {
                "Payment Approval (hrs)": stage_data["approval_hours"].median(),
                "Seller Handling (days)": stage_data["seller_handling_days"].median(),
                "Carrier Transit (days)": stage_data["carrier_transit_days"].median(),
            }
            fig_pipe = go.Figure()
            fig_pipe.add_trace(go.Bar(
                x=list(medians.values()),
                y=list(medians.keys()),
                orientation="h",
                marker=dict(color=[BLUE, ORANGE, GREEN], line=dict(width=0)),
                text=[f"{v:.1f}" for v in medians.values()],
                textposition="auto",
            ))
            fig_pipe.update_layout(
                title="Median Time per Delivery Stage",
                xaxis_title="Median Duration",
                yaxis_title="",
                height=320,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_pipe, width="stretch")

    with col_b:
        # Lateness bucket distribution
        bucket_order = ["Early (>1d early)", "On Time (-1d to 0d)", "1-3 Days Late", "4-7 Days Late", ">7 Days Late"]
        bucket_counts = filt["lateness_bucket"].value_counts().reindex(bucket_order).dropna()
        if len(bucket_counts):
            fig_bucket = go.Figure(go.Pie(
                labels=bucket_counts.index,
                values=bucket_counts.values,
                hole=0.55,
                marker=dict(colors=[GREEN, BLUE, YELLOW, ORANGE, RED]),
                textinfo="percent+label",
                textfont=dict(size=11),
            ))
            fig_bucket.update_layout(
                title="Order Lateness Distribution",
                height=320,
                showlegend=False,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_bucket, width="stretch")

    # Delay breakdown by lateness bucket (stacked bar)
    st.markdown("#### Stage-wise Duration by Lateness Bucket")
    stage_by_bucket = (
        filt.dropna(subset=["lateness_bucket", "approval_hours", "seller_handling_days", "carrier_transit_days"])
        .groupby("lateness_bucket")[["approval_hours", "seller_handling_days", "carrier_transit_days"]]
        .median()
        .reindex(bucket_order)
        .dropna()
    )
    if len(stage_by_bucket):
        # Convert approval hours to days for consistent stacking
        stage_by_bucket["approval_days"] = stage_by_bucket["approval_hours"] / 24
        fig_stack = go.Figure()
        for col, name, color in [
            ("approval_days", "Approval", BLUE),
            ("seller_handling_days", "Seller Handling", ORANGE),
            ("carrier_transit_days", "Carrier Transit", GREEN),
        ]:
            fig_stack.add_trace(go.Bar(
                x=stage_by_bucket.index, y=stage_by_bucket[col],
                name=name, marker_color=color,
            ))
        fig_stack.update_layout(
            barmode="stack",
            yaxis_title="Median Days",
            height=360,
            legend=dict(orientation="h", y=1.12),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_stack, width="stretch")

    # Key insight
    with st.expander("💡 Key Insight", expanded=True):
        st.info(
            "**Carrier transit** is the dominant contributor to late deliveries. "
            "For orders arriving >7 days late, carrier transit accounts for the majority of the total delay, "
            "while seller handling and payment approval remain relatively stable across lateness buckets. "
            "This suggests that improving logistics partnerships and route planning has the highest impact on reducing delays."
        )

# ══════════════════════════════════════════════════════════════
# TAB 2 — TIME-SERIES TRENDS
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">📊 Business Trends Over Time (Smoothing & Seasonality)</div>', unsafe_allow_html=True)
    st.caption("Daily order volumes smoothed with SMA, EWMA, and Median filters to reveal seasonality patterns while filtering weekday noise.")

    col_metric, col_window = st.columns(2)
    with col_metric:
        ts_metric = st.selectbox("Metric", ["raw_orders", "raw_revenue", "late_rate", "avg_review"], index=0)
    with col_window:
        ts_window = st.radio("Smoothing Window", ["7d", "14d", "30d"], horizontal=True)

    metric_labels = {
        "raw_orders": "Daily Orders",
        "raw_revenue": "Daily Revenue (R$)",
        "late_rate": "Daily Late Rate",
        "avg_review": "Daily Avg Review Score",
    }

    fig_ts = go.Figure()
    # Raw data
    fig_ts.add_trace(go.Scatter(
        x=ts["purchase_date"], y=ts[ts_metric],
        mode="lines", name="Raw",
        line=dict(color="#30363d", width=1),
        opacity=0.5,
    ))
    # SMA
    sma_col = f"sma_{ts_window}"
    if sma_col in ts.columns:
        fig_ts.add_trace(go.Scatter(
            x=ts["purchase_date"], y=ts[sma_col],
            mode="lines", name=f"SMA {ts_window}",
            line=dict(color=BLUE, width=2),
        ))
    # EWMA
    ewma_col = f"ewma_{ts_window}"
    if ewma_col in ts.columns:
        fig_ts.add_trace(go.Scatter(
            x=ts["purchase_date"], y=ts[ewma_col],
            mode="lines", name=f"EWMA {ts_window}",
            line=dict(color=GREEN, width=2),
        ))
    # Median
    med_col = f"median_{ts_window}"
    if med_col in ts.columns:
        fig_ts.add_trace(go.Scatter(
            x=ts["purchase_date"], y=ts[med_col],
            mode="lines", name=f"Median {ts_window}",
            line=dict(color=ORANGE, width=2),
        ))

    fig_ts.update_layout(
        title=f"{metric_labels[ts_metric]} — Raw vs Smoothed ({ts_window} window)",
        yaxis_title=metric_labels[ts_metric],
        height=420,
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
        **PLOTLY_LAYOUT,
    )
    # Annotate Black Friday 2017
    fig_ts.add_vline(x="2017-11-24", line_dash="dot", line_color=YELLOW)
    fig_ts.add_annotation(x="2017-11-24", y=1, yref="paper", text="Black Friday 2017",
                          showarrow=False, font=dict(color=YELLOW, size=11), yshift=10)
    st.plotly_chart(fig_ts, width="stretch")

    # Monthly aggregation
    st.markdown("#### Monthly Order Volume & Revenue Heatmap")
    monthly = filt.groupby([filt["purchase_date"].dt.year.rename("year"), filt["purchase_date"].dt.month.rename("month")]).agg(
        order_count=("order_id", "count"),
        revenue=("total_price", "sum"),
        late_pct=("is_late", "mean"),
    ).reset_index()

    heatmap_metric = st.radio("Heatmap Metric", ["order_count", "revenue", "late_pct"], horizontal=True,
                               format_func=lambda x: {"order_count": "Orders", "revenue": "Revenue", "late_pct": "Late %"}[x])

    if len(monthly):
        pivot = monthly.pivot(index="year", columns="month", values=heatmap_metric).fillna(0)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        color_scale = "RdYlGn_r" if heatmap_metric == "late_pct" else "Greens"
        fig_heat = px.imshow(
            pivot.values,
            labels=dict(x="Month", y="Year", color=heatmap_metric),
            x=pivot.columns.tolist(),
            y=[str(y) for y in pivot.index.tolist()],
            color_continuous_scale=color_scale,
            aspect="auto",
        )
        fig_heat.update_layout(
            title=f"Monthly {heatmap_metric.replace('_', ' ').title()} Heatmap",
            height=280,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_heat, width="stretch")

    with st.expander("💡 Key Insight", expanded=True):
        st.info(
            "Order volume shows strong growth through 2017 into early 2018, with a massive **Black Friday 2017** spike. "
            "The 30-day moving average best smooths weekday/weekend noise while preserving real seasonal peaks. "
            "Late delivery rates tend to increase following demand surges, suggesting logistics capacity cannot scale instantly with order spikes."
        )

# ══════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC INTELLIGENCE
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">🌎 Shipping Routes & Geographic Delay Patterns</div>', unsafe_allow_html=True)
    st.caption("Network graph analysis of Brazil's 27-state shipping routes — which corridors generate the most delays?")

    col_map1, col_map2 = st.columns(2)

    with col_map1:
        # State-level bar chart — orders + late rate
        state_agg = (
            filt.groupby("customer_state")
            .agg(orders=("order_id", "count"), late_pct=("is_late", "mean"), avg_review=("review_score", "mean"))
            .reset_index()
            .sort_values("orders", ascending=False)
            .head(15)
        )
        fig_state = make_subplots(specs=[[{"secondary_y": True}]])
        fig_state.add_trace(
            go.Bar(x=state_agg["customer_state"], y=state_agg["orders"],
                   name="Orders", marker_color=BLUE, opacity=0.8),
            secondary_y=False,
        )
        fig_state.add_trace(
            go.Scatter(x=state_agg["customer_state"], y=state_agg["late_pct"] * 100,
                       name="Late %", mode="lines+markers",
                       line=dict(color=RED, width=2), marker=dict(size=7)),
            secondary_y=True,
        )
        fig_state.update_layout(
            title="Top 15 Customer States: Volume vs Late %",
            height=380,
            legend=dict(orientation="h", y=1.12),
            **PLOTLY_LAYOUT,
        )
        fig_state.update_yaxes(title_text="Order Count", secondary_y=False)
        fig_state.update_yaxes(title_text="Late %", secondary_y=True)
        st.plotly_chart(fig_state, width="stretch")

    with col_map2:
        # Distance vs Delivery Time scatter (sampled)
        sample = filt.dropna(subset=["distance_km", "total_delivery_days", "review_score"]).sample(
            n=min(3000, len(filt)), random_state=42
        )
        fig_dist = px.scatter(
            sample, x="distance_km", y="total_delivery_days",
            color="review_score",
            color_continuous_scale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
            opacity=0.5,
            labels={"distance_km": "Distance (km)", "total_delivery_days": "Delivery Days", "review_score": "Review"},
        )
        fig_dist.update_layout(
            title="Distance vs Delivery Time (colored by Review Score)",
            height=380,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_dist, width="stretch")

    # Route heatmap — worst routes
    st.markdown("#### 🔥 Worst Shipping Routes (Seller State → Customer State)")
    min_orders = st.slider("Min orders per route", 10, 500, 50, step=10)
    routes_filt = routes[routes["order_count"] >= min_orders].copy()

    sort_metric = st.radio("Sort routes by", ["late_rate", "avg_days_late", "avg_review"], horizontal=True,
                            format_func=lambda x: {"late_rate": "Late Rate", "avg_days_late": "Avg Days Late", "avg_review": "Avg Review (asc)"}[x])
    ascending = True if sort_metric == "avg_review" else False
    worst = routes_filt.sort_values(sort_metric, ascending=not ascending).head(20)

    if len(worst):
        worst["route"] = worst["seller_state"] + " → " + worst["customer_state"]
        fig_worst = px.bar(
            worst, x=sort_metric, y="route", orientation="h",
            color=sort_metric,
            color_continuous_scale="RdYlGn_r" if sort_metric != "avg_review" else "RdYlGn",
            text=worst[sort_metric].round(2),
            labels={sort_metric: sort_metric.replace("_", " ").title()},
        )
        fig_worst.update_layout(
            title=f"Top 20 Worst Routes by {sort_metric.replace('_', ' ').title()}",
            height=520, yaxis=dict(autorange="reversed"),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_worst, width="stretch")

    # Cross-state vs intra-state
    st.markdown("#### Intra-State vs Cross-State Performance")
    c1, c2 = st.columns(2)
    with c1:
        cross = filt.groupby("is_cross_state").agg(
            avg_delivery=("total_delivery_days", "mean"),
            avg_late=("days_late", "mean"),
            avg_review=("review_score", "mean"),
            count=("order_id", "count"),
        ).reset_index()
        cross["type"] = cross["is_cross_state"].map({0: "Same State", 1: "Cross State"})
        fig_cross = px.bar(
            cross, x="type", y=["avg_delivery", "avg_late", "avg_review"],
            barmode="group",
            color_discrete_sequence=[BLUE, ORANGE, GREEN],
        )
        fig_cross.update_layout(title="Delivery Metrics: Same vs Cross State", height=320, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_cross, width="stretch")
    with c2:
        with st.expander("💡 Key Insight", expanded=True):
            st.info(
                "**Cross-state shipments** take significantly longer and generate more late deliveries. "
                "Routes originating from São Paulo (SP) to northern states like AM, RR, AP have the highest delay rates, "
                "reflecting Brazil's vast geographic distances and limited logistics infrastructure in remote regions. "
                "Intra-state orders enjoy ~30% shorter delivery times and higher review scores on average."
            )


# ══════════════════════════════════════════════════════════════
# TAB 4 — OUTLIERS & IMPUTATION (Lectures 6, 7, 8)
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">🛡️ Outlier Detection, Sensitivity Analysis & Imputation Benchmarks</div>', unsafe_allow_html=True)
    st.caption("Investigates extreme anomalies, tests distributional sensitivity under Winsorization/Log transforms, and evaluates missing-value imputation models (Lectures 6-8).")

    subtab1, subtab2, subtab3 = st.tabs(["🎯 Outlier Detection", "⚖️ Sensitivity & Transformations", "🧪 Missing Data & Imputation Experiment"])

    with subtab1:
        st.markdown("#### 🔍 Interactive Univariate Outlier Detection (IQR vs. Z-Score)")
        col_o1, col_o2, col_o3 = st.columns([2, 1, 1])
        with col_o1:
            outlier_var = st.selectbox(
                "Select Variable to Analyze",
                ["total_price", "total_freight", "total_weight_g", "total_delivery_days", "days_late", "distance_km", "approval_hours"],
                index=0,
                format_func=lambda x: {
                    "total_price": "Total Price (R$)",
                    "total_freight": "Total Freight Value (R$)",
                    "total_weight_g": "Product Weight (grams)",
                    "total_delivery_days": "Delivery Time (days)",
                    "days_late": "Days Late (days)",
                    "distance_km": "Customer-Seller Distance (km)",
                    "approval_hours": "Approval Time (hours)",
                }.get(x, x),
                key="outlier_var_select"
            )
        with col_o2:
            iqr_mult = st.slider("IQR Multiplier (k)", 1.0, 3.0, 1.5, 0.25, key="iqr_mult_slider")
        with col_o3:
            z_thresh = st.slider("Z-Score Cutoff (|z|)", 2.0, 4.0, 3.0, 0.25, key="z_thresh_slider")

        # Compute outliers dynamically on filtered dataset
        series = filt[outlier_var].dropna()
        if len(series) > 0:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr_val = q3 - q1
            iqr_lower = q1 - iqr_mult * iqr_val
            iqr_upper = q3 + iqr_mult * iqr_val
            iqr_mask = (series < iqr_lower) | (series > iqr_upper)
            iqr_outliers = series[iqr_mask]

            mean_val = float(series.mean())
            std_val = float(series.std()) if float(series.std()) > 0 else 1.0
            z_scores = (series - mean_val) / std_val
            z_mask = z_scores.abs() > z_thresh
            z_outliers = series[z_mask]

            overlap_mask = iqr_mask & z_mask
            overlap_count = int(overlap_mask.sum())

            # Metric KPIs
            mo1, mo2, mo3, mo4 = st.columns(4)
            with mo1:
                st.metric("Filtered Sample Size", f"{len(series):,}")
            with mo2:
                st.metric(f"IQR Outliers (k={iqr_mult})", f"{len(iqr_outliers):,} ({len(iqr_outliers)/len(series)*100:.1f}%)")
            with mo3:
                st.metric(f"Z-Score Outliers (|z|>{z_thresh})", f"{len(z_outliers):,} ({len(z_outliers)/len(series)*100:.1f}%)")
            with mo4:
                st.metric("Methods Agreement (Overlap)", f"{overlap_count:,} ({overlap_count/len(series)*100:.1f}%)")

            # Visualizations: Histogram + Box Plot
            vh1, vh2 = st.columns(2)
            with vh1:
                plot_sample = series.sample(min(5000, len(series)), random_state=42)
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=plot_sample,
                    nbinsx=50,
                    marker_color=BLUE,
                    opacity=0.75,
                    name="Distribution",
                ))
                fig_hist.add_vline(x=iqr_upper, line_dash="dash", line_color=RED)
                fig_hist.add_vline(x=iqr_lower, line_dash="dash", line_color=RED)
                fig_hist.add_annotation(x=iqr_upper, y=1, yref="paper", text=f"IQR Upper ({iqr_upper:.1f})", showarrow=False, font=dict(color=RED, size=10), yshift=10)
                if iqr_lower > plot_sample.min():
                    fig_hist.add_annotation(x=iqr_lower, y=1, yref="paper", text=f"IQR Lower ({iqr_lower:.1f})", showarrow=False, font=dict(color=RED, size=10), yshift=10)
                
                z_upper = mean_val + z_thresh * std_val
                fig_hist.add_vline(x=z_upper, line_dash="dot", line_color=YELLOW)
                fig_hist.add_annotation(x=z_upper, y=0.85, yref="paper", text=f"Z-Score Cutoff ({z_upper:.1f})", showarrow=False, font=dict(color=YELLOW, size=10))

                fig_hist.update_layout(
                    title=f"Distribution & Fences: {outlier_var.replace('_', ' ').title()}",
                    xaxis_title=outlier_var.replace("_", " ").title(),
                    yaxis_title="Count",
                    height=350,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_hist, width="stretch")

            with vh2:
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=plot_sample,
                    name=outlier_var.replace("_", " ").title(),
                    boxpoints="outliers",
                    marker_color=ORANGE,
                    line_color=GREEN,
                ))
                fig_box.update_layout(
                    title="Box Plot (Outlier Points Beyond Whiskers)",
                    height=350,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_box, width="stretch")

        st.markdown("#### 📋 Cross-Feature Outlier Detection Comparison (Full Dataset Benchmark)")
        outlier_summary_data = pd.DataFrame([
            {"Feature": "Total Price (R$)", "IQR Count": 7913, "IQR %": "8.02%", "Z-Score Count": 1709, "Z-Score %": "1.73%", "Detection Agreement": "Z-score is stricter; flags only extreme luxury purchases"},
            {"Feature": "Total Freight Value (R$)", "IQR Count": 9941, "IQR %": "10.08%", "Z-Score Count": 1581, "Z-Score %": "1.60%", "Detection Agreement": "Cross-state Amazonian routes flagged as heavy outliers"},
            {"Feature": "Product Weight (grams)", "IQR Count": 13804, "IQR %": "13.99%", "Z-Score Count": 1962, "Z-Score %": "1.99%", "Detection Agreement": "Heavy industrial/furniture goods create massive right skew"},
            {"Feature": "Total Delivery Days", "IQR Count": 4899, "IQR %": "5.08%", "Z-Score Count": 1571, "Z-Score %": "1.63%", "Detection Agreement": "Severe logistics disruptions (weather, strikes) flagged"},
            {"Feature": "Days Late (Delivery Delay)", "IQR Count": 4875, "IQR %": "5.05%", "Z-Score Count": 1333, "Z-Score %": "1.38%", "Detection Agreement": "Orders delayed past 2-3 weeks"},
            {"Feature": "Distance (km)", "IQR Count": 7622, "IQR %": "7.76%", "Z-Score Count": 1564, "Z-Score %": "1.59%", "Detection Agreement": "Transcontinental shipments (e.g., RS to RR ~3500+ km)"},
        ])
        st.dataframe(outlier_summary_data, width="stretch", hide_index=True)

    with subtab2:
        st.markdown("#### ⚖️ Sensitivity Analysis: Impact of Outlier Treatments")
        st.caption("Compares distribution statistics under 4 regimes: **Original Raw Data**, **Winsorization (1st & 99th percentiles)**, **Log Transformation (log1p)**, and **IQR Outlier Truncation**.")

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

        sel_sens_var = st.selectbox("Filter Variable for Treatment Comparison", sens_df["Variable"].unique(), index=0)
        sub_sens = sens_df[sens_df["Variable"] == sel_sens_var]

        col_st1, col_st2 = st.columns([3, 2])
        with col_st1:
            st.dataframe(
                sub_sens.style.background_gradient(subset=["Skewness"], cmap="RdYlGn_r"),
                width="stretch",
                hide_index=True
            )
        with col_st2:
            fig_skew = px.bar(
                sub_sens,
                x="Treatment", y="Skewness",
                color="Treatment",
                color_discrete_sequence=[RED, YELLOW, GREEN, BLUE],
                title=f"Skewness Reduction: {sel_sens_var}",
            )
            fig_skew.update_layout(height=280, showlegend=False, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_skew, width="stretch")

        st.markdown("##### 🧪 Visual Distribution Comparison (Live Filtered Data)")
        sample_sens = filt[filt["total_price"] > 0]["total_price"].sample(min(3000, len(filt)), random_state=42)
        raw_vals = sample_sens
        p1, p99 = sample_sens.quantile(0.01), sample_sens.quantile(0.99)
        win_vals = sample_sens.clip(p1, p99)
        log_vals = np.log1p(sample_sens)

        fig_treat_dist = make_subplots(rows=1, cols=3, subplot_titles=["Raw Distribution (Heavy Tail)", "Winsorized (1-99% Capped)", "Log1p Transformed (Gaussian-like)"])
        fig_treat_dist.add_trace(go.Histogram(x=raw_vals, nbinsx=40, marker_color=RED, name="Raw"), row=1, col=1)
        fig_treat_dist.add_trace(go.Histogram(x=win_vals, nbinsx=40, marker_color=YELLOW, name="Winsorized"), row=1, col=2)
        fig_treat_dist.add_trace(go.Histogram(x=log_vals, nbinsx=40, marker_color=GREEN, name="Log1p"), row=1, col=3)
        fig_treat_dist.update_layout(height=320, showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_treat_dist, width="stretch")

    with subtab3:
        st.markdown("#### 🧪 Missing Data Behavior & Masking Imputation Experiment")
        st.caption("Analyzes the mechanism of missing review comments (Informative Missingness / MNAR) and reviews the 15% masking imputation experiment results across ML techniques.")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            comm_by_score = filt.groupby("review_score")["has_comment"].mean().reset_index()
            comm_by_score["review_score"] = comm_by_score["review_score"].astype(str) + " ★"
            comm_by_score["comment_rate"] = (comm_by_score["has_comment"] * 100).round(1)

            fig_comm_score = px.bar(
                comm_by_score,
                x="review_score", y="comment_rate",
                color="comment_rate",
                color_continuous_scale="RdYlGn_r",
                text="comment_rate",
                title="Propensity to Write Review Text by Star Rating",
                labels={"review_score": "Review Score", "comment_rate": "% Customers Leaving Written Text"}
            )
            fig_comm_score.update_layout(height=340, yaxis_range=[0, 100], **PLOTLY_LAYOUT)
            st.plotly_chart(fig_comm_score, width="stretch")

        with col_m2:
            order_buckets = ["Early (>1d early)", "On Time (-1d to 0d)", "1-3 Days Late", "4-7 Days Late", ">7 Days Late"]
            comm_by_late = filt.groupby("lateness_bucket")["has_comment"].mean().reindex(order_buckets).dropna().reset_index()
            comm_by_late["comment_rate"] = (comm_by_late["has_comment"] * 100).round(1)

            fig_comm_late = px.bar(
                comm_by_late,
                x="lateness_bucket", y="comment_rate",
                color="comment_rate",
                color_continuous_scale="RdYlGn_r",
                text="comment_rate",
                title="Propensity to Write Review Text by Delivery Delay",
                labels={"lateness_bucket": "Delivery Lateness", "comment_rate": "% Writing Comments"}
            )
            fig_comm_late.update_layout(height=340, yaxis_range=[0, 100], **PLOTLY_LAYOUT)
            st.plotly_chart(fig_comm_late, width="stretch")

        st.markdown("#### 🔬 Product Dimension Imputation Experiment (15% Artificially Masked Values)")
        imputation_results = pd.DataFrame([
            {"Feature": "Product Weight (g)", "Method": "Mean", "MAE": 2550.90, "RMSE": 4368.12, "Masked N": 4942},
            {"Feature": "Product Weight (g)", "Method": "Median", "MAE": 1964.98, "RMSE": 4649.87, "Masked N": 4942},
            {"Feature": "Product Weight (g)", "Method": "kNN Imputer (k=5)", "MAE": 2130.60, "RMSE": 3944.67, "Masked N": 4942},
            {"Feature": "Product Weight (g)", "Method": "Multiple Imputation (MICE)", "MAE": 2300.23, "RMSE": 3523.95, "Masked N": 4942},

            {"Feature": "Product Length (cm)", "Method": "Mean", "MAE": 12.65, "RMSE": 16.77, "Masked N": 4942},
            {"Feature": "Product Length (cm)", "Method": "Median", "MAE": 11.92, "RMSE": 17.74, "Masked N": 4942},
            {"Feature": "Product Length (cm)", "Method": "kNN Imputer (k=5)", "MAE": 11.02, "RMSE": 15.91, "Masked N": 4942},
            {"Feature": "Product Length (cm)", "Method": "Multiple Imputation (MICE)", "MAE": 10.68, "RMSE": 15.06, "Masked N": 4942},

            {"Feature": "Product Height (cm)", "Method": "Mean", "MAE": 9.68, "RMSE": 13.89, "Masked N": 4942},
            {"Feature": "Product Height (cm)", "Method": "Median", "MAE": 9.29, "RMSE": 14.52, "Masked N": 4942},
            {"Feature": "Product Height (cm)", "Method": "kNN Imputer (k=5)", "MAE": 9.51, "RMSE": 13.77, "Masked N": 4942},
            {"Feature": "Product Height (cm)", "Method": "Multiple Imputation (MICE)", "MAE": 9.25, "RMSE": 12.76, "Masked N": 4942},

            {"Feature": "Product Width (cm)", "Method": "Mean", "MAE": 9.40, "RMSE": 12.04, "Masked N": 4942},
            {"Feature": "Product Width (cm)", "Method": "Median", "MAE": 8.77, "RMSE": 12.45, "Masked N": 4942},
            {"Feature": "Product Width (cm)", "Method": "kNN Imputer (k=5)", "MAE": 7.43, "RMSE": 10.71, "Masked N": 4942},
            {"Feature": "Product Width (cm)", "Method": "Multiple Imputation (MICE)", "MAE": 7.75, "RMSE": 10.62, "Masked N": 4942},
        ])

        col_imp_sel, col_imp_met = st.columns(2)
        with col_imp_sel:
            sel_imp_feat = st.selectbox("Select Dimension to Compare Imputation Errors", imputation_results["Feature"].unique(), index=0)
        with col_imp_met:
            sel_imp_metric = st.radio("Error Metric", ["MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)"], horizontal=True)

        metric_key = "MAE" if "MAE" in sel_imp_metric else "RMSE"
        sub_imp = imputation_results[imputation_results["Feature"] == sel_imp_feat]

        fig_imp = px.bar(
            sub_imp,
            x="Method", y=metric_key,
            color="Method",
            color_discrete_sequence=[BLUE, ORANGE, PURPLE, GREEN],
            text=sub_imp[metric_key].round(2),
            title=f"Imputation Accuracy on 15% Masked Values ({metric_key}): {sel_imp_feat}",
        )
        fig_imp.update_layout(height=340, showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_imp, width="stretch")

    with st.expander("💡 Key Analytical Insights — Lectures 6, 7 & 8", expanded=True):
        st.info(
            "1. **Outlier Mechanism:** Price and Freight exhibit extreme right-skewness (skewness > 9-12). While IQR flags ~8-10% as outliers, Z-score flags ~1.7%, confirming that Z-score is more conservative and suitable when retaining valid high-value orders.\n\n"
            "2. **Distribution Stabilization:** Applying **Log1p transformation** reduces price skewness from **9.73 to 0.29** and freight skewness from **12.05 to 0.47**, stabilizing variance for downstream PCA and K-Means clustering.\n\n"
            "3. **Missing Not At Random (MNAR):** Customers who rate 1★ write text comments **76.6%** of the time, compared to only **35.9%** for 5★ ratings. When an order is >7 days late, **65.4%** leave written feedback. This proves review comments suffer from non-random reporting bias.\n\n"
            "4. **Optimal Imputation Strategy:** In the 15% masking experiment, **Multiple Imputation by Chained Equations (MICE)** achieves the lowest RMSE across all physical dimensions (e.g. 3523.95 vs 4649.87 for Median on Weight), while **Category Median** yields the lowest MAE on heavily skewed weight."
        )

# ══════════════════════════════════════════════════════════════
# TAB 4 — REVIEW & SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">💬 What Do Customers Really Complain About?</div>', unsafe_allow_html=True)
    st.caption("NLP analysis of Portuguese review comments — TF-IDF extraction reveals the exact phrases behind 1-star vs 5-star ratings.")

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        # Review score distribution
        review_dist = filt["review_score"].value_counts().sort_index()
        colors_rev = [RED, ORANGE, YELLOW, BLUE, GREEN]
        fig_rev = go.Figure(go.Bar(
            x=review_dist.index.astype(int),
            y=review_dist.values,
            marker=dict(color=colors_rev),
            text=review_dist.values,
            textposition="auto",
        ))
        fig_rev.update_layout(
            title="Review Score Distribution",
            xaxis_title="Stars", yaxis_title="Count",
            height=340,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_rev, width="stretch")

    with col_r2:
        # Lateness bucket vs review score
        late_review = (
            filt.dropna(subset=["lateness_bucket", "review_score"])
            .groupby("lateness_bucket")["review_score"].mean()
            .reindex(bucket_order).dropna()
        )
        fig_lr = go.Figure(go.Bar(
            x=late_review.index,
            y=late_review.values,
            marker=dict(color=[GREEN, BLUE, YELLOW, ORANGE, RED]),
            text=late_review.values.round(2),
            textposition="auto",
        ))
        fig_lr.update_layout(
            title="Avg Review Score by Lateness Bucket",
            yaxis_title="Avg Review Score", yaxis_range=[0, 5.5],
            height=340,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_lr, width="stretch")

    # NLP: Top terms — 1-star vs 5-star
    st.markdown("#### 🔤 Top Complaint Phrases (1★) vs Praise Phrases (5★)")
    col_n1, col_n2 = st.columns(2)

    with col_n1:
        terms_1star = nlp_top[nlp_top["group"] == "1-star"].head(15)
        if len(terms_1star):
            fig_t1 = go.Figure(go.Bar(
                x=terms_1star["mean_tfidf"],
                y=terms_1star["term"],
                orientation="h",
                marker_color=RED,
                text=terms_1star["mean_tfidf"].round(4),
                textposition="auto",
            ))
            fig_t1.update_layout(
                title="Top 15 Terms in 1★ Reviews",
                xaxis_title="Mean TF-IDF",
                height=420,
                yaxis=dict(autorange="reversed"),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_t1, width="stretch")

    with col_n2:
        terms_5star = nlp_top[nlp_top["group"] == "5-star"].head(15)
        if len(terms_5star):
            fig_t5 = go.Figure(go.Bar(
                x=terms_5star["mean_tfidf"],
                y=terms_5star["term"],
                orientation="h",
                marker_color=GREEN,
                text=terms_5star["mean_tfidf"].round(4),
                textposition="auto",
            ))
            fig_t5.update_layout(
                title="Top 15 Terms in 5★ Reviews",
                xaxis_title="Mean TF-IDF",
                height=420,
                yaxis=dict(autorange="reversed"),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_t5, width="stretch")

    # Distinctive terms (difference between 1-star and 5-star)
    st.markdown("#### 🎯 Most Distinctive Terms: What Separates 1★ from 5★")
    top_n = st.slider("Show top N distinctive terms", 10, 50, 20)
    dist_data = nlp_dist.head(top_n)
    if len(dist_data):
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(
            x=dist_data["term"], y=dist_data["mean_1star"],
            name="1★ Reviews", marker_color=RED, opacity=0.8,
        ))
        fig_diff.add_trace(go.Bar(
            x=dist_data["term"], y=dist_data["mean_5star"],
            name="5★ Reviews", marker_color=GREEN, opacity=0.8,
        ))
        fig_diff.update_layout(
            title="TF-IDF Comparison: 1★ vs 5★ Reviews",
            barmode="group",
            xaxis_tickangle=-45,
            height=400,
            legend=dict(orientation="h", y=1.1),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_diff, width="stretch")

    with st.expander("💡 Key Insight", expanded=True):
        st.info(
            '**"não recebi" (did not receive)** is the single strongest signal of a 1★ review. '
            'Other dominant complaint terms include "não entregue" (not delivered), "não chegou" (didn\'t arrive), '
            'and "comprei" (I bought [but...]). In contrast, 5★ reviews use positive terms like "recomendo" (I recommend), '
            '"ótimo" (great), and "excelente". The analysis preserved Portuguese negation words (não, nunca, sem) '
            "during stopword removal, which was critical — without them, complaint phrases become uninterpretable."
        )

# ══════════════════════════════════════════════════════════════
# TAB 5 — SELLER SEGMENTATION
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">🎯 Seller Performance Clusters & Scorecards</div>', unsafe_allow_html=True)
    st.caption("K-Means clustering on PCA-reduced seller features identifies 4 seller archetypes with distinct operational profiles.")

    # Archetype distribution
    col_s1, col_s2 = st.columns([2, 3])

    with col_s1:
        arch_counts = sellers["seller_archetype"].value_counts()
        arch_colors = {
            "Tier 1: Star High-Volume Seller": GREEN,
            "Tier 2: Premium / High-Ticket Seller": BLUE,
            "Tier 3: Standard Marketplace Seller": YELLOW,
            "Tier 4: At-Risk / Slow Seller": RED,
        }
        fig_arch = go.Figure(go.Pie(
            labels=arch_counts.index,
            values=arch_counts.values,
            hole=0.5,
            marker=dict(colors=[arch_colors.get(a, PURPLE) for a in arch_counts.index]),
            textinfo="percent+label",
            textfont=dict(size=10),
        ))
        fig_arch.update_layout(
            title="Seller Archetype Distribution",
            height=380,
            showlegend=False,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_arch, width="stretch")

    with col_s2:
        # PCA scatter colored by archetype
        fig_pca = px.scatter(
            sellers.dropna(subset=["pca_dim1", "pca_dim2"]),
            x="pca_dim1", y="pca_dim2",
            color="seller_archetype",
            color_discrete_map=arch_colors,
            opacity=0.6,
            hover_data=["total_orders", "total_revenue", "late_rate", "avg_review_score"],
            labels={"pca_dim1": "PCA Component 1", "pca_dim2": "PCA Component 2"},
        )
        fig_pca.update_layout(
            title="Seller Clusters in PCA Space",
            height=380,
            legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_pca, width="stretch")

    # Radar chart comparing archetypes
    st.markdown("#### 🕸️ Archetype Performance Radar")
    radar_metrics = ["avg_price", "avg_handling_days", "avg_transit_days", "late_rate", "avg_review_score", "avg_distance_km"]
    radar_labels = ["Avg Price", "Handling Days", "Transit Days", "Late Rate", "Avg Review", "Avg Distance"]

    radar_data = sellers.groupby("seller_archetype")[radar_metrics].median().reset_index()

    # Normalize each metric to 0-1 for radar comparison
    radar_norm = radar_data.copy()
    for col in radar_metrics:
        col_min = radar_norm[col].min()
        col_max = radar_norm[col].max()
        if col_max > col_min:
            radar_norm[col] = (radar_norm[col] - col_min) / (col_max - col_min)
        else:
            radar_norm[col] = 0.5

    fig_radar = go.Figure()
    for _, row in radar_norm.iterrows():
        vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
        labs = radar_labels + [radar_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=labs,
            fill="toself",
            name=row["seller_archetype"],
            line=dict(color=arch_colors.get(row["seller_archetype"], PURPLE)),
            opacity=0.6,
        ))
    fig_radar.update_layout(
        title="Normalized Seller Archetype Profiles",
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1])),
        height=440,
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_radar, width="stretch")

    # Reliability score distribution by archetype
    st.markdown("#### 📊 Reliability Score Distribution by Archetype")
    fig_rel = px.box(
        sellers.dropna(subset=["reliability_score"]),
        x="seller_archetype", y="reliability_score",
        color="seller_archetype",
        color_discrete_map=arch_colors,
    )
    fig_rel.update_layout(
        height=360,
        showlegend=False,
        xaxis_title="",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_rel, width="stretch")

    with st.expander("💡 Key Insight", expanded=True):
        st.info(
            "**Tier 1 (Star High-Volume)** sellers handle the most orders with the best review scores and lowest late rates. "
            "**Tier 4 (At-Risk)** sellers show significantly higher handling times and late rates — "
            "these ~750 sellers are candidates for operational coaching or shorter shipping limit enforcement. "
            "The PCA projection shows clear cluster separation, validating that seller performance is multi-dimensional "
            "and cannot be captured by a single metric."
        )

# ══════════════════════════════════════════════════════════════
# TAB 6 — RISK PATTERN MINING
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">⛏️ Association Rule Mining — What Causes Bad Reviews?</div>', unsafe_allow_html=True)
    st.caption("FP-Growth/Apriori rules mined from order attributes → identifies exact combinations of route, delay, and category that predict 1-2★ reviews.")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### 🚨 Top Bad-Review Risk Rules")
        min_conf = st.slider("Minimum Confidence", 0.5, 1.0, 0.7, 0.05)
        risk_filt = rules_risk[rules_risk["confidence"] >= min_conf].sort_values("lift", ascending=False).head(20)

        if len(risk_filt):
            fig_risk = go.Figure(go.Bar(
                y=risk_filt["antecedent"].str[:60],
                x=risk_filt["lift"],
                orientation="h",
                marker=dict(
                    color=risk_filt["confidence"],
                    colorscale="YlOrRd",
                    colorbar=dict(title="Confidence"),
                ),
                text=risk_filt["lift"].round(2),
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Lift: %{x:.2f}<br>Confidence: %{marker.color:.2f}<extra></extra>",
            ))
            fig_risk.update_layout(
                title="Top Rules by Lift (→ Bad Review)",
                xaxis_title="Lift",
                height=550,
                yaxis=dict(autorange="reversed"),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_risk, width="stretch")
        else:
            st.warning("No rules found above the selected confidence threshold.")

    with col_r2:
        st.markdown("#### 🛒 Category Co-Purchase Rules")
        if len(rules_copurchase):
            cop = rules_copurchase.sort_values("lift", ascending=False).head(15)
            cop["rule"] = cop["antecedent"] + " → " + cop["consequent"]
            fig_cop = go.Figure(go.Bar(
                y=cop["rule"],
                x=cop["lift"],
                orientation="h",
                marker_color=PURPLE,
                text=cop["lift"].round(2),
                textposition="auto",
            ))
            fig_cop.update_layout(
                title="Top Category Co-Purchase Rules by Lift",
                xaxis_title="Lift",
                height=550,
                yaxis=dict(autorange="reversed"),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_cop, width="stretch")

    # Confidence vs Lift scatter for risk rules
    st.markdown("#### Confidence vs Lift — Rule Quality Map")
    if len(rules_risk):
        fig_rl = px.scatter(
            rules_risk,
            x="confidence", y="lift",
            size="n_orders",
            color="conviction",
            color_continuous_scale="Viridis",
            hover_data=["antecedent"],
            labels={"confidence": "Confidence", "lift": "Lift", "n_orders": "Orders", "conviction": "Conviction"},
            opacity=0.7,
        )
        fig_rl.update_layout(
            title="Association Rule Quality: Confidence × Lift × Order Volume",
            height=420,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_rl, width="stretch")

    with st.expander("💡 Key Insight", expanded=True):
        st.info(
            "The strongest risk pattern is clear: **cross-state shipments that are >7 days late** have an ~80% probability "
            "of receiving a 1-2★ review (lift ~6×). Adding factors like slow seller handling or the Southeast customer region "
            "makes the rule even more predictive. This validates that **delivery lateness is the #1 driver of bad reviews**, "
            "more impactful than product category, price, or payment method."
        )

# ══════════════════════════════════════════════════════════════
# TAB 7 — DEEP DIVE EXPLORER
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">🔎 Custom Data Explorer — Build Your Own Analysis</div>', unsafe_allow_html=True)
    st.caption("Flexible scatter plot and aggregation tool — choose any two variables to uncover hidden relationships.")

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    numeric_cols = [
        "total_price", "total_freight", "freight_ratio", "total_weight_g",
        "distance_km", "approval_hours", "seller_handling_days",
        "carrier_transit_days", "total_delivery_days", "days_late",
        "review_score", "item_count", "max_installments",
    ]
    with col_e1:
        x_var = st.selectbox("X-axis", numeric_cols, index=4)
    with col_e2:
        y_var = st.selectbox("Y-axis", numeric_cols, index=8)
    with col_e3:
        color_var = st.selectbox("Color by", ["review_score", "is_late", "lateness_bucket", "customer_state", "primary_category", "is_cross_state"], index=0)
    with col_e4:
        sample_size = st.slider("Sample size", 500, 10000, 3000, 500)

    # Sample data for scatter
    explore_data = filt.dropna(subset=[x_var, y_var]).sample(n=min(sample_size, len(filt)), random_state=42)

    if color_var == "review_score":
        fig_explore = px.scatter(
            explore_data, x=x_var, y=y_var, color=color_var,
            color_continuous_scale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
            opacity=0.5,
            trendline="ols",
        )
    else:
        fig_explore = px.scatter(
            explore_data, x=x_var, y=y_var, color=color_var,
            color_discrete_sequence=COLORS,
            opacity=0.5,
        )
    fig_explore.update_layout(
        title=f"{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}",
        height=480,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_explore, width="stretch")

    # Category-level aggregation table
    st.markdown("#### 📋 Category Performance Summary")
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
                         .format({"avg_price": "R$ {:.2f}", "avg_freight": "R$ {:.2f}", "avg_distance": "{:.0f} km"}),
        width="stretch",
        height=450,
    )

    # Correlation heatmap
    st.markdown("#### 🔗 Correlation Matrix — Key Numeric Features")
    corr_cols = ["total_price", "total_freight", "distance_km", "seller_handling_days",
                 "carrier_transit_days", "total_delivery_days", "days_late", "review_score"]
    corr_data = filt[corr_cols].dropna()
    if len(corr_data) > 100:
        corr_matrix = corr_data.corr()
        fig_corr = px.imshow(
            corr_matrix.values,
            x=[c.replace("_", " ").title() for c in corr_matrix.columns],
            y=[c.replace("_", " ").title() for c in corr_matrix.index],
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
        )
        fig_corr.update_layout(
            title="Pearson Correlation Matrix",
            height=500,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_corr, width="stretch")

    with st.expander("💡 Key Insight", expanded=True):
        st.info(
            "The correlation matrix reveals that **days_late has the strongest negative correlation with review_score** (~-0.34), "
            "confirming that delivery lateness is the primary driver of customer dissatisfaction. "
            "Freight cost correlates positively with distance (expected), while price has a surprisingly weak correlation with reviews — "
            "customers care far more about delivery reliability than product cost."
        )

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:0.8rem; padding: 10px;'>"
    "📦 <b>DELTA</b> — Delivery Estimate, Lateness & Trust Analytics  •  DS3030 Data Analytics  •  "
    "Built with Streamlit + Plotly  •  Dataset: Olist Brazilian E-Commerce (Kaggle)"
    "</div>",
    unsafe_allow_html=True,
)
