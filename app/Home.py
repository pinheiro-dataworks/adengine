from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_loader as dl
import theme

st.set_page_config(page_title="AdEngine — Executive Overview", page_icon="📈", layout="wide")
theme.inject_css()
theme.render_sidebar_brand()

if not dl.data_available():
    st.error("No pipeline artifacts found in data/marts/. Run `make pipeline && make train` first.")
    st.stop()

fact = dl.load_parquet("fact_transactions")
dim_customers = dl.load_parquet("dim_customers")
cf_test = dl.load_parquet("customer_features_test")
propensity_scores = dl.load_parquet("propensity_scores")
attribution = dl.load_parquet("synthetic_attribution")
channel_metrics = dl.load_parquet("channel_metrics")
segment_profiles = dl.load_parquet("segment_profiles")
ltv_table = dl.load_parquet("ltv_table")
dq_report = dl.load_dq_report()
pipe_cfg = dl.load_config("pipeline")

st.markdown(
    '<p class="ade-section-eyebrow">Page 1 of 6 · decision audience: leadership</p>'
    '<h1 style="margin-bottom:2px;">Executive Overview</h1>'
    f'<p style="color:{theme.SLATE};font-size:13.5px;max-width:640px;">'
    'Is demand growing, is spend efficient, and where is customer value concentrated right now?</p>',
    unsafe_allow_html=True,
)

# ---- KPI computation --------------------------------------------------
total_revenue = fact["revenue"].sum()
active_customers = len(dim_customers)
conversion_rate = cf_test["target_conversion"].mean()
total_spend = channel_metrics["spend_synthetic"].sum()
total_conversions = channel_metrics["conversions"].sum()
total_window_revenue = channel_metrics["revenue"].sum()
blended_cpa = total_spend / total_conversions
blended_roas = total_window_revenue / total_spend
avg_ltv = ltv_table["ltv_probabilistic"].mean()

seg_mix = segment_profiles[["segment_name", "size", "pct_of_base"]].sort_values("size", ascending=False)
seg_colors = [theme.SEGMENT_COLORS.get(n.rsplit(" ", 1)[0] if n.rsplit(" ", 1)[-1].isdigit() else n, theme.SLATE) for n in seg_mix["segment_name"]]

col1, col2 = st.columns([1.7, 1], gap="medium")
with col1:
    stage_bar = "".join(
        f'<div style="height:100%;width:{p*100:.1f}%;background:{c};"></div>'
        for p, c in zip(seg_mix["pct_of_base"], seg_colors)
    )
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:rgba(244,246,243,0.62);">'
        f'<span style="width:8px;height:8px;border-radius:2px;background:{c};flex-shrink:0;"></span>{n} · '
        f'<b class="ade-mono" style="color:{theme.PAPER};">{s:,}</b></div>'
        for n, s, c in zip(seg_mix["segment_name"], seg_mix["size"], seg_colors)
    )
    st.markdown(
        theme.hero_card(
            "Total revenue", f"£{total_revenue/1e6:.2f}M",
            f'Blended <b>{blended_roas:.1f}×</b> ROAS · <b>£{blended_cpa:,.0f}</b> CPA · <b>£{avg_ltv:,.0f}</b> avg. LTV (12mo projected)',
            live_tag="observed",
        ).replace(
            '<div class="ade-hero__sub">',
            f'<div style="height:10px;border-radius:6px;overflow:hidden;background:rgba(244,246,243,0.08);display:flex;margin:12px 0;">{stage_bar}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:4px;">{legend}</div>'
            '<div class="ade-hero__sub" style="display:none">',
        ),
        unsafe_allow_html=True,
    )
with col2:
    c1, c2 = st.columns(1), None
    st.markdown(theme.stat_card('Active Customers <span class="ade-tag ade-tag--observed">Observed</span>', f"{active_customers:,}"), unsafe_allow_html=True)
    st.write("")
    delta = f'<span style="color:{theme.OBSERVED}">Realized outcome, T_test cohort</span>'
    st.markdown(theme.stat_card('90-Day Conversion <span class="ade-tag ade-tag--observed">Observed</span>', f"{conversion_rate*100:.1f}%", delta), unsafe_allow_html=True)

st.write("")
st.write("")

# ---- Pipeline layers + health ------------------------------------------
tab_col, detail_col = st.columns([1.5, 1], gap="medium")
with tab_col:
    st.markdown("##### Data pipeline layers")
    layers = pd.DataFrame([
        {"Layer": "Bronze (raw ingest)", "Rows": f'{dq_report["rows_raw"]:,}', "Note": "no transformation"},
        {"Layer": "Silver (cleaned)", "Rows": f'{dq_report["rows_silver"]:,}', "Note": f'{dq_report["retention_pct"]}% retained · 5 DQ rules'},
        {"Layer": "Gold — fact_transactions", "Rows": f'{len(fact):,}', "Note": "grain: one row per line item"},
        {"Layer": "Gold — dim_customers", "Rows": f'{len(dim_customers):,}', "Note": "grain: one row per customer"},
    ])
    st.dataframe(layers, hide_index=True, use_container_width=True)
with detail_col:
    st.markdown("##### Pipeline health")
    st.markdown(
        '<div class="ade-stat"><div class="ade-stat__label">Bronze &rarr; Silver</div>'
        '<div style="display:flex;justify-content:space-between;font-size:13px;margin-top:8px;">'
        f'<span>Raw rows</span><span class="ade-mono">{dq_report["rows_raw"]:,}</span></div>'
        '<div style="display:flex;justify-content:space-between;font-size:13px;margin-top:6px;">'
        f'<span>Rows removed</span><span class="ade-mono">{dq_report["rows_raw"]-dq_report["rows_silver"]:,}</span></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-top:6px;border-top:1px solid {theme.LINE};padding-top:6px;font-weight:600;">'
        f'<span>Retention rate</span><span class="ade-mono" style="color:{theme.OBSERVED}">{dq_report["retention_pct"]}%</span></div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---- Charts -------------------------------------------------------------
left, right = st.columns([2, 1], gap="medium")

with left:
    st.markdown(
        f'<p class="ade-section-eyebrow">Which months carry disproportionate revenue weight?</p>'
        f'<h4 style="margin-top:0;">Monthly Revenue Trend {theme.tag("Observed", "observed")}</h4>',
        unsafe_allow_html=True,
    )
    monthly = fact.copy()
    monthly["month"] = monthly["invoice_date"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    monthly = monthly.groupby("month")["revenue"].sum().reset_index()
    fig = go.Figure(go.Scatter(
        x=monthly["month"], y=monthly["revenue"] / 1000, mode="lines", fill="tozeroy",
        line=dict(color=theme.OBSERVED, width=2), fillcolor="rgba(14,124,123,0.10)",
        hovertemplate="%{x|%b %Y}<br>£%{y:.0f}k<extra></extra>",
    ))
    fig.update_yaxes(title="Revenue (£k)")
    theme.apply_plotly_theme(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.caption("Two seasonal spikes (Nov–Dec) reflect UK holiday gifting demand — both fall before T_train and T_test, so seasonality is a control variable in the propensity model, not a leakage source.")

with right:
    st.markdown(
        f'<p class="ade-section-eyebrow">Where is revenue concentrated?</p>'
        f'<h4 style="margin-top:0;">Revenue Share by Country {theme.tag("Observed", "observed")}</h4>',
        unsafe_allow_html=True,
    )
    by_country = fact.groupby("country")["revenue"].sum().sort_values(ascending=False)
    top = by_country.head(5)
    other = by_country.iloc[5:].sum()
    country_df = pd.concat([top, pd.Series({"Other": other})]).reset_index()
    country_df.columns = ["country", "revenue"]
    country_df["pct"] = 100 * country_df["revenue"] / by_country.sum()
    fig = go.Figure(go.Bar(
        y=country_df["country"], x=country_df["pct"], orientation="h",
        marker_color=theme.CATEGORICAL[: len(country_df)],
        hovertemplate="%{y}: %{x:.1f}%% of revenue<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="% of revenue")
    theme.apply_plotly_theme(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

st.write("")
st.markdown(
    f'<p class="ade-section-eyebrow">How much of the base reaches high-value activation?</p>'
    f'<h4 style="margin-top:0;">Customer Activation Funnel {theme.tag("Modeled", "modeled")}</h4>',
    unsafe_allow_html=True,
)

scored = cf_test.merge(propensity_scores[["customer_id", "score_gbm_calibrated"]], on="customer_id").merge(
    attribution[["customer_id", "exposed"]], on="customer_id"
)
stage1 = scored
stage2 = scored[scored["target_conversion"] == 1]
stage3 = stage2[stage2["score_gbm_calibrated"] > 0.6]
stage4 = stage3[stage3["exposed"]]
funnel_stages = [
    ("All customers, T_test window base", len(stage1), theme.BLUE),
    ("Repeat purchasers (90d, realized)", len(stage2), theme.OBSERVED),
    ("...and correctly flagged (score > 0.6)", len(stage3), theme.MODELED),
    ("...and campaign-exposed (synthetic)", len(stage4), theme.SYNTHETIC),
]
fig = go.Figure(go.Funnel(
    y=[s[0] for s in funnel_stages], x=[s[1] for s in funnel_stages],
    marker=dict(color=[s[2] for s in funnel_stages]),
    textinfo="value+percent initial",
    textfont=dict(family=theme.FONT_MONO, size=12, color="white"),
    connector=dict(line=dict(color=theme.LINE, width=1)),
))
theme.apply_plotly_theme(fig, height=320)
st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
st.caption("Each stage is a strict subset of the one above: 90-day repeat purchasers who the calibrated model also ranked above 0.6, and who additionally landed in the synthetic campaign-exposed group.")

st.markdown("---")
st.caption(f"Source: Online Retail II (UCI) · window ends {pipe_cfg['temporal']['t_test']} · Provenance: Observed / Modeled / Synthetic")
