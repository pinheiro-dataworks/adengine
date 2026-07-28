from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_loader as dl
import theme

st.set_page_config(page_title="AdEngine — RFM Segmentation", page_icon="🧩", layout="wide")
theme.inject_css()
theme.render_sidebar_brand()

if not dl.data_available():
    st.error("No pipeline artifacts found in data/marts/. Run `make pipeline && make train` first.")
    st.stop()

profiles = dl.load_parquet("segment_profiles")
assignments = dl.load_parquet("segment_assignments")
k_sweep = dl.load_parquet("segment_k_sweep")
summary = dl.load_json("segmentation_summary")
cf_current = dl.load_parquet("customer_features_current")

st.markdown(
    '<h1 style="margin-bottom:2px;">RFM Segmentation</h1>'
    f'<p style="color:{theme.SLATE};font-size:13.5px;max-width:680px;">'
    'Which behavioral groups exist in the customer base, how stable are they, and what should the business do with each one?</p>',
    unsafe_allow_html=True,
)

def base_name(n: str) -> str:
    parts = n.rsplit(" ", 1)
    return parts[0] if len(parts) == 2 and parts[-1].isdigit() else n


top_segment = profiles.iloc[0]
col1, col2 = st.columns([1.7, 1], gap="medium")
with col1:
    stage_bar = "".join(
        f'<div style="height:100%;width:{p*100:.1f}%;background:{theme.SEGMENT_COLORS.get(base_name(n), theme.SLATE)};"></div>'
        for n, p in zip(profiles["segment_name"], profiles["pct_of_base"])
    )
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:rgba(244,246,243,0.62);">'
        f'<span style="width:8px;height:8px;border-radius:2px;background:{theme.SEGMENT_COLORS.get(base_name(n), theme.SLATE)};flex-shrink:0;"></span>{n} · '
        f'<b class="ade-mono" style="color:{theme.PAPER};">{s:,}</b></div>'
        for n, s in zip(profiles["segment_name"], profiles["size"])
    )
    st.markdown(
        f'<div class="ade-hero"><div class="ade-hero__label">Segment spotlight '
        f'<span class="ade-mono" style="color:{theme.LIME};font-size:11px;">&#9679; {top_segment["segment_name"].lower()}</span></div>'
        f'<div class="ade-hero__value">{top_segment["size"]:,}</div>'
        f'<div class="ade-hero__sub">{top_segment["segment_name"]} · <b>{top_segment["conversion_rate"]*100:.0f}%</b> 90-day conversion '
        f'· <b>£{top_segment["ltv_historical"]:,.0f}</b> avg. historical LTV</div>'
        f'<div style="height:10px;border-radius:6px;overflow:hidden;background:rgba(244,246,243,0.08);display:flex;margin:12px 0;">{stage_bar}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;">{legend}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(theme.stat_card('Segments Identified <span class="ade-tag ade-tag--modeled">Modeled</span>', str(summary["chosen_k"]), "K via elbow + silhouette + business read"), unsafe_allow_html=True)
    st.write("")
    stable_color = theme.OBSERVED if summary["stable"] else theme.RISK
    st.markdown(theme.stat_card(
        'Cluster Stability (ARI) <span class="ade-tag ade-tag--modeled">Modeled</span>',
        f'{summary["ari_bootstrap_mean"]:.2f}',
        f'<span style="color:{stable_color}">{"&#9650; above" if summary["stable"] else "&#9660; below"} {summary["ari_threshold"]} bar (bootstrap)</span>',
    ), unsafe_allow_html=True)

st.write("")
st.write("")

left, right = st.columns(2, gap="medium")
with left:
    st.markdown(
        f'<p class="ade-section-eyebrow">How many clusters are supported by the data?</p>'
        f'<h4 style="margin-top:0;">Elbow Method — Inertia by K {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure(go.Scatter(
        x=k_sweep["k"], y=k_sweep["inertia"], mode="lines+markers",
        line=dict(color=theme.SLATE, width=2),
        marker=dict(size=[12 if k == summary["chosen_k"] else 7 for k in k_sweep["k"]],
                    color=[theme.OBSERVED if k == summary["chosen_k"] else theme.SLATE for k in k_sweep["k"]]),
        hovertemplate="K=%{x}<br>Inertia=%{y:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(title="K (clusters)", dtick=1)
    fig.update_yaxes(title="Inertia (WCSS)")
    theme.apply_plotly_theme(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
with right:
    st.markdown(
        f'<p class="ade-section-eyebrow">Does the chosen K separate customers better than its silhouette-max neighbor?</p>'
        f'<h4 style="margin-top:0;">Silhouette Score by K {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure(go.Bar(
        x=k_sweep["k"], y=k_sweep["silhouette"],
        marker_color=[theme.OBSERVED if k == summary["chosen_k"] else "#CBD5E1" for k in k_sweep["k"]],
        hovertemplate="K=%{x}<br>Silhouette=%{y:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="K (clusters)", dtick=1)
    fig.update_yaxes(title="Silhouette score")
    theme.apply_plotly_theme(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

st.markdown(theme.callout(
    f'<b>K=2 maximizes silhouette</b> ({k_sweep["silhouette"].max():.3f}) but only separates one-time buyers from everyone else — '
    f'no distinct marketing action per group. <b>K={summary["chosen_k"]}</b> sits at a local silhouette peak right where the inertia '
    'elbow flattens, and yields five classically-named, independently-actionable RFM tiers. See docs/adr/006-segmentation-k-selection.md.'
), unsafe_allow_html=True)

st.write("")
st.markdown(
    f'<p class="ade-section-eyebrow">How do segments separate on Recency and Frequency?</p>'
    f'<h4 style="margin-top:0;">RFM Scatter — Recency × Frequency (bubble size = Monetary) {theme.tag("Modeled", "modeled")}</h4>',
    unsafe_allow_html=True,
)
scatter_df = cf_current.merge(assignments[["customer_id", "segment_name"]], on="customer_id")
fig = go.Figure()
for name in profiles["segment_name"]:
    sub = scatter_df[scatter_df["segment_name"] == name]
    color = theme.SEGMENT_COLORS.get(base_name(name), theme.SLATE)
    fig.add_trace(go.Scatter(
        x=sub["recency_days"], y=sub["frequency"], mode="markers", name=name,
        marker=dict(size=(sub["monetary"].clip(upper=sub["monetary"].quantile(0.98)) ** 0.5) / 3 + 5,
                    color=color, opacity=0.55, line=dict(width=0)),
        hovertemplate=f"{name}<br>Recency=%{{x}}d, Frequency=%{{y}}, Monetary=£%{{customdata:,.0f}}<extra></extra>",
        customdata=sub["monetary"],
    ))
fig.update_xaxes(title="Recency (days since last order)")
fig.update_yaxes(title="Frequency (orders in window)")
theme.apply_plotly_theme(fig, height=420)
st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
st.caption("Recency and Monetary were log-transformed and winsorized before scaling and clustering; raw units are shown here for interpretability.")

st.write("")
st.markdown('<h4 style="margin-top:0;">Segment Profiles &amp; Recommended Actions</h4>', unsafe_allow_html=True)
display_cols = profiles.rename(columns={
    "segment_name": "Segment", "size": "Customers", "pct_of_base": "% of base",
    "recency_days": "Avg. Recency (d)", "frequency": "Avg. Frequency", "monetary": "Avg. Historical LTV (£)",
    "avg_order_value": "Avg. Order Value (£)",
    "conversion_rate": "90d Conversion", "recommended_action": "Recommended Action",
})
display_cols["% of base"] = (display_cols["% of base"] * 100).round(1)
display_cols["90d Conversion"] = (display_cols["90d Conversion"] * 100).round(1)
for c in ["Avg. Recency (d)", "Avg. Frequency", "Avg. Historical LTV (£)", "Avg. Order Value (£)"]:
    display_cols[c] = display_cols[c].round(0)
st.dataframe(
    display_cols[["Segment", "Customers", "% of base", "Avg. Recency (d)", "Avg. Frequency",
                   "Avg. Order Value (£)", "Avg. Historical LTV (£)", "90d Conversion", "Recommended Action"]],
    hide_index=True, use_container_width=True,
)

csv = assignments.merge(cf_current[["customer_id", "recency_days", "frequency", "monetary"]], on="customer_id").to_csv(index=False).encode("utf-8")
st.download_button("Export segment assignments (CSV)", csv, "adengine_segments.csv", "text/csv")

st.markdown("---")
st.caption(f"K selected via elbow + silhouette, tie-broken by business interpretability · {len(assignments):,} customers · window ending {dl.load_config('pipeline')['temporal']['t_test']}")
