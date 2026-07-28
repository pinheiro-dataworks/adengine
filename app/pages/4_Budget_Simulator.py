from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_loader as dl
import theme

from adengine.simulator import HillParams, hill_response, optimize_allocation

st.set_page_config(page_title="AdEngine — Budget Simulator", page_icon="🧮", layout="wide")
theme.inject_css()
theme.render_sidebar_brand()

if not dl.data_available():
    st.error("No pipeline artifacts found in data/marts/. Run `make pipeline && make train` first.")
    st.stop()

sim_summary = dl.load_json("simulator_summary")
bootstrap = dl.load_parquet("simulator_bootstrap")["total_conversions"]
model_cfg = dl.load_config("model")
sim_cfg = model_cfg["simulator"]

channels = list(sim_summary["hill_params"].keys())
params = {c: HillParams(**sim_summary["hill_params"][c]) for c in channels}

st.markdown(
    '<p class="ade-section-eyebrow">Page 5 of 6 · Hill-function response curves + SLSQP optimization</p>'
    '<h1 style="margin-bottom:2px;">Budget Allocation Simulator</h1>'
    f'<p style="color:{theme.SLATE};font-size:13.5px;max-width:720px;">'
    'Given a fixed monthly budget, how should spend split across channels to maximize expected conversions under diminishing returns?</p>',
    unsafe_allow_html=True,
)

if "budget_total" not in st.session_state:
    st.session_state["budget_total"] = float(sim_summary["total_budget"])
for c in channels:
    if f"alloc_{c}" not in st.session_state:
        st.session_state[f"alloc_{c}"] = float(sim_summary["optimal_allocation"][c])

top = st.columns([1, 3])
with top[0]:
    st.number_input("Total monthly budget (£)", min_value=5000, max_value=500000, step=1000, key="budget_total")

budget = st.session_state["budget_total"]
bounds = [(sim_cfg["channel_bounds_pct"]["min"] * budget, sim_cfg["channel_bounds_pct"]["max"] * budget) for _ in channels]

action_col1, action_col2, _ = st.columns([1, 1, 3])
with action_col1:
    if st.button("Optimize allocation", type="primary", use_container_width=True):
        result = optimize_allocation(params, budget, sim_cfg["channel_bounds_pct"], sim_cfg["optimizer"]["multistart_points"], model_cfg["random_state"])
        for c in channels:
            st.session_state[f"alloc_{c}"] = result["allocation"][c]
        st.rerun()
with action_col2:
    if st.button("Reset to equal split", use_container_width=True):
        for c in channels:
            st.session_state[f"alloc_{c}"] = budget / len(channels)
        st.rerun()

st.write("")
slider_col, chart_col = st.columns([1, 1.6], gap="medium")

with slider_col:
    st.markdown("##### Channel allocation")
    for c in channels:
        lo, hi = bounds[channels.index(c)]
        st.session_state[f"alloc_{c}"] = float(np.clip(st.session_state[f"alloc_{c}"], lo, hi))
        st.markdown(f'<span style="color:{theme.CHANNEL_COLORS[c]};font-weight:600;">&#9679; {theme.CHANNEL_LABELS[c]}</span>', unsafe_allow_html=True)
        st.slider(" ", min_value=float(lo), max_value=float(hi), step=100.0, key=f"alloc_{c}", label_visibility="collapsed")

    allocation = {c: st.session_state[f"alloc_{c}"] for c in channels}
    total_allocated = sum(allocation.values())
    conversions_by_channel = {c: hill_response(allocation[c], params[c]) for c in channels}
    total_conversions = sum(conversions_by_channel.values())

    color = theme.RISK if abs(total_allocated - budget) > budget * 0.02 else theme.INK
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-top:1px solid {theme.LINE};margin-top:8px;">'
        f'<span style="color:{theme.SLATE};font-size:13px;">Allocated</span>'
        f'<b class="ade-display" style="font-size:19px;color:{color};">£{total_allocated:,.0f}</b></div>',
        unsafe_allow_html=True,
    )

with chart_col:
    st.markdown(
        f'<p class="ade-section-eyebrow">Conversions vs. monthly spend, current allocation marked</p>'
        f'<h4 style="margin-top:0;">Response Curves (Hill function) {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    max_spend = max(hi for _, hi in bounds)
    spend_grid = np.linspace(0, max_spend, 60)
    for c in channels:
        fig.add_trace(go.Scatter(x=spend_grid, y=hill_response(spend_grid, params[c]), mode="lines", name=theme.CHANNEL_LABELS[c], line=dict(color=theme.CHANNEL_COLORS[c], width=2.2)))
    fig.add_trace(go.Scatter(
        x=[allocation[c] for c in channels], y=[conversions_by_channel[c] for c in channels], mode="markers", name="Current allocation",
        marker=dict(size=11, color=[theme.CHANNEL_COLORS[c] for c in channels], line=dict(color="white", width=1.5)), showlegend=False,
    ))
    fig.update_xaxes(title="Monthly spend (£)")
    fig.update_yaxes(title="Expected conversions / month")
    theme.apply_plotly_theme(fig, height=360)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.caption("Fitted as V_max · spend^n / (k^n + spend^n) — k anchored to observed CPA (metrics.py), V_max to addressable audience × average calibrated propensity (propensity.py), n is the saturation shape.")

st.write("")
k1, k2, k3 = st.columns(3, gap="medium")
with k1:
    st.markdown(theme.stat_card('Projected Conversions <span class="ade-tag ade-tag--modeled">Modeled</span>', f"{total_conversions:,.0f}"), unsafe_allow_html=True)
with k2:
    blended_cpa = total_allocated / total_conversions if total_conversions > 0 else float("inf")
    st.markdown(theme.stat_card('Blended CPA <span class="ade-tag ade-tag--modeled">Modeled</span>', f"£{blended_cpa:,.0f}"), unsafe_allow_html=True)
with k3:
    p10, p50, p90 = bootstrap.quantile(0.10), bootstrap.quantile(0.50), bootstrap.quantile(0.90)
    st.markdown(theme.stat_card('Bootstrap 10–90% Range <span class="ade-tag ade-tag--modeled">Modeled</span>', f"{p10:,.0f} – {p90:,.0f}", f'<span style="color:{theme.SLATE}">at the precomputed optimum, 200 resamples</span>'), unsafe_allow_html=True)

st.write("")
st.markdown(
    f'<p class="ade-section-eyebrow">How much does the optimum move under resampling uncertainty?</p>'
    f'<h4 style="margin-top:0;">Bootstrap Distribution of Total Conversions {theme.tag("Modeled", "modeled")}</h4>',
    unsafe_allow_html=True,
)
fig = go.Figure(go.Histogram(x=bootstrap, nbinsx=30, marker_color=theme.MODELED))
fig.add_vline(x=bootstrap.quantile(0.10), line_dash="dash", line_color=theme.SLATE, annotation_text="p10")
fig.add_vline(x=bootstrap.quantile(0.90), line_dash="dash", line_color=theme.SLATE, annotation_text="p90")
fig.update_xaxes(title="Total conversions at re-optimized allocation")
fig.update_yaxes(title="Bootstrap resamples")
theme.apply_plotly_theme(fig, height=260)
st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
st.caption("200 bootstrap resamples of the customer base, Hill parameters re-derived and re-optimized on each — communicating uncertainty is a requirement here, not decoration: a point allocation with no interval implies false precision.")

st.markdown("---")
st.caption(f"Optimizer: scipy.optimize.minimize(method='SLSQP'), {sim_cfg['optimizer']['multistart_points']} multistart points · bounds: {sim_cfg['channel_bounds_pct']['min']*100:.0f}%–{sim_cfg['channel_bounds_pct']['max']*100:.0f}% of budget per channel")
