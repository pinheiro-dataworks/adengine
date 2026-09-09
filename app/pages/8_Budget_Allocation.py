from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_loader as dl
import theme

st.set_page_config(page_title="AdEngine — Budget Allocation", page_icon="🎯", layout="wide")
theme.inject_css()
theme.render_sidebar_brand()

if not dl.data_available():
    st.error("No pipeline artifacts found in data/marts/. Run `make pipeline && make train` first.")
    st.stop()

if not (dl.MARTS_DIR / "allocation_summary.json").exists():
    st.error(
        "No allocation artifacts found in data/marts/. Run "
        "`python -m adengine.causal_simulation && python -m adengine.allocation_optimizer` first."
    )
    st.stop()

summary = dl.load_json("allocation_summary")
result = dl.load_parquet("allocation_result")
alloc_cfg = dl.load_config("allocation")

optimal = summary["optimal"]
baseline = summary["greedy_baseline"]
budget = summary["budget"]

eligible = result[result["cate_hat"] > 0]
unconstrained_value = float(eligible["cate_hat"].sum())
unconstrained_cost = float(eligible["cost"].sum())

st.markdown(
    '<h1 style="margin-bottom:2px;">Budget Allocation</h1>'
    f'<p style="color:{theme.SLATE};font-size:13.5px;max-width:760px;">'
    'Given a fixed budget for a targeted retention intervention and a predicted CATE per customer, '
    'which customers should actually receive it — and how much better is solving that exactly than a simple heuristic?</p>',
    unsafe_allow_html=True,
)

st.markdown(theme.disclaimer(
    "<b>CATE scores are synthetic and individually noisy.</b> The per-customer targeting priority "
    "(t_learner_cate) is scored against the ADR-007 synthetic ground truth — its segment-level averages "
    "track the true effect well, but any single customer's predicted score carries real estimation "
    "noise (see ADR-009). This page demonstrates the allocation logic, not a real targeting recommendation."
), unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1], gap="medium")
with col1:
    st.markdown(
        theme.hero_card(
            "Total incremental value captured",
            f'{optimal["total_incremental_value"]:.1f}',
            f'£{optimal["total_cost"]:,.0f} of £{budget:,.0f} budget spent · '
            f'<b>{optimal["n_treated"]:,}</b> of {len(result):,} customers treated · '
            f'{"MILP matches or beats" if summary["milp_beats_or_matches_greedy"] else "MILP underperforms"} the greedy baseline',
            live_tag="MILP optimal",
        ),
        unsafe_allow_html=True,
    )
with col2:
    milp_edge = optimal["total_incremental_value"] - baseline["total_incremental_value"]
    st.markdown(theme.stat_card(
        f'MILP vs. greedy {theme.tag("Synthetic", "synthetic")}',
        f'+{milp_edge:.2f}',
        f'<span style="color:{theme.OBSERVED}">&#9650; exact optimum vs. cate/cost ranking heuristic</span>',
    ), unsafe_allow_html=True)

st.write("")
left, right = st.columns(2, gap="medium")
with left:
    st.markdown(
        '<p class="ade-section-eyebrow">Does solving the exact 0/1 knapsack beat a simple heuristic?</p>'
        f'<h4 style="margin-top:0;">Total Incremental Value by Strategy {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    strategies = ["MILP (optimal)", "Greedy (cate/cost)", "Treat all eligible\n(unconstrained)"]
    values = [optimal["total_incremental_value"], baseline["total_incremental_value"], unconstrained_value]
    costs = [optimal["total_cost"], baseline["total_cost"], unconstrained_cost]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=strategies, y=values, marker_color=[theme.MODELED, theme.SLATE, theme.SYNTHETIC], text=[f"{v:.0f}" for v in values], textposition="outside"))
    fig.update_yaxes(title="Total incremental value")
    theme.apply_plotly_theme(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.caption(f"Unconstrained (treat every customer with cate_hat > 0) would need £{unconstrained_cost:,.0f} — {unconstrained_cost/budget:.1f}x the £{budget:,.0f} budget.")
with right:
    st.markdown(
        '<p class="ade-section-eyebrow">Is the optimizer actually prioritizing high-CATE customers?</p>'
        f'<h4 style="margin-top:0;">Predicted CATE — Treated vs. Not {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result.loc[result["treated"], "cate_hat"], name="Treated", marker_color=theme.MODELED, opacity=0.85, nbinsx=30))
    fig.add_trace(go.Histogram(x=result.loc[~result["treated"], "cate_hat"], name="Not treated", marker_color=theme.SLATE, opacity=0.65, nbinsx=30))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Predicted CATE (t_learner_cate)")
    fig.update_yaxes(title="Customers")
    theme.apply_plotly_theme(fig, height=300)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

st.write("")
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    st.markdown(theme.stat_card("Budget", f'£{budget:,.0f}'), unsafe_allow_html=True)
with k2:
    st.markdown(theme.stat_card("Customers treated", f'{optimal["n_treated"]:,}'), unsafe_allow_html=True)
with k3:
    avg_cost = optimal["total_cost"] / optimal["n_treated"] if optimal["n_treated"] else 0.0
    st.markdown(theme.stat_card("Avg. cost per treated customer", f'£{avg_cost:.2f}'), unsafe_allow_html=True)
with k4:
    roi = optimal["total_incremental_value"] / optimal["total_cost"] if optimal["total_cost"] else 0.0
    st.markdown(theme.stat_card("Incremental value per £ spent", f'{roi:.3f}'), unsafe_allow_html=True)

st.write("")
st.markdown('<h4 style="margin-top:0;">Top Prioritized Customers</h4>', unsafe_allow_html=True)
segment_filter = st.selectbox("Filter by segment", ["All segments"] + sorted(result["segment_name"].dropna().unique().tolist()))
treated_only = result[result["treated"]].sort_values("cate_hat", ascending=False)
if segment_filter != "All segments":
    treated_only = treated_only[treated_only["segment_name"] == segment_filter]
display = treated_only[["customer_id", "segment_name", "cate_hat", "cost"]].rename(columns={
    "customer_id": "Customer ID", "segment_name": "Segment", "cate_hat": "Predicted CATE", "cost": "Treatment Cost (£)",
})
st.dataframe(
    display.head(200), hide_index=True, use_container_width=True,
    column_config={
        "Predicted CATE": st.column_config.NumberColumn(format="%.3f"),
        "Treatment Cost (£)": st.column_config.NumberColumn(format="£%.2f"),
    },
)
st.caption(f"Showing top {min(200, len(display))} of {len(display)} treated customers, ranked by predicted CATE.")

st.markdown("---")
capacity = summary.get("segment_capacity_pct") or {}
capacity_note = ", ".join(f"{seg} ≤ {pct*100:.0f}%" for seg, pct in capacity.items()) if capacity else "none"
st.caption(f"Optimizer: scipy.optimize.milp (HiGHS) · segment capacity constraints: {capacity_note} · cost model: {alloc_cfg['cost_model']}")
