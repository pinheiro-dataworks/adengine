from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_loader as dl
import theme

st.set_page_config(page_title="AdEngine — Causal Identification", page_icon="🔬", layout="wide")
theme.inject_css()
theme.render_sidebar_brand()

if not dl.data_available():
    st.error("No pipeline artifacts found in data/marts/. Run `make pipeline && make train` first.")
    st.stop()

if not (dl.MARTS_DIR / "causal_summary.json").exists():
    st.error(
        "No causal-inference artifacts found in data/marts/. Run "
        "`python -m adengine.causal_simulation && python -m adengine.causal_diagnostics` first."
    )
    st.stop()

summary = dl.load_json("causal_summary")
true_effects = dl.load_json("causal_true_effects")
estimates = dl.load_parquet("causal_estimates_comparison")
balance = dl.load_parquet("causal_balance_table")
overlap_hist = dl.load_parquet("causal_overlap")
sensitivity = dl.load_parquet("causal_sensitivity")
pretrend = dl.load_parquet("causal_did_panel_summary")

METHOD_LABELS = {
    "naive_diff_in_means": "Naive diff-in-means",
    "psm": "Propensity Score Matching",
    "ipw": "Inverse Propensity Weighting",
    "did": "Difference-in-Differences",
    "dml": "Double ML",
}

st.markdown(
    '<h1 style="margin-bottom:2px;">Causal Identification</h1>'
    f'<p style="color:{theme.SLATE};font-size:13.5px;max-width:760px;">'
    'Five estimators, one confounded treatment assignment, one known ground truth: '
    'which methods actually recover the true effect once assignment is no longer random?</p>',
    unsafe_allow_html=True,
)

st.markdown(theme.disclaimer(
    "<b>Ground truth is synthetic, by design.</b> Online Retail II has no real experiment to validate "
    "an estimator against, so this page runs every estimator against a fully synthetic potential-outcomes "
    "model with a <i>known</i> true effect (ADR-007) — a standard way to validate a causal-inference "
    "estimator before trusting it on real data. Nothing here is a causal claim about real customers."
), unsafe_allow_html=True)

best_row = estimates.iloc[0]
worst_row = estimates.iloc[-1]

col1, col2 = st.columns([1.6, 1], gap="medium")
with col1:
    st.markdown(
        theme.hero_card(
            "True ATE (known by construction)",
            f'{true_effects["true_ate"]:.3f}',
            f'Confounding strength <b>{summary["confounding_strength"]}</b> · '
            f'Best recovery: <b>{METHOD_LABELS[best_row["method"]]}</b> (error {best_row["abs_error"]:.3f}) · '
            f'Worst: <b>{METHOD_LABELS[worst_row["method"]]}</b> (error {worst_row["abs_error"]:.3f})',
            live_tag="synthetic ground truth",
        ),
        unsafe_allow_html=True,
    )
with col2:
    delta = best_row["abs_error"] - worst_row["abs_error"]
    st.markdown(theme.stat_card(
        f'Best method error {theme.tag("Synthetic", "synthetic")}',
        f'{best_row["abs_error"]:.3f}',
        f'<span style="color:{theme.OBSERVED}">&#9660; {abs(delta):.3f} vs. worst ({METHOD_LABELS[worst_row["method"]]})</span>',
    ), unsafe_allow_html=True)

st.write("")
st.markdown(
    '<p class="ade-section-eyebrow">Does correcting for confounding actually move the answer closer to the truth?</p>'
    f'<h4 style="margin-top:0;">Estimated ATE by Method {theme.tag("Synthetic", "synthetic")}</h4>',
    unsafe_allow_html=True,
)
ordered = estimates.sort_values("abs_error", ascending=False)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ordered["estimated_ate"], y=[METHOD_LABELS[m] for m in ordered["method"]],
    mode="markers", marker=dict(size=12, color=theme.MODELED),
    error_x=dict(
        type="data", symmetric=False,
        array=ordered["ci_high"] - ordered["estimated_ate"],
        arrayminus=ordered["estimated_ate"] - ordered["ci_low"],
        color=theme.MODELED,
    ),
    name="Estimated ATE (95% CI)",
))
fig.add_vline(x=true_effects["true_ate"], line_dash="dash", line_color=theme.RISK, annotation_text="true ATE")
fig.update_xaxes(title="Average Treatment Effect")
theme.apply_plotly_theme(fig, height=300)
st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
st.caption(
    "Naive diff-in-means is deliberately included, not deleted — it is the biased baseline every other "
    "method is judged against. A method whose 95% CI does not cross the true ATE line is not necessarily "
    "\"wrong\": a 95% interval is expected to miss the truth some of the time."
)

left, right = st.columns(2, gap="medium")
with left:
    st.markdown(
        '<p class="ade-section-eyebrow">Did matching actually fix the covariate imbalance?</p>'
        f'<h4 style="margin-top:0;">Covariate Balance — PSM {theme.tag("Synthetic", "synthetic")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(y=balance["covariate"], x=balance["smd_before"], name="Before matching", orientation="h", marker_color=theme.SYNTHETIC))
    fig.add_trace(go.Bar(y=balance["covariate"], x=balance["smd_after"], name="After matching", orientation="h", marker_color=theme.OBSERVED))
    fig.add_vline(x=0.1, line_dash="dash", line_color=theme.RISK, annotation_text="0.1 acceptance bar")
    fig.update_xaxes(title="Standardized Mean Difference")
    theme.apply_plotly_theme(fig, height=280)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
with right:
    st.markdown(
        '<p class="ade-section-eyebrow">Is there enough common support to trust the comparison?</p>'
        f'<h4 style="margin-top:0;">Propensity Score Overlap {theme.tag("Synthetic", "synthetic")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=overlap_hist["bin_center"], y=overlap_hist["treated_count"], name="Treated", marker_color=theme.MODELED, opacity=0.85))
    fig.add_trace(go.Bar(x=overlap_hist["bin_center"], y=overlap_hist["control_count"], name="Control", marker_color=theme.SLATE, opacity=0.85))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Propensity score")
    fig.update_yaxes(title="Customers")
    theme.apply_plotly_theme(fig, height=280)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.caption(f'{summary["overlap"]["pct_in_common_support"]:.1f}% of customers fall in the [0.05, 0.95] common-support region.')

st.write("")
left2, right2 = st.columns(2, gap="medium")
with left2:
    st.markdown(
        '<p class="ade-section-eyebrow">Do treated and control move together before the launch date?</p>'
        f'<h4 style="margin-top:0;">DiD Pre/Post Trends {theme.tag("Synthetic", "synthetic")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    for treated, label, color in [(True, "Treated", theme.MODELED), (False, "Control", theme.SLATE)]:
        sub = pretrend[pretrend["treatment"] == treated].sort_values("relative_month")
        fig.add_trace(go.Scatter(x=sub["relative_month"], y=sub["mean_outcome"], mode="lines+markers", name=label, line=dict(color=color, width=2.2)))
    fig.add_vline(x=-0.5, line_dash="dot", line_color=theme.RISK, annotation_text="launch")
    fig.update_xaxes(title="Months relative to launch")
    fig.update_yaxes(title="Mean outcome")
    theme.apply_plotly_theme(fig, height=280)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    trend_status = "hold" if summary["did_parallel_trends_holds"] else "FAIL"
    st.caption(f"Pre-period parallel-trends check: {trend_status}. Reported explicitly either way — never silently assumed.")
with right2:
    st.markdown(
        '<p class="ade-section-eyebrow">How large a hidden confounder would it take to undo the PSM result?</p>'
        f'<h4 style="margin-top:0;">Rosenbaum Sensitivity {theme.tag("Synthetic", "synthetic")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sensitivity["gamma"], y=sensitivity["p_value"], mode="lines+markers", line=dict(color=theme.MODELED, width=2.2)))
    fig.add_hline(y=0.05, line_dash="dash", line_color=theme.RISK, annotation_text="p = 0.05")
    fig.update_xaxes(title="Gamma (odds of hidden bias)")
    fig.update_yaxes(title="Worst-case p-value")
    theme.apply_plotly_theme(fig, height=280)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.caption("Gamma = 1 means no hidden bias. The Gamma where the curve crosses p = 0.05 is how much unobserved confounding it would take to overturn the matched-pairs result.")

st.markdown("---")
st.caption(
    'True CATE by segment: ' + " · ".join(f"{seg} {v:+.3f}" for seg, v in true_effects["true_cate_by_segment"].items())
    + f' · confounding strength {summary["confounding_strength"]} (configs/causal.yaml)'
)
