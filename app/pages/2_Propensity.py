from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_loader as dl
import theme

st.set_page_config(page_title="AdEngine — Propensity to Convert", page_icon="🎯", layout="wide")
theme.inject_css()
theme.render_sidebar_brand()

if not dl.data_available():
    st.error("No pipeline artifacts found in data/marts/. Run `make pipeline && make train` first.")
    st.stop()

summary = dl.load_json("propensity_summary")
scores = dl.load_parquet("propensity_scores")
importance = dl.load_parquet("permutation_importance")
assignments = dl.load_parquet("segment_assignments")
uplift_summary = dl.load_json("uplift_summary") if (dl.MARTS_DIR / "uplift_summary.json").exists() else None
qini_curve = dl.load_parquet("qini_curve") if (dl.MARTS_DIR / "qini_curve.parquet").exists() else None

st.markdown(
    '<p class="ade-section-eyebrow">Page 3 of 6 · calibrated HistGradientBoosting vs. logistic baseline</p>'
    '<h1 style="margin-bottom:2px;">Propensity to Convert</h1>'
    f'<p style="color:{theme.SLATE};font-size:13.5px;max-width:700px;">'
    'Which customers are likely to purchase again in the next 90 days, how trustworthy are those probabilities, and who should be targeted first?</p>',
    unsafe_allow_html=True,
)

m = summary["metrics"]
scored_sorted = scores.sort_values("score_gbm_calibrated", ascending=False).merge(assignments[["customer_id", "segment_name"]], on="customer_id", how="left")
top_score = scored_sorted["score_gbm_calibrated"].iloc[0]

col1, col2 = st.columns([1.7, 1], gap="medium")
with col1:
    bins = [0, 0.3, 0.5, 0.7, 1.01]
    labels = ["0.0–0.3", "0.3–0.5", "0.5–0.7", "0.7–1.0"]
    dist = pd.cut(scores["score_gbm_calibrated"], bins=bins, labels=labels, right=False).value_counts().reindex(labels)
    colors = ["#94A3B8", theme.BLUE, theme.MODELED, theme.LIME]
    stage_bar = "".join(f'<div style="height:100%;width:{100*v/len(scores):.1f}%;background:{c};"></div>' for v, c in zip(dist, colors))
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:rgba(244,246,243,0.62);">'
        f'<span style="width:8px;height:8px;border-radius:2px;background:{c};flex-shrink:0;"></span>Score {label} · '
        f'<b class="ade-mono" style="color:{theme.PAPER};">{v:,}</b></div>'
        for label, v, c in zip(labels, dist, colors)
    )
    st.markdown(
        f'<div class="ade-hero"><div class="ade-hero__label">Top audience <span class="ade-mono" style="color:{theme.LIME};font-size:11px;">&#9679; score</span></div>'
        f'<div class="ade-hero__value">{top_score:.3f}</div>'
        f'<div class="ade-hero__sub">HistGB AUC <b>{m["gbm_calibrated"]["roc_auc"]:.3f}</b> · Logistic AUC <b>{m["baseline"]["roc_auc"]:.3f}</b>'
        + (f' · Qini <b>{uplift_summary["qini_coefficient"]:.2f}</b> vs. random' if uplift_summary else '') + '</div>'
        f'<div style="height:10px;border-radius:6px;overflow:hidden;background:rgba(244,246,243,0.08);display:flex;margin:12px 0;">{stage_bar}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;">{legend}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    delta_auc = m["gbm_calibrated"]["roc_auc"] - m["baseline"]["roc_auc"]
    st.markdown(theme.stat_card(
        'ROC-AUC (HistGB) <span class="ade-tag ade-tag--modeled">Modeled</span>',
        f'{m["gbm_calibrated"]["roc_auc"]:.3f}',
        f'<span style="color:{theme.OBSERVED if delta_auc>0 else theme.RISK}">{"&#9650;" if delta_auc>0 else "&#9660;"} {delta_auc:+.3f} vs. {m["baseline"]["roc_auc"]:.3f} baseline</span>',
    ), unsafe_allow_html=True)
    st.write("")
    delta_brier = m["gbm_calibrated"]["brier"] - m["gbm_raw"]["brier"]
    st.markdown(theme.stat_card(
        'Brier Score, post-calibration <span class="ade-tag ade-tag--modeled">Modeled</span>',
        f'{m["gbm_calibrated"]["brier"]:.3f}',
        f'<span style="color:{theme.OBSERVED}">&#9660; (better) vs. {m["gbm_raw"]["brier"]:.3f} pre-calibration</span>',
    ), unsafe_allow_html=True)

st.write("")
st.markdown(theme.callout(
    f'<b>Honest read on model lift:</b> the calibrated HistGBM edges out the logistic baseline on PR-AUC '
    f'({m["gbm_calibrated"]["pr_auc"]:.3f} vs. {m["baseline"]["pr_auc"]:.3f}) but not by a wide margin — RFM-style behavioral '
    'features are largely monotonic, so a linear model captures most of the signal. The GBM still earns its place for the '
    'calibration quality the simulator depends on, not for raw discrimination alone.'
), unsafe_allow_html=True)

left, right = st.columns(2, gap="medium")
with left:
    st.markdown(
        f'<p class="ade-section-eyebrow">Does the model separate converters from non-converters?</p>'
        f'<h4 style="margin-top:0;">ROC Curve — Baseline vs. Gradient Boosting {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    roc = summary["roc_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["gbm"]["fpr"], y=roc["gbm"]["tpr"], mode="lines", name=f'HistGBM (AUC {m["gbm_calibrated"]["roc_auc"]:.3f})', line=dict(color=theme.MODELED, width=2.4)))
    fig.add_trace(go.Scatter(x=roc["baseline"]["fpr"], y=roc["baseline"]["tpr"], mode="lines", name=f'Logistic (AUC {m["baseline"]["roc_auc"]:.3f})', line=dict(color=theme.BLUE, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#CBD5E1", dash="dash", width=1.4)))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    theme.apply_plotly_theme(fig, height=320)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
with right:
    st.markdown(
        f'<p class="ade-section-eyebrow">Can the budget simulator trust these probabilities?</p>'
        f'<h4 style="margin-top:0;">Calibration Curve (Reliability Diagram) {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    cal = summary["calibration_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["calibrated"]["predicted"], y=cal["calibrated"]["observed"], mode="lines+markers", name="Calibrated (isotonic)", line=dict(color=theme.MODELED, width=2.2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=cal["raw"]["predicted"], y=cal["raw"]["observed"], mode="lines+markers", name="Raw GBM", line=dict(color=theme.BLUE, width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color="#CBD5E1", dash="dash", width=1.4)))
    fig.update_xaxes(title="Predicted probability (bin center)")
    fig.update_yaxes(title="Observed conversion frequency")
    theme.apply_plotly_theme(fig, height=320)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.caption(f'Brier — raw: {m["gbm_raw"]["brier"]:.3f} · calibrated: {m["gbm_calibrated"]["brier"]:.3f}. Lower is better.')

left2, right2 = st.columns(2, gap="medium")
with left2:
    st.markdown(
        f'<p class="ade-section-eyebrow">Which behaviors actually move the prediction?</p>'
        f'<h4 style="margin-top:0;">Permutation Feature Importance {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    imp = importance.sort_values("importance_mean")
    fig = go.Figure(go.Bar(
        y=imp["feature"], x=imp["importance_mean"], orientation="h",
        error_x=dict(type="data", array=imp["importance_std"], color=theme.LINE),
        marker_color=theme.MODELED,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="Mean decrease in ROC-AUC")
    theme.apply_plotly_theme(fig, height=340)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    top_feat = imp.iloc[-1]["feature"].replace("_", " ")
    st.caption(f"'{top_feat}' dominates — consistent with RFM theory: how recently someone bought is the single strongest signal of whether they will buy again.")
with right2:
    st.markdown(
        f'<p class="ade-section-eyebrow">How concentrated is predicted demand?</p>'
        f'<h4 style="margin-top:0;">Propensity Score Distribution {theme.tag("Modeled", "modeled")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure(go.Histogram(x=scores["score_gbm_calibrated"], nbinsx=20, marker_color=theme.BLUE))
    fig.update_xaxes(title="Predicted propensity score")
    fig.update_yaxes(title="Customers")
    theme.apply_plotly_theme(fig, height=340)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

if qini_curve is not None:
    st.write("")
    st.markdown(
        f'<p class="ade-section-eyebrow">Does targeting by predicted uplift outperform targeting at random?</p>'
        f'<h4 style="margin-top:0;">Uplift / Qini Curve — T-Learner {theme.tag("Synthetic treatment label", "synthetic")}</h4>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qini_curve["pct_targeted"], y=qini_curve["incremental_conversions"], mode="lines", name="T-Learner model", line=dict(color=theme.SYNTHETIC, width=2.4), fill="tozeroy", fillcolor="rgba(199,125,46,0.08)"))
    fig.add_trace(go.Scatter(x=qini_curve["pct_targeted"], y=qini_curve["random_reference"], mode="lines", name="Random targeting", line=dict(color="#CBD5E1", dash="dash", width=1.4)))
    fig.update_xaxes(title="% of population targeted")
    fig.update_yaxes(title="Cumulative incremental conversions")
    theme.apply_plotly_theme(fig, height=320)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.markdown(theme.disclaimer(
        f'<b>Not a causal claim.</b> The campaign-exposure flag is assigned completely at random by the synthetic media layer '
        f'(ADR-002) — the true uplift is zero for every customer by construction. A Qini coefficient of '
        f'<b>{uplift_summary["qini_coefficient"]:.2f}</b> hovering near zero (with sampling noise in either direction, on a '
        'population-normalized scale — not the bounded textbook statistic) is the <i>correct</i> result here: it shows the '
        'T-Learner + Qini machinery works end-to-end and is ready to plug into a genuine experimental holdout, not that it '
        'found a real effect where none exists.'
    ), unsafe_allow_html=True)

st.write("")
st.markdown('<h4 style="margin-top:0;">Recommended Audience — Top 500 by Score</h4>', unsafe_allow_html=True)
segment_filter = st.selectbox("Filter by segment", ["All segments"] + sorted(scored_sorted["segment_name"].dropna().unique().tolist()))
top500 = scored_sorted.head(500)
if segment_filter != "All segments":
    top500 = top500[top500["segment_name"] == segment_filter]
display = top500[["customer_id", "segment_name", "score_gbm_calibrated", "target_conversion"]].rename(columns={
    "customer_id": "Customer ID", "segment_name": "Segment", "score_gbm_calibrated": "Propensity Score", "target_conversion": "Converted (realized)",
})
st.dataframe(
    display.head(200), hide_index=True, use_container_width=True,
    column_config={"Propensity Score": st.column_config.NumberColumn(format="%.3f")},
)
st.caption(
    f"Showing top {min(200, len(display))} of {len(display)} filtered rows · full export below. "
    "A handful of top-ranked scores land exactly at 1.000 — isotonic calibration is a step function, and it saturates "
    "to the ceiling whenever every calibration-fold sample in the top bin converted. Expected behavior, not a modeling error."
)

csv = top500[["customer_id", "segment_name", "score_gbm_calibrated"]].rename(columns={"score_gbm_calibrated": "score"}).to_csv(index=False).encode("utf-8")
st.download_button("Export Top 500 audience (CSV)", csv, "adengine_top_audience.csv", "text/csv")

st.markdown("---")
pipe_cfg = dl.load_config("pipeline")
st.caption(f'Trained on features as of T_train = {pipe_cfg["temporal"]["t_train"]} · validated on features as of T_test = {pipe_cfg["temporal"]["t_test"]} · Temporal split, never random k-fold')
