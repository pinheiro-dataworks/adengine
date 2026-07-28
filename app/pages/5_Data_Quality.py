from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_loader as dl
import theme

st.set_page_config(page_title="AdEngine — Pipeline & Data Quality", page_icon="🔍", layout="wide")
theme.inject_css()
theme.render_sidebar_brand()

if not dl.data_available():
    st.error("No pipeline artifacts found in data/marts/. Run `make pipeline && make train` first.")
    st.stop()

dq = dl.load_dq_report()
manifest = dl.load_manifest()
fact = dl.load_parquet("fact_transactions")
dim_customers = dl.load_parquet("dim_customers")

st.markdown(
    '<p class="ade-section-eyebrow">Page 6 of 6 · medallion architecture</p>'
    '<h1 style="margin-bottom:2px;">Pipeline &amp; Data Quality</h1>'
    f'<p style="color:{theme.SLATE};font-size:13.5px;max-width:680px;">'
    'What happened to the raw data before any chart on this dashboard could be trusted?</p>',
    unsafe_allow_html=True,
)

rules_df = pd.DataFrame(dq["rules"])
row_rules = rules_df[rules_df["rows_removed"] > 0]

col1, col2 = st.columns([1.7, 1], gap="medium")
with col1:
    colors = theme.CATEGORICAL
    stage_bar = "".join(f'<div style="height:100%;width:{100*r/dq["rows_raw"]:.2f}%;background:{colors[i%len(colors)]};"></div>' for i, r in enumerate(row_rules["rows_removed"]))
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:rgba(244,246,243,0.62);">'
        f'<span style="width:8px;height:8px;border-radius:2px;background:{colors[i%len(colors)]};flex-shrink:0;"></span>{rule.replace("_"," ").title()} · '
        f'<b class="ade-mono" style="color:{theme.PAPER};">{removed:,}</b></div>'
        for i, (rule, removed) in enumerate(zip(row_rules["rule"], row_rules["rows_removed"]))
    )
    st.markdown(
        f'<div class="ade-hero"><div class="ade-hero__label">Rows retained '
        f'<span class="ade-mono" style="color:{theme.LIME};font-size:11px;">&#9679; bronze &rarr; silver</span></div>'
        f'<div class="ade-hero__value">{dq["retention_pct"]:.1f}%</div>'
        f'<div class="ade-hero__sub">{len(row_rules)} documented rules with row impact · <b>{dq["rows_raw"]-dq["rows_silver"]:,}</b> rows removed of <b>{dq["rows_raw"]:,}</b></div>'
        f'<div style="height:10px;border-radius:6px;overflow:hidden;background:rgba(244,246,243,0.08);display:flex;margin:12px 0;">{stage_bar}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:14px;">{legend}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(theme.stat_card('Bronze Rows <span class="ade-tag ade-tag--observed">Observed</span>', f'{dq["rows_raw"]:,}'), unsafe_allow_html=True)
    st.write("")
    st.markdown(theme.stat_card('Silver Rows <span class="ade-tag ade-tag--observed">Observed</span>', f'{dq["rows_silver"]:,}'), unsafe_allow_html=True)

st.write("")
st.markdown('<h4 style="margin-top:0;">Silver-layer Cleaning Rules, in Applied Order</h4>', unsafe_allow_html=True)
display_rules = rules_df.rename(columns={
    "rule": "Rule", "rationale": "Rationale", "rows_before": "Rows Before", "rows_removed": "Rows Removed", "pct_of_input": "% of Input",
})
display_rules["Rule"] = display_rules["Rule"].str.replace("_", " ").str.title()
st.dataframe(display_rules[["Rule", "Rows Before", "Rows Removed", "% of Input", "Rationale"]], hide_index=True, use_container_width=True)

csv = display_rules.to_csv(index=False).encode("utf-8")
st.download_button("Download DQ report (CSV)", csv, "adengine_dq_report.csv", "text/csv")

st.write("")
st.markdown(
    f'<p class="ade-section-eyebrow">Gold contract, enforced at write time</p>'
    f'<h4 style="margin-top:0;">customer_features — Schema Contract (pandera) {theme.tag("Enforced", "modeled")}</h4>',
    unsafe_allow_html=True,
)
contract_rows = [
    ("customer_id", "str", "unique, not null"),
    ("recency_days", "int", ">= 0"),
    ("frequency", "int", ">= 1"),
    ("monetary", "float", "> 0"),
    ("product_diversity", "int", ">= 1"),
    ("avg_days_between_orders", "float", ">= 0, nullable (single-buyer sentinel)"),
    ("single_buyer", "bool", "not null"),
    ("spend_trend_slope", "float", "nullable"),
    ("tenure_days", "int", ">= 0"),
    ("is_uk", "bool", "not null"),
    ("target_conversion", "Int8", "isin [0, 1] — withheld from the feature store at inference time"),
]
code_lines = "\n".join(f"{name:<26}{dtype:<10}{rule}" for name, dtype, rule in contract_rows)
st.code(f"# validated on every pipeline run — build fails on any violation\n{code_lines}", language="text")

st.write("")
st.markdown('<h4 style="margin-top:0;">Architecture Decision Records</h4>', unsafe_allow_html=True)
adr_col1, adr_col2 = st.columns(2, gap="medium")
adrs = [
    ("ADR-001 — Temporal Anti-Leakage Design", "Accepted",
     "Naive random train/test splits let future purchase behavior leak into features, inflating offline metrics beyond what production would see.",
     "Two fixed cutoffs — T_train and T_test — with a strict 90-day target window; every feature function takes an explicit as_of and only reads transactions up to it.",
     "Fewer usable training examples than k-fold CV would allow, in exchange for an honest estimate of real-world performance."),
    ("ADR-002 — Synthetic Media Attribution", "Accepted",
     "Online Retail II has no channel, cost, or campaign fields, but CPA / ROAS / uplift logic still needs real data to exercise it end-to-end.",
     "Generate channel, cost, and exposure fields via a probabilistic rule set parameterized in YAML, conditioned on real historical monetary value; label every derived output as synthetic in the UI and docs.",
     "Media-metric numbers are directional demonstrations of the engine, not real performance claims — swapping in a real ad-platform export requires new config, not new code."),
]
for col, (title, status, context, decision, tradeoff) in zip([adr_col1, adr_col2], adrs):
    with col:
        st.markdown(
            f'<div class="ade-callout" style="border-left-color:{theme.OBSERVED};">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<b style="color:{theme.INK};">{title}</b>'
            f'<span class="ade-tag ade-tag--observed">{status}</span></div>'
            f'<div style="font-size:12.5px;margin-bottom:6px;"><b>Context</b> — {context}</div>'
            f'<div style="font-size:12.5px;margin-bottom:6px;"><b>Decision</b> — {decision}</div>'
            f'<div style="font-size:12.5px;"><b>Trade-off</b> — {tradeoff}</div></div>',
            unsafe_allow_html=True,
        )

if manifest:
    st.write("")
    st.markdown('<h4 style="margin-top:0;">Source Lineage</h4>', unsafe_allow_html=True)
    st.json(manifest, expanded=False)

st.markdown("---")
st.caption("Reproducible from a clean checkout via `python -m adengine.cleaning && python -m adengine.features && ...` (see Makefile) · Global seed fixed for determinism")
