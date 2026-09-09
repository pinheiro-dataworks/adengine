"""Diagnostics for the causal-estimator comparison — ADR-008.

  covariate_balance_table   SMD before/after PSM matching, per covariate.
  overlap_check             propensity-score common-support summary.
  overlap_histogram         binned propensity-score counts for the dashboard.
  rosenbaum_sensitivity     binary-outcome Rosenbaum-bounds sensitivity sweep.
  recovery_of_truth         the central report: every estimator's ATE next to
                            the known true ATE, absolute error, and whether
                            its CI captures the truth -- every method is
                            reported, including whichever performs worst.
  did_pretrend_summary      mean outcome by relative_month x arm -- the data
                            behind the dashboard's parallel-trends chart.

main() orchestrates the full causal analysis: reads causal_simulation.py's
output, re-derives the DiD panel (cheap, deterministic -- not persisted in
full, only its pre-trend summary is), runs all five estimators from
causal_estimators.py on the confounded layer, runs every diagnostic here,
and writes the marts artifacts the dashboard's Causal Identification page reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from adengine.causal_estimators import ATEEstimate
from adengine.contracts import causal_estimates_schema
from adengine.logging_conf import get_logger, log_step

logger = get_logger("causal_diagnostics")


def _smd(df: pd.DataFrame, col: str, is_treated: pd.Series) -> float:
    treated = df.loc[is_treated, col]
    control = df.loc[~is_treated, col]
    pooled_std = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return float(abs(treated.mean() - control.mean()) / pooled_std) if pooled_std > 0 else 0.0


def covariate_balance_table(
    full_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str,
    matched_group_col: str = "matched_group",
) -> pd.DataFrame:
    """SMD per covariate, before (full population) and after (PSM matched pairs)."""
    rows = []
    for col in feature_cols:
        smd_before = _smd(full_df, col, full_df[treatment_col].astype(bool))
        smd_after = _smd(matched_df, col, matched_df[matched_group_col] == "treated")
        rows.append({
            "covariate": col,
            "smd_before": smd_before,
            "smd_after": smd_after,
            "improved": smd_after < smd_before,
            "below_acceptance_bar": smd_after < 0.1,
        })
    return pd.DataFrame(rows)


def overlap_check(propensity: np.ndarray, treatment: np.ndarray, common_support: tuple[float, float] = (0.05, 0.95)) -> dict:
    """Positivity/overlap summary -- what fraction of the sample sits in a
    propensity-score region where both treated and control units are actually observed.
    """
    lo, hi = common_support
    in_support = (propensity >= lo) & (propensity <= hi)
    return {
        "pct_in_common_support": float(in_support.mean() * 100),
        "propensity_treated_mean": float(propensity[treatment].mean()),
        "propensity_control_mean": float(propensity[~treatment].mean()),
        "min_propensity": float(propensity.min()),
        "max_propensity": float(propensity.max()),
    }


def overlap_histogram(propensity: np.ndarray, treatment: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    """Binned propensity-score counts per arm, for the dashboard's overlap chart."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(propensity, bins) - 1, 0, n_bins - 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    df = pd.DataFrame({"bin_idx": bin_idx, "treatment": treatment})
    counts = df.groupby(["bin_idx", "treatment"]).size().unstack(fill_value=0)
    counts = counts.reindex(range(n_bins), fill_value=0)
    return pd.DataFrame({
        "bin_center": bin_centers,
        "treated_count": counts.get(True, pd.Series(0, index=counts.index)).to_numpy(),
        "control_count": counts.get(False, pd.Series(0, index=counts.index)).to_numpy(),
    })


def rosenbaum_sensitivity(
    matched_df: pd.DataFrame,
    outcome_col: str,
    matched_group_col: str = "matched_group",
    gamma_range: np.ndarray | None = None,
) -> pd.DataFrame:
    """Rosenbaum-bounds sensitivity analysis for binary matched-pair outcomes.

    For each Gamma (the odds an unobserved confounder could differentially
    affect treatment assignment within a matched pair), reports the
    worst-case one-sided p-value for "no treatment effect", computed from
    discordant pairs via a binomial test -- the standard McNemar-type
    formulation of Rosenbaum bounds for binary outcomes. Gamma=1 recovers the
    ordinary matched-pairs test (no hidden bias); the sweep shows how large
    hidden confounding would need to be before the conclusion could flip.
    """
    if gamma_range is None:
        gamma_range = np.round(np.arange(1.0, 3.01, 0.25), 2)

    treated_outcome = matched_df.loc[matched_df[matched_group_col] == "treated", outcome_col].to_numpy()
    control_outcome = matched_df.loc[matched_df[matched_group_col] == "control", outcome_col].to_numpy()
    n_favor_treatment = int(((treated_outcome == 1) & (control_outcome == 0)).sum())
    n_favor_control = int(((treated_outcome == 0) & (control_outcome == 1)).sum())
    n_discordant = n_favor_treatment + n_favor_control

    rows = []
    for gamma in gamma_range:
        p_plus = gamma / (1 + gamma)
        p_value = (
            1.0 if n_discordant == 0
            else float(stats.binomtest(n_favor_treatment, n_discordant, p_plus, alternative="greater").pvalue)
        )
        rows.append({
            "gamma": float(gamma),
            "p_value": p_value,
            "n_discordant_pairs": n_discordant,
            "n_favor_treatment": n_favor_treatment,
        })
    return pd.DataFrame(rows)


def recovery_of_truth(estimates: dict[str, ATEEstimate], true_ate: float) -> pd.DataFrame:
    """The central report card: every method's ATE next to the known truth.

    Every estimator passed in is reported -- including one that performs
    worse than another -- never filtered down to only the flattering ones.
    """
    rows = []
    for method, estimate in estimates.items():
        abs_error = abs(estimate.ate - true_ate)
        ci_captures_truth = estimate.ci_low < true_ate < estimate.ci_high
        rows.append({
            "method": method,
            "estimated_ate": float(estimate.ate),
            "true_ate": float(true_ate),
            "abs_error": float(abs_error),
            "ci_low": float(estimate.ci_low),
            "ci_high": float(estimate.ci_high),
            "ci_captures_truth": bool(ci_captures_truth),
        })
    out = pd.DataFrame(rows).sort_values("abs_error").reset_index(drop=True)
    causal_estimates_schema.validate(out, lazy=True)
    return out


def did_pretrend_summary(
    panel: pd.DataFrame,
    outcome_col: str = "did_outcome",
    treatment_col: str = "treatment",
    relative_month_col: str = "relative_month",
) -> pd.DataFrame:
    """Mean outcome by relative_month x treatment arm -- the data behind the
    dashboard's parallel-trends chart (pre- and post-launch, both arms).
    """
    return (
        panel.groupby([relative_month_col, treatment_col])[outcome_col]
        .mean()
        .reset_index()
        .rename(columns={outcome_col: "mean_outcome"})
    )


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from adengine.causal_estimators import (
        _fit_propensity_scores,
        difference_in_differences,
        double_ml_ate,
        inverse_propensity_weighting,
        naive_diff_in_means,
        propensity_score_matching,
    )
    from adengine.causal_simulation import build_did_panel, true_effect_summary
    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    causal_cfg = load_config(args.config)
    pipe_cfg = load_config(args.pipeline_config)
    marts_dir = Path(pipe_cfg["paths"]["marts_dir"])

    dataset = pd.read_parquet(marts_dir / "causal_simulation.parquet")
    fact = pd.read_parquet(marts_dir / "fact_transactions.parquet")
    fact["invoice_date"] = pd.to_datetime(fact["invoice_date"], utc=True)

    true_ate = true_effect_summary(dataset)["true_ate"]
    feature_cols = ["recency_z", "frequency_z", "monetary_z"]
    seed = causal_cfg["seed"]

    with log_step(logger, "causal_diagnostics.run_estimators") as rec:
        naive = naive_diff_in_means(dataset, "layer2_outcome", "layer2_treatment")
        psm_estimate, matched = propensity_score_matching(
            dataset, feature_cols, "layer2_treatment", "layer2_outcome", seed=seed
        )
        ipw_estimate = inverse_propensity_weighting(
            dataset, feature_cols, "layer2_treatment", "layer2_outcome", seed=seed
        )
        panel = build_did_panel(dataset, fact, causal_cfg)
        did_estimate = difference_in_differences(panel)
        dml_estimate = double_ml_ate(dataset, feature_cols, "layer2_treatment", "layer2_outcome", seed=seed)
        rec["estimators_run"] = 5

    estimates = {
        "naive_diff_in_means": naive,
        "psm": psm_estimate,
        "ipw": ipw_estimate,
        "did": did_estimate,
        "dml": dml_estimate,
    }
    report = recovery_of_truth(estimates, true_ate)

    balance = covariate_balance_table(dataset, matched, feature_cols, "layer2_treatment")
    propensity = _fit_propensity_scores(dataset, feature_cols, "layer2_treatment", seed)
    treatment = dataset["layer2_treatment"].to_numpy()
    overlap = overlap_check(propensity, treatment)
    overlap_hist = overlap_histogram(propensity, treatment)
    sensitivity = rosenbaum_sensitivity(matched, "layer2_outcome")
    pretrend = did_pretrend_summary(panel)

    balance.to_parquet(marts_dir / "causal_balance_table.parquet", index=False)
    report.to_parquet(marts_dir / "causal_estimates_comparison.parquet", index=False)
    overlap_hist.to_parquet(marts_dir / "causal_overlap.parquet", index=False)
    sensitivity.to_parquet(marts_dir / "causal_sensitivity.parquet", index=False)
    pretrend.to_parquet(marts_dir / "causal_did_panel_summary.parquet", index=False)

    summary = {
        "true_ate": true_ate,
        "best_method": report.iloc[0]["method"],
        "worst_method": report.iloc[-1]["method"],
        "confounding_strength": causal_cfg["confounding"]["confounding_strength"],
        "overlap": overlap,
        "did_parallel_trends_holds": did_estimate.extra["parallel_trends_holds"],
    }
    (marts_dir / "causal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(json.dumps({"step": "causal_diagnostics.done", **summary}, default=str))


if __name__ == "__main__":
    main()
