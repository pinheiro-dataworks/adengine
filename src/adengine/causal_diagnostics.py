"""Diagnostics for the causal-estimator comparison — ADR-008.

  covariate_balance_table   SMD before/after PSM matching, per covariate.
  overlap_check             propensity-score common-support summary.
  overlap_histogram         binned propensity-score counts for the dashboard.
  rosenbaum_sensitivity     binary-outcome Rosenbaum-bounds sensitivity sweep.
  recovery_of_truth         the central report: every estimator's ATE next to
                            the known true ATE, absolute error, and whether
                            its CI captures the truth -- every method is
                            reported, including whichever performs worst.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from adengine.causal_estimators import ATEEstimate
from adengine.contracts import causal_estimates_schema
from adengine.logging_conf import get_logger

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
