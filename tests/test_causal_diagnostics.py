"""causal_diagnostics.py is exclusively pure functions -- tested here with
small, hand-computable fixtures the way metrics.py's tests are (fixed
inputs where the expected SMD / p-value / ranking can be verified by hand).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adengine.causal_diagnostics import (
    covariate_balance_table,
    did_pretrend_summary,
    overlap_check,
    overlap_histogram,
    recovery_of_truth,
    rosenbaum_sensitivity,
)
from adengine.causal_estimators import ATEEstimate


def test_covariate_balance_table_improves_and_flags_acceptance_bar():
    full_df = pd.DataFrame({
        "x": [0.0, 0.5, -0.5, 10.0, 10.5, 9.5],  # treated (last 3) badly imbalanced vs control (first 3)
        "treated": [False, False, False, True, True, True],
    })
    matched_df = pd.DataFrame({
        "x": [0.0, 0.5, -0.5, 0.1, 0.4, -0.4],  # matched pairs, nearly identical -> good balance
        "matched_group": ["control", "control", "control", "treated", "treated", "treated"],
    })

    balance = covariate_balance_table(full_df, matched_df, ["x"], "treated")
    row = balance.iloc[0]
    assert row["covariate"] == "x"
    assert row["smd_before"] > 2.0  # huge imbalance by construction
    assert row["smd_after"] < 0.1
    assert row["improved"]
    assert row["below_acceptance_bar"]


def test_overlap_check_common_support_and_means():
    propensity = np.array([0.01, 0.5, 0.5, 0.5, 0.99])
    treatment = np.array([True, True, False, False, True])
    result = overlap_check(propensity, treatment, common_support=(0.05, 0.95))
    assert result["pct_in_common_support"] == pytest.approx(60.0)  # 3 of 5 rows in [0.05, 0.95]
    assert result["propensity_treated_mean"] == pytest.approx((0.01 + 0.5 + 0.99) / 3)
    assert result["propensity_control_mean"] == pytest.approx(0.5)
    assert result["min_propensity"] == pytest.approx(0.01)
    assert result["max_propensity"] == pytest.approx(0.99)


def test_overlap_histogram_bin_counts_sum_to_total():
    rng = np.random.default_rng(1)
    propensity = rng.uniform(0, 1, 300)
    treatment = rng.random(300) < 0.5
    hist = overlap_histogram(propensity, treatment, n_bins=10)
    assert len(hist) == 10
    assert hist["treated_count"].sum() + hist["control_count"].sum() == 300


def test_rosenbaum_sensitivity_matches_hand_computed_binomial_pvalue():
    # 10 discordant pairs, 8 favor treatment, 2 favor control.
    # At gamma=1 (p_plus=0.5): P(X>=8 | Binomial(10, 0.5)) = (C(10,8)+C(10,9)+C(10,10)) / 2^10 = 56/1024.
    matched = pd.DataFrame({
        "matched_group": ["treated"] * 10 + ["control"] * 10,
        "outcome": [1] * 8 + [0] * 2 + [0] * 8 + [1] * 2,
    })
    result = rosenbaum_sensitivity(matched, "outcome", gamma_range=[1.0])
    assert result.iloc[0]["p_value"] == pytest.approx(56 / 1024, abs=1e-6)
    assert result.iloc[0]["n_discordant_pairs"] == 10
    assert result.iloc[0]["n_favor_treatment"] == 8


def test_rosenbaum_sensitivity_p_value_is_non_decreasing_in_gamma():
    matched = pd.DataFrame({
        "matched_group": ["treated"] * 10 + ["control"] * 10,
        "outcome": [1] * 8 + [0] * 2 + [0] * 8 + [1] * 2,
    })
    result = rosenbaum_sensitivity(matched, "outcome")
    p_values = result["p_value"].to_numpy()
    assert (p_values[1:] >= p_values[:-1] - 1e-12).all()


def test_rosenbaum_sensitivity_zero_discordant_pairs_is_always_inconclusive():
    matched = pd.DataFrame({
        "matched_group": ["treated"] * 5 + ["control"] * 5,
        "outcome": [1] * 10,  # no discordant pairs at all
    })
    result = rosenbaum_sensitivity(matched, "outcome")
    assert (result["p_value"] == 1.0).all()
    assert (result["n_discordant_pairs"] == 0).all()


def _fake_estimate(method: str, ate: float, ci_low: float, ci_high: float) -> ATEEstimate:
    return ATEEstimate(method=method, ate=ate, ci_low=ci_low, ci_high=ci_high, n_treated=100, n_control=100)


def test_recovery_of_truth_ranks_by_absolute_error_and_reports_every_method():
    # Real method names on purpose -- causal_estimates_schema validates
    # `method` against the fixed 5-value set, same convention as
    # synthetic_attribution_schema's channel isin() check in contracts.py.
    true_ate = 0.10
    estimates = {
        "dml": _fake_estimate("dml", ate=0.101, ci_low=0.08, ci_high=0.12),
        "naive_diff_in_means": _fake_estimate("naive_diff_in_means", ate=0.40, ci_low=0.35, ci_high=0.45),
        "psm": _fake_estimate("psm", ate=0.13, ci_low=0.10, ci_high=0.16),
    }
    report = recovery_of_truth(estimates, true_ate)
    assert list(report["method"]) == ["dml", "psm", "naive_diff_in_means"]
    assert len(report) == 3  # the worst-performing method must still be reported, never dropped
    assert report.loc[report["method"] == "naive_diff_in_means", "ci_captures_truth"].iloc[0] == False  # noqa: E712
    assert report.loc[report["method"] == "dml", "ci_captures_truth"].iloc[0] == True  # noqa: E712


def test_recovery_of_truth_abs_error_is_exact():
    estimates = {"ipw": _fake_estimate("ipw", ate=0.25, ci_low=0.2, ci_high=0.3)}
    report = recovery_of_truth(estimates, true_ate=0.10)
    assert report.iloc[0]["abs_error"] == pytest.approx(0.15)


def test_did_pretrend_summary_aggregates_by_month_and_arm():
    panel = pd.DataFrame({
        "relative_month": [-1, -1, -1, -1, 0, 0],
        "treatment": [True, True, False, False, True, False],
        "did_outcome": [1, 0, 0, 0, 1, 1],
    })
    summary = did_pretrend_summary(panel)
    row = summary[(summary["relative_month"] == -1) & (summary["treatment"])].iloc[0]
    assert row["mean_outcome"] == pytest.approx(0.5)
    assert len(summary) == 4  # (-1, True), (-1, False), (0, True), (0, False)
