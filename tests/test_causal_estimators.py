"""Estimators are judged two ways throughout this file: against a small,
hand-computable fixture (exact arithmetic), and against a larger synthetic
confounded dataset built with causal_simulation.build_causal_dataset, where
the known true_ate from ADR-007 lets us check that each estimator actually
moves the answer closer to the truth than the naive baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adengine.causal_estimators import (
    _fit_propensity_scores,
    inverse_propensity_weighting,
    naive_diff_in_means,
    propensity_score_matching,
)
from adengine.causal_simulation import build_causal_dataset, true_effect_summary

SEGMENTS = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"]
CFG = {
    "seed": 123,
    "outcome_model": {"intercept": -0.4, "recency_coef": -0.6, "frequency_coef": 0.5, "monetary_coef": 0.4},
    "treatment_effect": {
        "base_ate": 0.08,
        "segment_uplift": {
            "Champions": -0.03,
            "Loyal Customers": 0.0,
            "Potential Loyalists": 0.05,
            "At Risk": 0.07,
            "Hibernating": 0.03,
        },
    },
    "confounding": {"confound_intercept": 0.0, "confounding_strength": 1.2},
}
FEATURE_COLS = ["recency_z", "frequency_z", "monetary_z"]


def _smd(df: pd.DataFrame, col: str, treatment_col: str) -> float:
    treated = df.loc[df[treatment_col], col]
    control = df.loc[~df[treatment_col], col]
    pooled_std = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    return float(abs(treated.mean() - control.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _synthetic_customer_features(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id": [str(i) for i in range(n)],
        "recency_days": rng.integers(1, 400, n),
        "frequency": rng.integers(1, 30, n),
        "monetary": rng.uniform(10, 5000, n),
    })


def _synthetic_segments(customer_ids: pd.Series, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"customer_id": customer_ids, "segment_name": rng.choice(SEGMENTS, len(customer_ids))})


@pytest.fixture(scope="module")
def confounded_dataset() -> pd.DataFrame:
    cf = _synthetic_customer_features(2000, seed=100)
    seg = _synthetic_segments(cf["customer_id"], seed=200)
    return build_causal_dataset(cf, seg, CFG)


def test_naive_diff_in_means_exact_arithmetic():
    df = pd.DataFrame({
        "y": [10.0, 12.0, 8.0, 20.0, 22.0, 18.0],
        "t": [False, False, False, True, True, True],
    })
    estimate = naive_diff_in_means(df, outcome_col="y", treatment_col="t")
    assert estimate.ate == pytest.approx(20.0 - 10.0)
    assert estimate.n_treated == 3
    assert estimate.n_control == 3
    assert estimate.ci_low < estimate.ate < estimate.ci_high


def test_naive_is_biased_but_psm_corrects_most_of_it(confounded_dataset):
    truth = true_effect_summary(confounded_dataset)["true_ate"]
    naive = naive_diff_in_means(confounded_dataset, "layer2_outcome", "layer2_treatment")
    psm_estimate, _ = propensity_score_matching(
        confounded_dataset, FEATURE_COLS, "layer2_treatment", "layer2_outcome", caliper=0.2, seed=42
    )
    naive_error = abs(naive.ate - truth)
    psm_error = abs(psm_estimate.ate - truth)
    assert psm_error < naive_error, "PSM must land closer to the known true ATE than the naive comparison"
    assert psm_error < 0.05


def test_naive_on_the_random_layer_is_already_close_to_truth(confounded_dataset):
    # Sanity check on ADR-007's design: Layer 1 needs no correction at all.
    truth = true_effect_summary(confounded_dataset)["true_ate"]
    naive_layer1 = naive_diff_in_means(confounded_dataset, "layer1_outcome", "layer1_treatment")
    assert abs(naive_layer1.ate - truth) < 0.05


def test_psm_matching_is_one_to_one(confounded_dataset):
    estimate, matched = propensity_score_matching(
        confounded_dataset, FEATURE_COLS, "layer2_treatment", "layer2_outcome", caliper=0.2, seed=42
    )
    assert estimate.n_treated == estimate.n_control
    assert (matched["matched_group"] == "treated").sum() == (matched["matched_group"] == "control").sum()


def test_psm_improves_covariate_balance_below_smd_threshold(confounded_dataset):
    """Acceptance bar from the extension's Definition of Done: SMD < 0.1 on
    every matched covariate, same 0.1 convention widely used for PSM balance
    (mirrors the ARI >= 0.85 acceptance-bar pattern already used for
    segmentation stability in ADR-006).
    """
    before = {col: _smd(confounded_dataset, col, "layer2_treatment") for col in FEATURE_COLS}
    assert before["monetary_z"] > 0.5, "the confounding variable must show real imbalance before matching"

    _, matched = propensity_score_matching(
        confounded_dataset, FEATURE_COLS, "layer2_treatment", "layer2_outcome", caliper=0.2, seed=42
    )
    matched_treatment_flag = matched["matched_group"] == "treated"
    matched = matched.assign(_is_treated=matched_treatment_flag)
    after = {col: _smd(matched, col, "_is_treated") for col in FEATURE_COLS}

    for col in FEATURE_COLS:
        assert after[col] < 0.1, f"post-matching SMD for {col} exceeds the 0.1 acceptance bar"
        assert after[col] < before[col]


def test_ipw_stabilized_weights_average_near_one(confounded_dataset):
    # Population-level property of the Hajek stabilization: E[weight] = 1.
    # Holds closely here because the confounded layer's propensity model is
    # well-specified and not near-separable -- see the extreme scenario below
    # for the case where finite-sample weights can drift far from 1.
    propensity = _fit_propensity_scores(confounded_dataset, FEATURE_COLS, "layer2_treatment", seed=42)
    treatment = confounded_dataset["layer2_treatment"].to_numpy()
    p_marginal = treatment.mean()
    raw_weight = np.where(treatment, p_marginal / propensity, (1 - p_marginal) / (1 - propensity))
    assert abs(raw_weight.mean() - 1.0) < 0.1


def test_ipw_truncates_extreme_weights_on_near_separable_data():
    # Craft near-perfect treatment/covariate separation with a handful of
    # crossover units -- those get propensity scores near 0/1 and therefore
    # raw weights far above the 99th-percentile cap, which is exactly the
    # scenario weight truncation exists to bound.
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    treatment = x > 0
    treatment[:5] = ~treatment[:5]
    outcome = 2.0 * treatment + 0.5 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"x": x, "t": treatment, "y": outcome})

    estimate = inverse_propensity_weighting(df, ["x"], "t", "y", weight_trunc_pct=99, seed=1)
    assert estimate.extra["n_weights_truncated"] > 0
    assert estimate.extra["max_weight_before_truncation"] > estimate.extra["weight_cap"]


def test_ipw_reduces_bias_vs_naive(confounded_dataset):
    truth = true_effect_summary(confounded_dataset)["true_ate"]
    naive = naive_diff_in_means(confounded_dataset, "layer2_outcome", "layer2_treatment")
    ipw_estimate = inverse_propensity_weighting(
        confounded_dataset, FEATURE_COLS, "layer2_treatment", "layer2_outcome", weight_trunc_pct=99, seed=42
    )
    assert abs(ipw_estimate.ate - truth) < abs(naive.ate - truth)
    assert abs(ipw_estimate.ate - truth) < 0.05
    assert ipw_estimate.ci_low < truth < ipw_estimate.ci_high
