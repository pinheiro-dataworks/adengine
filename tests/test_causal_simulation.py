"""Layer 1 (random) must be independent of covariates and Layer 2 (confounded)
must not be -- that contrast is the entire point of this module (ADR-007).
Also verifies the closed-form ground truth (true_tau) is recovered by the
Bernoulli sampling process itself, within the acceptance tolerance from
configs/causal.yaml (validation.true_effect_tolerance_pct).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adengine.causal_simulation import (
    build_causal_dataset,
    build_did_panel,
    compute_baseline_probability,
    compute_treatment_effect,
    simulate_confounded_assignment,
    simulate_random_assignment,
    true_effect_summary,
)
from adengine.segmentation import build_rfm_matrix

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


def _empty_fact_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": pd.Series(dtype=str),
        "invoice_date": pd.Series(dtype="datetime64[ns, UTC]"),
        "revenue": pd.Series(dtype=float),
    })


DID_CFG = {
    **CFG,
    "did": {
        "launch_date": "2023-05-15",
        "pre_months": 3,
        "post_months": 3,
        "trend_coef": 0.03,
        "violation_coef": 0.15,
        "violate_parallel_trends": False,
    },
}


def _pretrend_slope(panel: pd.DataFrame, treatment: bool) -> float:
    sub = panel[(~panel["is_post"]) & (panel["treatment"] == treatment)].groupby("relative_month")["did_outcome"].mean()
    return float(np.polyfit(sub.index.to_numpy(), sub.to_numpy(), 1)[0])


@pytest.fixture
def causal_dataset() -> pd.DataFrame:
    cf = _synthetic_customer_features(400, seed=1)
    seg = _synthetic_segments(cf["customer_id"], seed=2)
    return build_causal_dataset(cf, seg, CFG)


def test_layer1_assignment_is_independent_of_monetary_value(causal_dataset):
    corr = np.corrcoef(causal_dataset["layer1_treatment"].astype(float), causal_dataset["monetary_z"])[0, 1]
    assert abs(corr) < 0.1, "Layer 1 must be a randomized controlled trial by construction"


def test_layer2_assignment_correlates_with_monetary_value(causal_dataset):
    corr = np.corrcoef(causal_dataset["layer2_treatment"].astype(float), causal_dataset["monetary_z"])[0, 1]
    assert corr > 0.3, "Layer 2 must exhibit real selection bias toward high-value customers"


def test_layer2_high_value_customers_treated_more_often(causal_dataset):
    top_quartile = causal_dataset["monetary_z"] >= causal_dataset["monetary_z"].quantile(0.75)
    bottom_quartile = causal_dataset["monetary_z"] <= causal_dataset["monetary_z"].quantile(0.25)
    assert causal_dataset.loc[top_quartile, "layer2_treatment"].mean() > causal_dataset.loc[bottom_quartile, "layer2_treatment"].mean()


def test_both_layers_share_the_same_ground_truth(causal_dataset):
    # p0/p1/true_tau must not depend on which layer's treatment/outcome is observed.
    assert (causal_dataset["p1"] - causal_dataset["p0"] - causal_dataset["true_tau"]).abs().max() < 1e-9


def test_true_ate_matches_config_weighted_by_segment_mix(causal_dataset):
    summary = true_effect_summary(causal_dataset)
    base = CFG["treatment_effect"]["base_ate"]
    uplift = CFG["treatment_effect"]["segment_uplift"]
    for segment, expected_uplift in uplift.items():
        assert summary["true_cate_by_segment"][segment] == pytest.approx(base + expected_uplift, abs=1e-6)


def test_p1_is_clipped_to_valid_probability_range():
    cf = _synthetic_customer_features(200, seed=3)
    seg = _synthetic_segments(cf["customer_id"], seed=4)
    extreme_cfg = {
        **CFG,
        "treatment_effect": {"base_ate": 5.0, "segment_uplift": {s: 0.0 for s in SEGMENTS}},
    }
    ds = build_causal_dataset(cf, seg, extreme_cfg)
    assert (ds["p1"] <= 1.0).all() and (ds["p1"] >= 0.0).all()


def test_outcome_model_never_reads_real_target_conversion():
    # customer_features here has no target_conversion column at all -- if
    # build_causal_dataset needed it, this would raise KeyError.
    cf = _synthetic_customer_features(50, seed=5)
    assert "target_conversion" not in cf.columns
    seg = _synthetic_segments(cf["customer_id"], seed=6)
    build_causal_dataset(cf, seg, CFG)  # must not raise


def test_true_effect_recovered_within_tolerance_over_1000_repetitions():
    """Acceptance criterion from configs/causal.yaml: validation.true_effect_tolerance_pct."""
    cf = _synthetic_customer_features(300, seed=10)
    seg = _synthetic_segments(cf["customer_id"], seed=11)
    rfm_z, raw = build_rfm_matrix(cf)
    segment_name = raw[["customer_id"]].merge(seg, on="customer_id", how="left")["segment_name"]
    p0 = compute_baseline_probability(rfm_z, CFG)
    tau = compute_treatment_effect(segment_name, CFG)
    p1 = np.clip(p0 + tau, 0.0, 1.0)

    n_reps = CFG.get("validation", {}).get("n_repetitions", 1000)
    treated_rates, control_rates = [], []
    for rep in range(n_reps):
        treatment, outcome = simulate_random_assignment(p0, p1, seed=rep)
        treated_rates.append(outcome[treatment].mean())
        control_rates.append(outcome[~treatment].mean())

    tolerance = 0.05  # matches validation.true_effect_tolerance_pct = 5.0
    assert abs(np.mean(treated_rates) - p1.mean()) / p1.mean() < tolerance
    assert abs(np.mean(control_rates) - p0.mean()) / p0.mean() < tolerance


def test_confounded_layer_treated_arm_still_shows_higher_conversion_on_average():
    cf = _synthetic_customer_features(300, seed=12)
    seg = _synthetic_segments(cf["customer_id"], seed=13)
    rfm_z, raw = build_rfm_matrix(cf)
    segment_name = raw[["customer_id"]].merge(seg, on="customer_id", how="left")["segment_name"]
    p0 = compute_baseline_probability(rfm_z, CFG)
    tau = compute_treatment_effect(segment_name, CFG)
    p1 = np.clip(p0 + tau, 0.0, 1.0)

    n_reps = 1000
    treated_rates, control_rates = [], []
    for rep in range(n_reps):
        _, treatment, outcome = simulate_confounded_assignment(rfm_z, p0, p1, CFG, seed=rep)
        treated_rates.append(outcome[treatment].mean())
        control_rates.append(outcome[~treatment].mean())

    # Layer 2's treated/control groups are *not* representative of the whole
    # population (that's the confounding) -- so we can only check that the
    # observed rates track the population-average p1/p0 loosely, not tightly.
    assert 0.0 < np.mean(treated_rates) < 1.0
    assert 0.0 < np.mean(control_rates) < 1.0
    assert np.mean(treated_rates) > np.mean(control_rates), "treated arm should still show higher conversion on average"


@pytest.fixture(scope="module")
def did_fixtures():
    # A large N is needed here: pre-trend slope is estimated from only 3
    # monthly means per arm, so small samples make the slope comparison noisy
    # (verified empirically -- N=600 flips the sign of which slope is larger;
    # N=6000 is stable across many random seeds).
    cf = _synthetic_customer_features(6000, seed=10)
    seg = _synthetic_segments(cf["customer_id"], seed=20)
    fact = _empty_fact_transactions()
    dataset = build_causal_dataset(cf, seg, DID_CFG)
    return dataset, fact


def test_did_panel_shape_and_columns(did_fixtures):
    dataset, fact = did_fixtures
    panel = build_did_panel(dataset, fact, DID_CFG)
    pre_months, post_months = DID_CFG["did"]["pre_months"], DID_CFG["did"]["post_months"]
    assert len(panel) == len(dataset) * (pre_months + post_months)
    assert set(panel.columns) == {
        "customer_id", "segment_name", "treatment", "month", "relative_month",
        "is_post", "did_outcome", "n_real_transactions", "real_active_this_month",
    }
    assert panel["did_outcome"].isin([0, 1]).all()
    assert set(panel["relative_month"].unique()) == set(range(-pre_months, post_months))
    assert (panel["is_post"] == (panel["relative_month"] >= 0)).all()


def test_did_panel_pre_trends_are_parallel_by_default(did_fixtures):
    dataset, fact = did_fixtures
    panel = build_did_panel(dataset, fact, DID_CFG)
    slope_diff = abs(_pretrend_slope(panel, True) - _pretrend_slope(panel, False))
    assert slope_diff < 0.015


def test_did_panel_violation_flag_breaks_parallel_trends(did_fixtures):
    dataset, fact = did_fixtures
    default_panel = build_did_panel(dataset, fact, DID_CFG)
    violated_cfg = {**DID_CFG, "did": {**DID_CFG["did"], "violate_parallel_trends": True}}
    violated_panel = build_did_panel(dataset, fact, violated_cfg)

    default_diff = abs(_pretrend_slope(default_panel, True) - _pretrend_slope(default_panel, False))
    violated_diff = abs(_pretrend_slope(violated_panel, True) - _pretrend_slope(violated_panel, False))
    assert violated_diff > default_diff
    assert violated_diff > 0.015, "the violation flag must produce a detectable, not just marginally larger, break"


def test_did_panel_treatment_bump_only_applies_post_launch(did_fixtures):
    dataset, fact = did_fixtures
    panel = build_did_panel(dataset, fact, DID_CFG)

    def mean_outcome(treatment: bool, is_post: bool) -> float:
        return panel[(panel["treatment"] == treatment) & (panel["is_post"] == is_post)]["did_outcome"].mean()

    manual_did = (mean_outcome(True, True) - mean_outcome(True, False)) - (mean_outcome(False, True) - mean_outcome(False, False))
    true_ate = true_effect_summary(dataset)["true_ate"]
    assert manual_did == pytest.approx(true_ate, abs=0.03)
