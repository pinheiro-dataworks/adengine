"""Property-based checks on the optimizer (ADR-005): allocations
sum to budget, respect per-channel bounds, and more budget never yields
fewer conversions (Hill curves are monotonic non-decreasing by construction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adengine.simulator import (
    HillParams,
    bootstrap_scenarios,
    derive_hill_params,
    hill_response,
    optimize_allocation,
)


PARAMS = {
    "email": HillParams(vmax=400.0, k=100.0, s=1.3),
    "paid_social": HillParams(vmax=400.0, k=280.0, s=1.3),
    "paid_search": HillParams(vmax=700.0, k=330.0, s=1.3),
    "display": HillParams(vmax=200.0, k=500.0, s=1.3),
}
BOUNDS_PCT = {"min": 0.05, "max": 0.70}


def test_hill_response_is_zero_at_zero_spend():
    p = HillParams(vmax=500.0, k=100.0, s=1.3)
    assert hill_response(0.0, p) == 0.0


def test_hill_response_is_monotonic_non_decreasing():
    p = HillParams(vmax=500.0, k=100.0, s=1.3)
    spend = np.linspace(0, 10_000, 200)
    conv = hill_response(spend, p)
    assert np.all(np.diff(conv) >= -1e-9)


def test_hill_response_saturates_toward_vmax():
    p = HillParams(vmax=500.0, k=100.0, s=1.3)
    assert hill_response(1_000_000.0, p) == pytest.approx(p.vmax, rel=1e-3)


@pytest.mark.parametrize("budget", [20_000, 50_000, 100_000])
def test_optimize_allocation_sums_to_budget(budget):
    result = optimize_allocation(PARAMS, budget, BOUNDS_PCT, multistart_points=5, seed=42)
    total = sum(result["allocation"].values())
    assert total == pytest.approx(budget, rel=1e-4)


def test_optimize_allocation_respects_bounds():
    budget = 50_000
    result = optimize_allocation(PARAMS, budget, BOUNDS_PCT, multistart_points=5, seed=42)
    lo, hi = BOUNDS_PCT["min"] * budget, BOUNDS_PCT["max"] * budget
    for spend in result["allocation"].values():
        assert lo - 1e-6 <= spend <= hi + 1e-6


def test_more_budget_never_decreases_total_conversions():
    small = optimize_allocation(PARAMS, 30_000, BOUNDS_PCT, multistart_points=5, seed=42)
    large = optimize_allocation(PARAMS, 60_000, BOUNDS_PCT, multistart_points=5, seed=42)
    assert large["total_conversions"] >= small["total_conversions"] - 1e-6


def _scored_attribution() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attribution = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(6)],
        "channel": ["email", "email", "email", "display", "display", "paid_search"],
    })
    scores = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(6)],
        "score_gbm_calibrated": [0.9, 0.5, 0.1, 0.4, 0.2, 0.8],
    })
    channel_metrics = pd.DataFrame({"channel": ["email", "display", "paid_search"], "cpa": [95.0, 507.0, 0.4]})
    return attribution, scores, channel_metrics


def test_derive_hill_params_anchors_on_audience_and_observed_cpa():
    attribution, scores, channel_metrics = _scored_attribution()
    params = derive_hill_params(channel_metrics, attribution, scores, shape=1.3)

    # ADR-005: vmax = audience size x mean calibrated propensity, k = observed CPA.
    assert params["email"].vmax == pytest.approx(3 * 0.5)
    assert params["display"].vmax == pytest.approx(2 * 0.3)
    assert params["email"].k == 95.0 and params["email"].s == 1.3
    # A near-zero CPA is floored at 1 so the curve cannot saturate at zero spend.
    assert params["paid_search"].k == 1.0


def test_bootstrap_scenarios_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    n = 3000
    channels = np.array(["email", "paid_social", "paid_search", "display"])
    scored = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "channel": rng.choice(channels, n, p=[0.25, 0.25, 0.35, 0.15]),
        "score_gbm_calibrated": rng.beta(2, 5, n),
    })
    channel_metrics = pd.DataFrame({"channel": channels, "cpa": [95.0, 270.0, 330.0, 500.0]})
    budget = 50_000

    params = derive_hill_params(channel_metrics, scored[["customer_id", "channel"]], scored, shape=1.3)
    point = optimize_allocation(params, budget, BOUNDS_PCT, multistart_points=3, seed=0)["total_conversions"]
    boot = bootstrap_scenarios(scored, channel_metrics, budget, BOUNDS_PCT, shape=1.3, n_resamples=40, seed=1)

    assert len(boot) == 40
    assert boot.quantile(0.05) < point < boot.quantile(0.95)
    assert boot.std() > 0  # resampling must actually propagate uncertainty
