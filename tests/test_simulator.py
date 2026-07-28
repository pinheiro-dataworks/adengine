"""Property-based checks on the optimizer (§7.3 exit criteria): allocations
sum to budget, respect per-channel bounds, and more budget never yields
fewer conversions (Hill curves are monotonic non-decreasing by construction).
"""
from __future__ import annotations

import numpy as np
import pytest

from adengine.simulator import HillParams, hill_response, optimize_allocation


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
