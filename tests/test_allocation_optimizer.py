"""allocation_optimizer.py's core property (§7.3-equivalent exit criteria):
the MILP solution always respects the budget and never does worse than the
greedy baseline -- including the classic knapsack case where greedy is
provably suboptimal and the MILP still finds the true optimum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adengine.allocation_optimizer import (
    build_allocation_result,
    generate_treatment_cost,
    greedy_baseline,
    optimize_targeting,
)


def test_milp_finds_the_true_optimum_where_greedy_is_provably_suboptimal():
    # Classic 0/1 knapsack counterexample to greedy-by-ratio: A(60,10,r=6),
    # B(100,20,r=5), C(120,30,r=4), budget=50. Greedy picks A+B (cost 30,
    # value 160) and has no room left for C. The true optimum is B+C
    # (cost 50, value 220) -- greedy's ratio heuristic can't see it because
    # it never reconsiders A once picked.
    cate = np.array([60.0, 100.0, 120.0])
    cost = np.array([10.0, 20.0, 30.0])
    budget = 50.0

    optimal = optimize_targeting(cate, cost, budget)
    baseline = greedy_baseline(cate, cost, budget)

    assert optimal["total_incremental_value"] == pytest.approx(220.0)
    assert list(optimal["treat"]) == [False, True, True]
    assert baseline["total_incremental_value"] == pytest.approx(160.0)
    assert optimal["total_incremental_value"] > baseline["total_incremental_value"]


def test_milp_never_worse_than_greedy_across_random_scenarios():
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = rng.integers(5, 30)
        cate = rng.normal(0.1, 0.15, n)
        cost = rng.uniform(1, 20, n)
        budget = rng.uniform(20, 100)
        optimal = optimize_targeting(cate, cost, budget)
        baseline = greedy_baseline(cate, cost, budget)
        assert optimal["total_incremental_value"] >= baseline["total_incremental_value"] - 1e-6, f"trial {trial} failed"


@pytest.mark.parametrize("budget", [10.0, 35.0, 100.0, 500.0])
def test_milp_solution_never_exceeds_budget(budget):
    rng = np.random.default_rng(1)
    n = 25
    cate = rng.normal(0.1, 0.15, n)
    cost = rng.uniform(1, 20, n)
    result = optimize_targeting(cate, cost, budget)
    assert result["total_cost"] <= budget + 1e-6


def test_greedy_solution_never_exceeds_budget():
    rng = np.random.default_rng(2)
    cate = rng.normal(0.1, 0.15, 25)
    cost = rng.uniform(1, 20, 25)
    result = greedy_baseline(cate, cost, budget=40.0)
    assert result["total_cost"] <= 40.0 + 1e-6


def test_customers_with_non_positive_cate_are_never_treated():
    cate = np.array([-0.1, 0.0, 0.05, 0.2])
    cost = np.array([1.0, 1.0, 1.0, 1.0])
    result = optimize_targeting(cate, cost, budget=100.0)
    assert not result["treat"][0]
    assert not result["treat"][1]  # cate == 0 is not worth spending on either


def test_optimize_targeting_handles_no_eligible_customers():
    cate = np.array([-0.1, -0.2, 0.0])
    cost = np.array([1.0, 1.0, 1.0])
    result = optimize_targeting(cate, cost, budget=100.0)
    assert result["n_treated"] == 0
    assert result["total_cost"] == 0.0
    assert result["solve_status"] == "no_eligible_customers"


def test_segment_capacity_constraint_is_respected_when_binding():
    # 3 Champions with high CATE, cheap -- would consume the whole budget if
    # unconstrained, but a 20% cap forces the optimizer to leave some behind.
    cate = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
    cost = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    segment = np.array(["Champions"] * 3 + ["Other"] * 3)
    budget = 60.0

    unconstrained = optimize_targeting(cate, cost, budget, segment=segment)
    assert unconstrained["treat"][:3].all()  # all 3 Champions selected when unconstrained

    capped = optimize_targeting(cate, cost, budget, segment=segment, segment_capacity_pct={"Champions": 0.2})
    champions_spend = cost[capped["treat"] & (segment == "Champions")].sum()
    assert champions_spend <= 0.2 * budget + 1e-6
    assert not capped["treat"][:3].all()  # the cap must actually bind here


def test_generate_treatment_cost_is_positive_and_deterministic():
    cfg = {"cost_model": {"mean_log": 2.0, "sigma_log": 0.4}}
    cost_a = generate_treatment_cost(100, cfg, seed=42)
    cost_b = generate_treatment_cost(100, cfg, seed=42)
    assert (cost_a > 0).all()
    assert np.array_equal(cost_a, cost_b)


def test_build_allocation_result_schema_validates():
    result = build_allocation_result(
        customer_id=pd.Series(["1", "2", "3"]),
        segment_name=pd.Series(["Champions", "At Risk", "Hibernating"]),
        cate_hat=np.array([0.2, 0.1, -0.05]),
        cost=np.array([5.0, 6.0, 7.0]),
        treat=np.array([True, False, False]),
    )
    assert len(result) == 3
    assert result["treated"].sum() == 1
