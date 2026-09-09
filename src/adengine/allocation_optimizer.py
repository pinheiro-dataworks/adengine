"""Budget-constrained treatment targeting — ADR-009.

Given a per-customer CATE estimate (causal_estimators.t_learner_cate) and a
per-customer treatment cost, decides which customers receive treatment to
maximize total incremental value under a fixed budget: a textbook 0/1
knapsack, solved as a mixed-integer linear program via
scipy.optimize.milp (HiGHS solver, already bundled with scipy — no new
dependency, see ADR-009 for why this replaces OR-Tools).

Complements, does not replace, simulator.py's Hill-curve budget simulator:
that decides how much to spend per channel; this decides which customers,
within a channel's spend, actually receive the treatment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from adengine.logging_conf import get_logger, log_step

logger = get_logger("allocation_optimizer")


def _empty_result(n: int) -> dict:
    return {
        "treat": np.zeros(n, dtype=bool),
        "total_cost": 0.0,
        "total_incremental_value": 0.0,
        "n_treated": 0,
        "solve_status": "no_eligible_customers",
    }


def optimize_targeting(
    cate: np.ndarray,
    cost: np.ndarray,
    budget: float,
    segment: np.ndarray | None = None,
    segment_capacity_pct: dict[str, float] | None = None,
) -> dict:
    """0/1 knapsack: maximize sum(cate_i * x_i) subject to sum(cost_i * x_i)
    <= budget, x_i in {0,1}, plus one optional linear constraint per segment
    in segment_capacity_pct (spend on that segment <= pct * budget).

    Customers with non-positive predicted CATE are excluded before
    optimization -- never worth spending budget on a predicted-negative or
    zero effect, and it shrinks the problem.
    """
    n = len(cate)
    eligible = cate > 0
    if not eligible.any():
        return _empty_result(n)

    cate_eligible = cate[eligible]
    cost_eligible = cost[eligible]
    objective = -cate_eligible  # milp minimizes; negate to maximize incremental value

    rows = [cost_eligible]
    upper_bounds = [budget]
    if segment is not None and segment_capacity_pct:
        segment_eligible = segment[eligible]
        for segment_name, pct in segment_capacity_pct.items():
            row = np.where(segment_eligible == segment_name, cost_eligible, 0.0)
            rows.append(row)
            upper_bounds.append(pct * budget)

    constraint = LinearConstraint(np.vstack(rows), lb=-np.inf, ub=np.array(upper_bounds))
    bounds = Bounds(0, 1)
    integrality = np.ones(len(objective))

    result = milp(objective, constraints=constraint, bounds=bounds, integrality=integrality)
    if not result.success:
        raise RuntimeError(f"MILP allocation failed to solve: {result.message}")

    decision_eligible = result.x > 0.5
    treat = np.zeros(n, dtype=bool)
    treat[np.where(eligible)[0]] = decision_eligible

    return {
        "treat": treat,
        "total_cost": float(cost[treat].sum()),
        "total_incremental_value": float(cate[treat].sum()),
        "n_treated": int(treat.sum()),
        "solve_status": "optimal",
    }


def greedy_baseline(cate: np.ndarray, cost: np.ndarray, budget: float) -> dict:
    """Sanity-check baseline: sort by cate/cost descending, take until budget
    exhausted. Optimal for the *continuous* relaxation of the knapsack, not
    always optimal for the strict 0/1 case that optimize_targeting solves --
    the MILP solution should never do worse. Does not support segment
    capacity constraints; only comparable to optimize_targeting when called
    without segment_capacity_pct.
    """
    n = len(cate)
    eligible_idx = np.where(cate > 0)[0]
    if len(eligible_idx) == 0:
        return {"treat": np.zeros(n, dtype=bool), "total_cost": 0.0, "total_incremental_value": 0.0, "n_treated": 0, "solve_status": "greedy"}

    ratio = cate[eligible_idx] / cost[eligible_idx]
    order = eligible_idx[np.argsort(-ratio)]

    treat = np.zeros(n, dtype=bool)
    spent = 0.0
    for idx in order:
        if spent + cost[idx] <= budget:
            treat[idx] = True
            spent += cost[idx]

    return {
        "treat": treat,
        "total_cost": float(spent),
        "total_incremental_value": float(cate[treat].sum()),
        "n_treated": int(treat.sum()),
        "solve_status": "greedy",
    }


def generate_treatment_cost(n: int, cfg: dict, seed: int) -> np.ndarray:
    """Synthetic per-customer treatment cost -- log-normal, same generation
    pattern as attribution.py's CAC, deliberately on a much smaller scale
    (a retention nudge to an existing customer, not new-customer acquisition).
    """
    rng = np.random.default_rng(seed)
    cost_cfg = cfg["cost_model"]
    return rng.lognormal(cost_cfg["mean_log"], cost_cfg["sigma_log"], size=n)


def build_allocation_result(
    customer_id: pd.Series,
    segment_name: pd.Series,
    cate_hat: np.ndarray,
    cost: np.ndarray,
    treat: np.ndarray,
) -> pd.DataFrame:
    from adengine.contracts import allocation_result_schema

    out = pd.DataFrame({
        "customer_id": customer_id.to_numpy(),
        "segment_name": segment_name.to_numpy(),
        "cate_hat": cate_hat,
        "cost": cost,
        "treated": treat,
    })
    allocation_result_schema.validate(out, lazy=True)
    return out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from adengine.causal_estimators import t_learner_cate
    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/allocation.yaml")
    parser.add_argument("--causal-config", default="configs/causal.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    alloc_cfg = load_config(args.config)
    causal_cfg = load_config(args.causal_config)
    pipe_cfg = load_config(args.pipeline_config)
    marts_dir = Path(pipe_cfg["paths"]["marts_dir"])

    dataset = pd.read_parquet(marts_dir / "causal_simulation.parquet")
    feature_cols = ["recency_z", "frequency_z", "monetary_z"]
    seed = alloc_cfg["seed"]

    with log_step(logger, "allocation_optimizer.score_customers") as rec:
        cate_hat = t_learner_cate(dataset, feature_cols, "layer2_treatment", "layer2_outcome", seed=causal_cfg["seed"])
        cost = generate_treatment_cost(len(dataset), alloc_cfg, seed)
        rec["mean_cate_hat"] = float(cate_hat.mean())
        rec["mean_cost"] = float(cost.mean())

    budget = alloc_cfg["budget"]
    segment = dataset["segment_name"].to_numpy()
    capacity_pct = alloc_cfg.get("segment_capacity_pct") or None

    with log_step(logger, "allocation_optimizer.optimize") as rec:
        optimal = optimize_targeting(cate_hat, cost, budget, segment=segment, segment_capacity_pct=capacity_pct)
        rec["n_treated"] = optimal["n_treated"]
        rec["total_incremental_value"] = optimal["total_incremental_value"]

    baseline = greedy_baseline(cate_hat, cost, budget)

    result = build_allocation_result(dataset["customer_id"], dataset["segment_name"], cate_hat, cost, optimal["treat"])
    result.to_parquet(marts_dir / "allocation_result.parquet", index=False)

    summary = {
        "budget": budget,
        "optimal": {k: v for k, v in optimal.items() if k != "treat"},
        "greedy_baseline": {k: v for k, v in baseline.items() if k != "treat"},
        "milp_beats_or_matches_greedy": optimal["total_incremental_value"] >= baseline["total_incremental_value"] - 1e-6,
        "segment_capacity_pct": alloc_cfg.get("segment_capacity_pct") or {},
    }
    (marts_dir / "allocation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(json.dumps({"step": "allocation_optimizer.done", **summary}, default=str))


if __name__ == "__main__":
    main()
