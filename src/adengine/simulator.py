"""Budget allocation simulator — Hill response curves + constrained optimization (§7).

Hill over a log-response curve because Hill saturates: audiences are finite,
so a curve that keeps climbing at unbounded spend would let the optimizer
"break" the simulator at the extremes (see ADR-005). Curve parameters are
anchored to real pipeline output — `vmax` to addressable audience size times
that channel's average calibrated propensity, `k` to the observed CPA from
metrics.py — so the simulator is not free-floating fiction even though the
underlying spend is synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from adengine.logging_conf import get_logger, log_step

logger = get_logger("simulator")


@dataclass(frozen=True)
class HillParams:
    vmax: float
    k: float
    s: float


def hill_response(spend: np.ndarray | float, p: HillParams) -> np.ndarray | float:
    spend = np.asarray(spend, dtype=float)
    spend = np.clip(spend, 0, None)
    return p.vmax * spend**p.s / (p.k**p.s + spend**p.s)


def derive_hill_params(
    channel_metrics: pd.DataFrame,
    attribution: pd.DataFrame,
    propensity_scores: pd.DataFrame,
    shape: float,
) -> dict[str, HillParams]:
    scored = attribution.merge(propensity_scores[["customer_id", "score_gbm_calibrated"]], on="customer_id")
    params = {}
    for _, row in channel_metrics.iterrows():
        channel = row["channel"]
        audience = scored[scored["channel"] == channel]
        volume = len(audience)
        avg_propensity = audience["score_gbm_calibrated"].mean()
        vmax = volume * avg_propensity
        k = max(row["cpa"], 1.0)
        params[channel] = HillParams(vmax=float(vmax), k=float(k), s=shape)
    return params


def _feasible_start(n: int, budget: float, bounds: list[tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    weights = rng.dirichlet(np.ones(n))
    x = weights * budget
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    for _ in range(20):
        x = np.clip(x, lo, hi)
        diff = budget - x.sum()
        if abs(diff) < 1e-6:
            break
        free = (x > lo) & (x < hi) if diff < 0 else (x < hi)
        if not free.any():
            break
        x[free] += diff / free.sum()
    return x


def optimize_allocation(
    params: dict[str, HillParams],
    total_budget: float,
    bounds_pct: dict[str, float],
    multistart_points: int,
    seed: int,
) -> dict:
    channels = list(params.keys())
    n = len(channels)
    bounds = [(bounds_pct["min"] * total_budget, bounds_pct["max"] * total_budget) for _ in channels]

    def neg_total_conversions(x: np.ndarray) -> float:
        return -sum(hill_response(x[i], params[c]) for i, c in enumerate(channels))

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - total_budget}]

    rng = np.random.default_rng(seed)
    best = None
    for i in range(multistart_points):
        x0 = _feasible_start(n, total_budget, bounds, rng)
        result = minimize(neg_total_conversions, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                           options={"maxiter": 200, "ftol": 1e-9})
        if result.success and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        raise RuntimeError("SLSQP failed to converge from every multistart point")

    allocation = {c: float(best.x[i]) for i, c in enumerate(channels)}
    conversions = {c: float(hill_response(best.x[i], params[c])) for i, c in enumerate(channels)}
    return {"allocation": allocation, "conversions_by_channel": conversions, "total_conversions": float(-best.fun)}


def bootstrap_scenarios(
    scored_attribution: pd.DataFrame,
    channel_metrics: pd.DataFrame,
    total_budget: float,
    bounds_pct: dict[str, float],
    shape: float,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Re-derive Hill params on resampled customers and re-optimize each time (§7.3)."""
    rng = np.random.default_rng(seed)
    channels = channel_metrics["channel"].tolist()
    cpa_by_channel = dict(zip(channel_metrics["channel"], channel_metrics["cpa"]))
    n = len(scored_attribution)

    outcomes = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        sample = scored_attribution.iloc[idx]
        params = {}
        for c in channels:
            audience = sample[sample["channel"] == c]
            volume = len(audience)
            avg_propensity = audience["score_gbm_calibrated"].mean() if volume > 0 else 0.0
            params[c] = HillParams(vmax=float(volume * avg_propensity), k=max(cpa_by_channel[c], 1.0), s=shape)
        result = optimize_allocation(params, total_budget, bounds_pct, multistart_points=3, seed=int(rng.integers(0, 1_000_000)))
        outcomes.append(result["total_conversions"])
    return pd.Series(outcomes, name="total_conversions")


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe_cfg = load_config(args.pipeline_config)
    marts_dir = Path(pipe_cfg["paths"]["marts_dir"])
    sim_cfg = cfg["simulator"]

    channel_metrics = pd.read_parquet(marts_dir / "channel_metrics.parquet")
    attribution = pd.read_parquet(marts_dir / "synthetic_attribution.parquet")
    propensity_scores = pd.read_parquet(marts_dir / "propensity_scores.parquet")

    with log_step(logger, "simulator.derive_hill_params") as rec:
        params = derive_hill_params(channel_metrics, attribution, propensity_scores, sim_cfg["hill_shape_default"])
        rec["params"] = {c: {"vmax": p.vmax, "k": p.k, "s": p.s} for c, p in params.items()}

    with log_step(logger, "simulator.optimize") as rec:
        result = optimize_allocation(
            params, sim_cfg["total_budget"], sim_cfg["channel_bounds_pct"],
            sim_cfg["optimizer"]["multistart_points"], cfg["random_state"],
        )
        rec["total_conversions"] = result["total_conversions"]

    scored_attribution = attribution.merge(propensity_scores[["customer_id", "score_gbm_calibrated"]], on="customer_id")
    with log_step(logger, "simulator.bootstrap") as rec:
        boot = bootstrap_scenarios(
            scored_attribution, channel_metrics, sim_cfg["total_budget"], sim_cfg["channel_bounds_pct"],
            sim_cfg["hill_shape_default"], sim_cfg["bootstrap_resamples"], cfg["random_state"],
        )
        rec["p10"] = float(boot.quantile(0.10))
        rec["p90"] = float(boot.quantile(0.90))

    # Response curves for plotting (spend grid per channel, 0 .. bounds max).
    curves = []
    for c, p in params.items():
        spend_grid = np.linspace(0, sim_cfg["channel_bounds_pct"]["max"] * sim_cfg["total_budget"], 60)
        conv = hill_response(spend_grid, p)
        curves.append(pd.DataFrame({"channel": c, "spend": spend_grid, "conversions": conv}))
    curves_df = pd.concat(curves, ignore_index=True)
    curves_df.to_parquet(marts_dir / "simulator_response_curves.parquet", index=False)

    boot.to_frame().to_parquet(marts_dir / "simulator_bootstrap.parquet", index=False)

    artifact = {
        "hill_params": {c: {"vmax": p.vmax, "k": p.k, "s": p.s} for c, p in params.items()},
        "optimal_allocation": result["allocation"],
        "conversions_by_channel": result["conversions_by_channel"],
        "total_conversions": result["total_conversions"],
        "bootstrap_p10": float(boot.quantile(0.10)),
        "bootstrap_p50": float(boot.quantile(0.50)),
        "bootstrap_p90": float(boot.quantile(0.90)),
        "total_budget": sim_cfg["total_budget"],
    }
    (marts_dir / "simulator_summary.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info(json.dumps({"step": "simulator.done", **artifact}))


if __name__ == "__main__":
    main()
