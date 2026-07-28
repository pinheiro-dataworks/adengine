"""Synthetic media-attribution layer — ADR-002.

Online Retail II carries no channel, cost, or campaign data. This module
manufactures a channel, an acquisition cost, and a campaign-exposure flag per
customer from a parameterized probabilistic rule set (configs/attribution.yaml),
conditioned on real historical monetary value so it is analytically
interesting rather than pure noise. Every downstream artifact built from this
table must carry a "synthetic" label — see metrics.py and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adengine.contracts import synthetic_attribution_schema
from adengine.logging_conf import get_logger, log_step

logger = get_logger("attribution")


def generate_synthetic_attribution(customer_features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    channels = list(cfg["channels"].keys())
    base_probs = np.array([cfg["channels"][c]["base_prob"] for c in channels])
    base_probs = base_probs / base_probs.sum()

    shift = cfg["conditioning"]["high_value_shift"]
    shifted_probs = base_probs + np.array([shift.get(c, 0.0) for c in channels])
    shifted_probs = np.clip(shifted_probs, 0.0, None)
    shifted_probs = shifted_probs / shifted_probs.sum()

    threshold = np.percentile(customer_features["monetary"], cfg["conditioning"]["high_value_percentile"])
    is_high_value = customer_features["monetary"].to_numpy() >= threshold

    n = len(customer_features)
    probs = np.where(is_high_value[:, None], shifted_probs[None, :], base_probs[None, :])
    cum_probs = probs.cumsum(axis=1)
    draws = rng.random(n)
    channel_idx = (draws[:, None] < cum_probs).argmax(axis=1)
    channel = np.array(channels)[channel_idx]

    mean_log = np.array([cfg["channels"][c]["cac_mean_log"] for c in channels])[channel_idx]
    sigma_log = np.array([cfg["channels"][c]["cac_sigma_log"] for c in channels])[channel_idx]
    cac = rng.lognormal(mean_log, sigma_log)

    exposed = rng.random(n) < cfg["campaign_exposure_rate"]

    out = pd.DataFrame({
        "customer_id": customer_features["customer_id"].to_numpy(),
        "channel": channel,
        "cac_synthetic": cac,
        "exposed": exposed,
    })
    synthetic_attribution_schema.validate(out, lazy=True)
    return out


def main() -> None:
    import argparse

    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/attribution.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    attr_cfg = load_config(args.config)
    pipe_cfg = load_config(args.pipeline_config)
    marts_dir = Path(pipe_cfg["paths"]["marts_dir"])

    customer_features = pd.read_parquet(marts_dir / "customer_features_current.parquet")

    with log_step(logger, "attribution.generate") as rec:
        attribution = generate_synthetic_attribution(customer_features, attr_cfg)
        attribution.to_parquet(marts_dir / "synthetic_attribution.parquet", index=False)
        rec["rows_out"] = len(attribution)
        rec["channel_mix"] = attribution["channel"].value_counts(normalize=True).round(3).to_dict()
        rec["exposed_rate"] = float(attribution["exposed"].mean())

    logger.info('{"step": "attribution.done"}')


if __name__ == "__main__":
    main()
