"""Confounded-assignment causal simulation — ADR-007.

Online Retail II carries no experiment or campaign data with a known causal
effect. This module builds a fully synthetic potential-outcomes model on top
of `customer_features_current` and `segment_assignments`, then observes it
under two different assignment mechanisms:

  Layer 1 (random):     T ~ Bernoulli(0.5), independent of X — a randomized
                         controlled trial by construction.
  Layer 2 (confounded):  P(T=1|X) increases with monetary value — higher-value
                         customers are preferentially targeted, the most
                         common and realistic selection bias in marketing.

Both layers observe the *exact same* p0(x)/p1(x)/tau(x) functions, so the true
ATE and the true CATE-by-segment are identical and known in both layers — only
Layer 2 requires an estimator that corrects for confounding. That symmetry is
what makes recovery-of-truth (causal_diagnostics.py) a meaningful test: a
naive comparison is unbiased on Layer 1 and biased on Layer 2 by construction,
not by accident.

Every artifact this module produces is synthetic and must carry the
"Synthetic" label wherever it surfaces, same convention as attribution.py
(ADR-002).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adengine.contracts import causal_simulation_schema
from adengine.logging_conf import get_logger, log_step
from adengine.segmentation import build_rfm_matrix

logger = get_logger("causal_simulation")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_baseline_probability(rfm_z: np.ndarray, cfg: dict) -> np.ndarray:
    """p0(x) — untreated conversion probability, a function of standardized RFM only.

    Deliberately never reads target_conversion: keeping the ground truth fully
    synthetic and closed-form is what lets recovery-of-truth be checked exactly.
    """
    om = cfg["outcome_model"]
    recency_z, frequency_z, monetary_z = rfm_z[:, 0], rfm_z[:, 1], rfm_z[:, 2]
    logit = (
        om["intercept"]
        + om["recency_coef"] * recency_z
        + om["frequency_coef"] * frequency_z
        + om["monetary_coef"] * monetary_z
    )
    return _sigmoid(logit)


def compute_treatment_effect(segment_names: pd.Series, cfg: dict) -> np.ndarray:
    """tau(x) = base_ate + segment_uplift[segment] — heterogeneous by RFM segment.

    Mirrors the diminishing-returns logic already used by the Hill response
    curves in simulator.py: near-ceiling segments (Champions) get less
    incremental lift, segments with the most room to move get more.
    """
    te = cfg["treatment_effect"]
    base = te["base_ate"]
    uplift_map = te["segment_uplift"]
    return segment_names.map(lambda s: base + uplift_map.get(s, 0.0)).to_numpy(dtype=float)


def simulate_random_assignment(p0: np.ndarray, p1: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Layer 1 — T ~ Bernoulli(0.5), independent of X. An RCT by construction."""
    rng = np.random.default_rng(seed)
    treatment = rng.random(len(p0)) < 0.5
    prob = np.where(treatment, p1, p0)
    outcome = (rng.random(len(p0)) < prob).astype("int8")
    return treatment, outcome


def simulate_confounded_assignment(
    rfm_z: np.ndarray, p0: np.ndarray, p1: np.ndarray, cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Layer 2 — P(T=1|x) depends on monetary value. Same p0/p1 as Layer 1; only
    the assignment mechanism changes, which is exactly what an observational
    estimator (PSM/IPW/DiD/DML) has to correct for.
    """
    conf = cfg["confounding"]
    monetary_z = rfm_z[:, 2]
    logit = conf["confound_intercept"] + conf["confounding_strength"] * monetary_z
    propensity = _sigmoid(logit)
    rng = np.random.default_rng(seed + 1)  # distinct stream from Layer 1
    treatment = rng.random(len(p0)) < propensity
    prob = np.where(treatment, p1, p0)
    outcome = (rng.random(len(p0)) < prob).astype("int8")
    return propensity, treatment, outcome


def build_causal_dataset(customer_features: pd.DataFrame, segment_assignments: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Builds the full two-layer synthetic causal dataset — one row per customer."""
    seed = cfg["seed"]
    rfm_z, raw_rfm = build_rfm_matrix(customer_features)
    customer_id = raw_rfm["customer_id"].reset_index(drop=True)

    segments = customer_id.to_frame().merge(
        segment_assignments[["customer_id", "segment_name"]], on="customer_id", how="left"
    )
    segment_name = segments["segment_name"]

    p0 = compute_baseline_probability(rfm_z, cfg)
    true_tau = compute_treatment_effect(segment_name, cfg)
    p1 = np.clip(p0 + true_tau, 0.0, 1.0)

    layer1_treatment, layer1_outcome = simulate_random_assignment(p0, p1, seed)
    layer2_propensity, layer2_treatment, layer2_outcome = simulate_confounded_assignment(rfm_z, p0, p1, cfg, seed)

    out = pd.DataFrame({
        "customer_id": customer_id,
        "segment_name": segment_name,
        "recency_z": rfm_z[:, 0].astype(float),
        "frequency_z": rfm_z[:, 1].astype(float),
        "monetary_z": rfm_z[:, 2].astype(float),
        "p0": p0.astype(float),
        "p1": p1.astype(float),
        "true_tau": true_tau.astype(float),
        "layer1_treatment": layer1_treatment.astype(bool),
        "layer1_outcome": layer1_outcome,
        "layer2_propensity": layer2_propensity.astype(float),
        "layer2_treatment": layer2_treatment.astype(bool),
        "layer2_outcome": layer2_outcome,
    })
    out["layer1_outcome"] = out["layer1_outcome"].astype("Int8")
    out["layer2_outcome"] = out["layer2_outcome"].astype("Int8")

    causal_simulation_schema.validate(out, lazy=True)
    return out


def true_effect_summary(causal_dataset: pd.DataFrame) -> dict:
    """The known ground truth every estimator in causal_estimators.py is validated against."""
    overall_ate = float(causal_dataset["true_tau"].mean())
    by_segment = causal_dataset.groupby("segment_name")["true_tau"].mean().round(4).to_dict()
    return {"true_ate": round(overall_ate, 4), "true_cate_by_segment": by_segment}


def main() -> None:
    import argparse
    import json

    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe_cfg = load_config(args.pipeline_config)
    marts_dir = Path(pipe_cfg["paths"]["marts_dir"])

    customer_features = pd.read_parquet(marts_dir / "customer_features_current.parquet")
    segment_assignments = pd.read_parquet(marts_dir / "segment_assignments.parquet")

    with log_step(logger, "causal_simulation.build_dataset") as rec:
        dataset = build_causal_dataset(customer_features, segment_assignments, cfg)
        dataset.to_parquet(marts_dir / "causal_simulation.parquet", index=False)
        rec["rows_out"] = len(dataset)

    summary = true_effect_summary(dataset)
    (marts_dir / "causal_true_effects.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(json.dumps({"step": "causal_simulation.done", **summary}))


if __name__ == "__main__":
    main()
