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


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _baseline_logit(rfm_z: np.ndarray, cfg: dict) -> np.ndarray:
    """logit(p0(x)) — kept separate from compute_baseline_probability so the DiD
    panel (build_did_panel) can add time fixed effects at the logit scale before
    applying sigmoid, instead of recomputing this per-customer baseline from scratch.
    """
    om = cfg["outcome_model"]
    recency_z, frequency_z, monetary_z = rfm_z[:, 0], rfm_z[:, 1], rfm_z[:, 2]
    return (
        om["intercept"]
        + om["recency_coef"] * recency_z
        + om["frequency_coef"] * frequency_z
        + om["monetary_coef"] * monetary_z
    )


def compute_baseline_probability(rfm_z: np.ndarray, cfg: dict) -> np.ndarray:
    """p0(x) — untreated conversion probability, a function of standardized RFM only.

    Deliberately never reads target_conversion: keeping the ground truth fully
    synthetic and closed-form is what lets recovery-of-truth be checked exactly.
    """
    return _sigmoid(_baseline_logit(rfm_z, cfg))


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


def _real_monthly_activity(fact: pd.DataFrame, customer_ids: pd.Series, months: pd.PeriodIndex) -> pd.DataFrame:
    """Real (non-synthetic) customer x month activity flag, attached to the DiD
    panel purely for descriptive/plotting context — never fed into did_outcome.
    """
    hist = fact[fact["customer_id"].isin(customer_ids)][["customer_id", "invoice_date"]].copy()
    hist["month"] = hist["invoice_date"].dt.tz_localize(None).dt.to_period("M")
    active = hist.groupby(["customer_id", "month"]).size().rename("n_real_transactions").reset_index()
    active = active[active["month"].isin(months)]
    active["real_active_this_month"] = True
    return active[["customer_id", "month", "n_real_transactions", "real_active_this_month"]]


def build_did_panel(causal_dataset: pd.DataFrame, fact_transactions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Customer x month panel for the Difference-in-Differences estimator.

    The panel's outcome (did_outcome) is fully synthetic and shares the exact
    same true_tau as Layers 1/2: a shared linear time trend applies to both
    arms pre-launch (parallel trends by construction), and only the treated
    arm gets a treatment_effect bump post-launch. Setting
    did.violate_parallel_trends=true injects an extra pre-period slope for the
    treated arm only, so the parallel-trends diagnostic has something real to
    catch (see tests/test_causal_simulation.py).

    fact_transactions is used only to attach each customer's real transaction
    activity as a descriptive covariate (n_real_transactions,
    real_active_this_month) -- it never drives did_outcome, so this dataset's
    real Nov/Dec seasonality spike cannot leak into the synthetic effect.
    """
    did_cfg = cfg["did"]
    launch_period = pd.Timestamp(did_cfg["launch_date"]).to_period("M")
    pre_months, post_months = did_cfg["pre_months"], did_cfg["post_months"]
    trend_coef = did_cfg["trend_coef"]
    violation_coef = did_cfg["violation_coef"]
    violate = did_cfg["violate_parallel_trends"]
    seed = cfg["seed"]

    relative_months = list(range(-pre_months, post_months))
    months = pd.PeriodIndex([launch_period + m for m in relative_months])

    base = causal_dataset[["customer_id", "segment_name", "true_tau", "layer2_treatment"]].rename(
        columns={"layer2_treatment": "treatment"}
    )
    base["baseline_logit"] = _logit(causal_dataset["p0"].to_numpy())

    panel = base.merge(pd.DataFrame({"relative_month": relative_months}), how="cross")
    panel["month"] = panel["relative_month"].apply(lambda m: launch_period + m)
    panel["is_post"] = panel["relative_month"] >= 0

    violation_term = np.where(
        violate & panel["treatment"].to_numpy() & (panel["relative_month"].to_numpy() < 0),
        violation_coef * panel["relative_month"].to_numpy(),
        0.0,
    )
    logit_val = panel["baseline_logit"].to_numpy() + trend_coef * panel["relative_month"].to_numpy() + violation_term
    prob_no_bump = _sigmoid(logit_val)
    treatment_bump = panel["true_tau"].to_numpy() * panel["treatment"].to_numpy() * panel["is_post"].to_numpy()
    prob = np.clip(prob_no_bump + treatment_bump, 0.0, 1.0)

    rng = np.random.default_rng(seed + 2)  # distinct stream from Layers 1 and 2
    panel["did_outcome"] = (rng.random(len(panel)) < prob).astype("int8")

    activity = _real_monthly_activity(fact_transactions, causal_dataset["customer_id"], months)
    txn_counts = activity.set_index(["customer_id", "month"])["n_real_transactions"]
    panel_index = pd.MultiIndex.from_frame(panel[["customer_id", "month"]])
    panel["n_real_transactions"] = txn_counts.reindex(panel_index).fillna(0).astype(int).to_numpy()
    panel["real_active_this_month"] = panel["n_real_transactions"] > 0

    return panel[[
        "customer_id", "segment_name", "treatment", "month", "relative_month", "is_post",
        "did_outcome", "n_real_transactions", "real_active_this_month",
    ]]


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
