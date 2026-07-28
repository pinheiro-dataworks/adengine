"""RFM segmentation — K selection, stability validation, and named profiles.

Design notes (adengine_escopo_tecnico.md §4): Monetary/Frequency are
right-skewed (wholesale-buyer tail), so K-Means (Euclidean distance,
spherical clusters) needs a log1p + light winsorization + StandardScaler
pipeline before it can find meaningful clusters. K is chosen by silhouette,
then the winning K must also clear an Adjusted Rand Index stability bar
under both reseeding and bootstrap resampling — an unstable "pretty" K is
rejected because a cluster that reshuffles every run cannot support a media
targeting decision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from adengine.logging_conf import get_logger, log_step

logger = get_logger("segmentation")

SEGMENT_ACTIONS = {
    "Champions": "VIP early access, no discount needed.",
    "Loyal Customers": "Cross-sell bundles, loyalty tier nudge.",
    "Potential Loyalists": "Second-purchase incentive within 30 days.",
    "At Risk": "Win-back email plus a moderate discount.",
    "Hibernating": "Low-cost reactivation only — cap spend per contact.",
}


def build_rfm_matrix(customer_features: pd.DataFrame, recency_winsor_pct: float = 99) -> tuple[np.ndarray, pd.DataFrame]:
    """log1p(F, M) + light winsorization on recency, then StandardScaler.

    Returns the scaled matrix and a companion frame with the raw (business-unit)
    R/F/M columns aligned by position, so profiles can be reported in £/days.
    """
    raw = customer_features[["customer_id", "recency_days", "frequency", "monetary"]].reset_index(drop=True)

    recency_cap = np.percentile(raw["recency_days"], recency_winsor_pct)
    recency_w = raw["recency_days"].clip(upper=recency_cap)

    freq_log = np.log1p(raw["frequency"])
    monetary_log = np.log1p(raw["monetary"])

    X = np.column_stack([recency_w.to_numpy(), freq_log.to_numpy(), monetary_log.to_numpy()])
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, raw


def select_k(X: np.ndarray, k_range: tuple[int, int], seed: int) -> pd.DataFrame:
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_) if k > 1 else np.nan
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def stability_seeds(X: np.ndarray, k: int, ref_labels: np.ndarray, n_runs: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for s in rng.integers(0, 1_000_000, n_runs):
        labels = KMeans(n_clusters=k, random_state=int(s), n_init=10).fit_predict(X)
        scores.append(adjusted_rand_score(ref_labels, labels))
    return float(np.mean(scores)), float(np.min(scores))


def stability_bootstrap(X: np.ndarray, k: int, ref_labels: np.ndarray, n_bootstrap: int, frac: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(X)
    sample_size = int(n * frac)
    scores = []
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=sample_size, replace=False)
        labels = KMeans(n_clusters=k, random_state=int(rng.integers(0, 1_000_000)), n_init=10).fit_predict(X[idx])
        scores.append(adjusted_rand_score(ref_labels[idx], labels))
    return float(np.mean(scores)), float(np.min(scores))


def name_segments(profile: pd.DataFrame) -> pd.DataFrame:
    """Rule-based naming off cluster-level medians (§4.4).

    Recency and frequency place each cluster in one of four behavioral
    quadrants. Within the "good recency + high frequency" quadrant, monetary
    value further splits Champions (top tier) from Loyal Customers (same
    engagement, lower spend) — the quadrant-only rule cannot tell those two
    apart on its own when more than one cluster lands there.
    """
    med_recency = profile["recency_days"].median()
    med_freq = profile["frequency"].median()

    quadrant = []
    for _, row in profile.iterrows():
        low_recency = row["recency_days"] <= med_recency
        high_freq = row["frequency"] >= med_freq
        if low_recency and high_freq:
            quadrant.append("engaged")
        elif low_recency and not high_freq:
            quadrant.append("Potential Loyalists")
        elif not low_recency and high_freq:
            quadrant.append("At Risk")
        else:
            quadrant.append("Hibernating")
    profile = profile.copy()
    profile["_quadrant"] = quadrant

    names = pd.Series(index=profile.index, dtype=object)
    engaged = profile[profile["_quadrant"] == "engaged"].sort_values("monetary", ascending=False)
    if len(engaged) > 0:
        names.loc[engaged.index[0]] = "Champions"
        for idx in engaged.index[1:]:
            names.loc[idx] = "Loyal Customers"

    other = profile[profile["_quadrant"] != "engaged"]
    names.loc[other.index] = other["_quadrant"]
    profile["segment_name"] = names
    profile = profile.drop(columns="_quadrant")

    # Disambiguate remaining duplicates (K != 5 can still yield more than one
    # cluster per label) by monetary rank within the group, richest first.
    for name, group in profile.groupby("segment_name"):
        if len(group) > 1:
            ordered = group.sort_values("monetary", ascending=False).index
            for rank, idx in enumerate(ordered, start=1):
                profile.loc[idx, "segment_name"] = f"{name} {rank}"

    profile["recommended_action"] = profile["segment_name"].apply(
        lambda n: SEGMENT_ACTIONS.get(n.rsplit(" ", 1)[0] if n.rsplit(" ", 1)[-1].isdigit() else n,
                                       SEGMENT_ACTIONS.get(n, "Monitor and re-evaluate next cycle."))
    )
    return profile


def build_segment_profiles(
    customer_features: pd.DataFrame,
    cluster_labels: np.ndarray,
    raw_rfm: pd.DataFrame,
) -> pd.DataFrame:
    df = raw_rfm.copy()
    df["cluster"] = cluster_labels
    df["target_conversion"] = customer_features["target_conversion"].to_numpy()
    df["ltv_historical"] = df["monetary"]
    df["avg_order_value"] = df["monetary"] / df["frequency"]

    n_total = len(df)
    profile = (
        df.groupby("cluster")
        .agg(
            size=("customer_id", "count"),
            recency_days=("recency_days", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            ltv_historical=("ltv_historical", "mean"),
            conversion_rate=("target_conversion", "mean"),
        )
        .reset_index()
    )
    profile["pct_of_base"] = profile["size"] / n_total
    profile = name_segments(profile)
    return profile.sort_values("monetary", ascending=False).reset_index(drop=True)


def run_segmentation(customer_features: pd.DataFrame, cfg: dict) -> dict:
    seed = cfg["random_state"]
    seg_cfg = cfg["segmentation"]
    k_range = tuple(seg_cfg["k_range"])

    X, raw_rfm = build_rfm_matrix(customer_features)

    with log_step(logger, "segmentation.select_k") as rec:
        k_sweep = select_k(X, k_range, seed)
        rec["k_range"] = list(k_range)

    chosen_k = seg_cfg.get("chosen_k")
    if not chosen_k:
        chosen_k = int(k_sweep.loc[k_sweep["silhouette"].idxmax(), "k"])

    with log_step(logger, "segmentation.fit_final") as rec:
        km = KMeans(n_clusters=chosen_k, random_state=seed, n_init=10).fit(X)
        labels = km.labels_
        rec["chosen_k"] = chosen_k

    stab_cfg = seg_cfg["stability"]
    with log_step(logger, "segmentation.stability_seeds") as rec:
        seed_mean, seed_min = stability_seeds(X, chosen_k, labels, stab_cfg["n_runs"], seed)
        rec["ari_mean"] = seed_mean
        rec["ari_min"] = seed_min

    with log_step(logger, "segmentation.stability_bootstrap") as rec:
        boot_mean, boot_min = stability_bootstrap(
            X, chosen_k, labels, stab_cfg["n_bootstrap"], stab_cfg["bootstrap_frac"], seed
        )
        rec["ari_mean"] = boot_mean
        rec["ari_min"] = boot_min

    profile = build_segment_profiles(customer_features, labels, raw_rfm)

    assignments = raw_rfm[["customer_id"]].copy()
    assignments["cluster"] = labels
    assignments = assignments.merge(profile[["cluster", "segment_name"]], on="cluster", how="left")

    return {
        "k_sweep": k_sweep,
        "chosen_k": chosen_k,
        "ari_seed_mean": seed_mean,
        "ari_seed_min": seed_min,
        "ari_bootstrap_mean": boot_mean,
        "ari_bootstrap_min": boot_min,
        "ari_threshold": stab_cfg["ari_acceptance_threshold"],
        "stable": boot_mean >= stab_cfg["ari_acceptance_threshold"],
        "profile": profile,
        "assignments": assignments,
    }


def main() -> None:
    import argparse
    import json

    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe_cfg = load_config(args.pipeline_config)
    marts_dir = Path(pipe_cfg["paths"]["marts_dir"])

    customer_features = pd.read_parquet(marts_dir / "customer_features_current.parquet")
    result = run_segmentation(customer_features, cfg)

    result["profile"].to_parquet(marts_dir / "segment_profiles.parquet", index=False)
    result["assignments"].to_parquet(marts_dir / "segment_assignments.parquet", index=False)
    result["k_sweep"].to_parquet(marts_dir / "segment_k_sweep.parquet", index=False)

    summary = {k: v for k, v in result.items() if k not in ("k_sweep", "profile", "assignments")}
    (marts_dir / "segmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(json.dumps({"step": "segmentation.done", **summary}))


if __name__ == "__main__":
    main()
