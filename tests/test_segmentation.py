"""Segmentation feeds targeting decisions (ADR-006), so the properties that
matter are: preprocessing tames the heavy tails, K selection and the ARI
stability bar behave correctly on data with known structure, and segment
naming is deterministic so a rerun never relabels the same customers.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from adengine import segmentation
from adengine.segmentation import (
    SEGMENT_ACTIONS,
    build_rfm_matrix,
    name_segments,
    run_segmentation,
    select_k,
    stability_bootstrap,
    stability_seeds,
)

CFG = {
    "random_state": 42,
    "segmentation": {
        "k_range": [2, 5],
        "chosen_k": 3,
        "stability": {"n_runs": 3, "n_bootstrap": 3, "bootstrap_frac": 0.8, "ari_acceptance_threshold": 0.85},
    },
}


def _three_group_customers(n_per_group: int = 150, seed: int = 0) -> pd.DataFrame:
    """Three well-separated behavioral groups: recent heavy buyers, lapsed
    mid-value buyers, and dormant one-off buyers."""
    rng = np.random.default_rng(seed)
    groups = [(10, 40, 8000.0), (180, 8, 900.0), (500, 1, 60.0)]
    frames = []
    for g, (recency, freq, monetary) in enumerate(groups):
        frames.append(pd.DataFrame({
            "recency_days": np.clip(rng.normal(recency, recency * 0.1 + 2, n_per_group), 0, None).astype(int),
            "frequency": np.clip(rng.normal(freq, freq * 0.1, n_per_group), 1, None).round().astype(int),
            "monetary": np.clip(rng.normal(monetary, monetary * 0.1, n_per_group), 1, None),
            "target_conversion": rng.binomial(1, [0.8, 0.3, 0.05][g], n_per_group),
            "true_group": g,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "customer_id", [f"C{i}" for i in range(len(df))])
    return df


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    return _three_group_customers()


def test_rfm_matrix_is_standardized_and_aligned(customers):
    X, raw = build_rfm_matrix(customers)
    assert X.shape == (len(customers), 3)
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-9)
    assert raw["customer_id"].tolist() == customers["customer_id"].tolist()


def test_rfm_matrix_winsorizes_recency_outliers(customers):
    with_outlier = customers.copy()
    with_outlier.loc[0, "recency_days"] = 100_000

    X, raw = build_rfm_matrix(with_outlier)
    assert raw.loc[0, "recency_days"] == 100_000  # business-unit copy is untouched
    # The outlier is clipped to the 99th percentile, so it lands at the
    # column maximum shared with other top values instead of far beyond it.
    assert X[0, 0] == pytest.approx(X[:, 0].max())
    assert (X[:, 0] >= X[0, 0] - 1e-12).sum() > 1
    # Without the clip, StandardScaler would squeeze every other customer's
    # recency into a near-constant band (std ~0.04); with it, spread survives.
    assert X[1:, 0].std() > 0.9


def test_select_k_finds_the_true_number_of_groups(customers):
    X, _ = build_rfm_matrix(customers)
    sweep = select_k(X, (2, 5), seed=42)
    assert sweep["k"].tolist() == [2, 3, 4, 5]
    assert sweep["inertia"].is_monotonic_decreasing
    assert sweep.loc[sweep["silhouette"].idxmax(), "k"] == 3


def test_stability_is_high_on_well_separated_groups(customers):
    from sklearn.cluster import KMeans

    X, _ = build_rfm_matrix(customers)
    ref = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    seed_mean, seed_min = stability_seeds(X, 3, ref, n_runs=5, seed=1)
    boot_mean, boot_min = stability_bootstrap(X, 3, ref, n_bootstrap=5, frac=0.8, seed=1)
    assert seed_min >= 0.99
    assert boot_min >= 0.95
    assert seed_mean >= seed_min and boot_mean >= boot_min


def test_stability_is_low_when_k_exceeds_real_structure():
    from sklearn.cluster import KMeans

    # Uniform noise has no cluster structure: a large K reshuffles across
    # reseeds, which is exactly what the ARI bar exists to reject.
    X = np.random.default_rng(3).uniform(size=(400, 3))
    ref = KMeans(n_clusters=8, random_state=0, n_init=1).fit_predict(X)
    seed_mean, _ = stability_seeds(X, 8, ref, n_runs=5, seed=2)
    assert seed_mean < CFG["segmentation"]["stability"]["ari_acceptance_threshold"]


def _five_cluster_profile() -> pd.DataFrame:
    return pd.DataFrame({
        "cluster": [0, 1, 2, 3, 4],
        "recency_days": [10.0, 20.0, 30.0, 200.0, 400.0],
        "frequency": [30.0, 15.0, 2.0, 12.0, 1.0],
        "monetary": [9000.0, 3000.0, 400.0, 2500.0, 80.0],
    })


def test_name_segments_assigns_the_five_rfm_tiers():
    named = name_segments(_five_cluster_profile()).set_index("cluster")["segment_name"]
    assert named.to_dict() == {
        0: "Champions",
        1: "Loyal Customers",
        2: "Potential Loyalists",
        3: "At Risk",
        4: "Hibernating",
    }


def test_name_segments_is_deterministic_and_maps_actions():
    a = name_segments(_five_cluster_profile())
    b = name_segments(_five_cluster_profile().sample(frac=1, random_state=0)).sort_values("cluster")
    assert a.set_index("cluster")["segment_name"].equals(b.set_index("cluster")["segment_name"])
    assert a["recommended_action"].tolist() == [SEGMENT_ACTIONS[n] for n in a["segment_name"]]


def test_name_segments_disambiguates_duplicates_by_monetary_rank():
    profile = pd.concat([
        _five_cluster_profile(),
        pd.DataFrame({"cluster": [5], "recency_days": [450.0], "frequency": [1.0], "monetary": [150.0]}),
    ], ignore_index=True)
    named = name_segments(profile).set_index("cluster")
    assert named.loc[5, "segment_name"] == "Hibernating 1"  # richer of the two
    assert named.loc[4, "segment_name"] == "Hibernating 2"
    assert named.loc[5, "recommended_action"] == SEGMENT_ACTIONS["Hibernating"]


def test_run_segmentation_assigns_every_customer(customers):
    result = run_segmentation(customers, CFG)
    assert result["chosen_k"] == 3
    assert result["stable"] is True
    assert len(result["assignments"]) == len(customers)
    assert result["assignments"]["segment_name"].notna().all()
    assert result["profile"]["pct_of_base"].sum() == pytest.approx(1.0)
    # Recovered clusters match the generating groups exactly (up to label order).
    crosstab = pd.crosstab(result["assignments"]["cluster"], customers["true_group"])
    assert (crosstab.gt(0).sum(axis=1) == 1).all()


def test_run_segmentation_falls_back_to_silhouette_when_k_unpinned(customers):
    cfg = {**CFG, "segmentation": {**CFG["segmentation"], "chosen_k": None}}
    assert run_segmentation(customers, cfg)["chosen_k"] == 3


def test_main_writes_segmentation_marts(tmp_path, monkeypatch, customers):
    marts = tmp_path / "marts"
    marts.mkdir()
    customers.drop(columns="true_group").to_parquet(marts / "customer_features_current.parquet", index=False)
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(json.dumps(CFG), encoding="utf-8")
    pipe_cfg = tmp_path / "pipeline.yaml"
    pipe_cfg.write_text(json.dumps({"paths": {"marts_dir": str(marts)}}), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["segmentation", "--config", str(model_cfg), "--pipeline-config", str(pipe_cfg)])
    segmentation.main()

    summary = json.loads((marts / "segmentation_summary.json").read_text(encoding="utf-8"))
    assert summary["chosen_k"] == 3
    assert len(pd.read_parquet(marts / "segment_assignments.parquet")) == len(customers)
    assert len(pd.read_parquet(marts / "segment_k_sweep.parquet")) == 4
    assert len(pd.read_parquet(marts / "segment_profiles.parquet")) == 3
