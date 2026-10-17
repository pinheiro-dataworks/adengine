"""The synthetic attribution layer (ADR-002) must be reproducible from its
seed, honor its contract, and actually implement the conditioning it
documents: high-value customers are shifted toward paid search. A layer
that silently ignored its config would make every media metric meaningless.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from adengine import attribution
from adengine.attribution import generate_synthetic_attribution
from adengine.contracts import synthetic_attribution_schema

CFG = {
    "seed": 42,
    "channels": {
        "paid_search": {"base_prob": 0.35, "cac_mean_log": 4.74, "cac_sigma_log": 0.35},
        "paid_social": {"base_prob": 0.25, "cac_mean_log": 4.43, "cac_sigma_log": 0.35},
        "email": {"base_prob": 0.25, "cac_mean_log": 3.38, "cac_sigma_log": 0.30},
        "display": {"base_prob": 0.15, "cac_mean_log": 4.93, "cac_sigma_log": 0.40},
    },
    "conditioning": {
        "high_value_percentile": 75,
        "high_value_shift": {"paid_search": 0.10, "email": -0.05, "display": -0.05},
    },
    "campaign_exposure_rate": 0.5,
}


def _customers(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "monetary": rng.lognormal(6, 1, n),
    })


@pytest.fixture(scope="module")
def large_attribution() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = _customers(20_000)
    return customers, generate_synthetic_attribution(customers, CFG)


def test_output_honors_contract_and_preserves_customers(large_attribution):
    customers, out = large_attribution
    synthetic_attribution_schema.validate(out, lazy=True)
    assert out["customer_id"].tolist() == customers["customer_id"].tolist()


def test_same_seed_reproduces_and_new_seed_changes_output():
    customers = _customers(500)
    a = generate_synthetic_attribution(customers, CFG)
    b = generate_synthetic_attribution(customers, CFG)
    c = generate_synthetic_attribution(customers, {**CFG, "seed": 7})
    pd.testing.assert_frame_equal(a, b)
    assert not a["channel"].equals(c["channel"])


def test_high_value_customers_shift_toward_paid_search(large_attribution):
    customers, out = large_attribution
    threshold = np.percentile(customers["monetary"], 75)
    high = out[customers["monetary"].to_numpy() >= threshold]
    low = out[customers["monetary"].to_numpy() < threshold]

    share = lambda df, c: (df["channel"] == c).mean()  # noqa: E731
    # Configured shift: +0.10 paid_search, -0.05 email (before renormalization).
    assert share(high, "paid_search") - share(low, "paid_search") == pytest.approx(0.10, abs=0.03)
    assert share(high, "email") < share(low, "email")


def test_exposure_rate_and_cac_ordering_follow_config(large_attribution):
    _, out = large_attribution
    assert out["exposed"].mean() == pytest.approx(CFG["campaign_exposure_rate"], abs=0.02)
    median_cac = out.groupby("channel")["cac_synthetic"].median()
    # Lognormal median = exp(mean_log): email is the cheapest channel, display the most expensive.
    assert median_cac.idxmin() == "email"
    assert median_cac.idxmax() == "display"


def test_main_writes_attribution_mart(tmp_path, monkeypatch):
    marts = tmp_path / "marts"
    marts.mkdir()
    _customers(300).to_parquet(marts / "customer_features_current.parquet", index=False)
    attr_cfg = tmp_path / "attribution.yaml"
    attr_cfg.write_text(json.dumps(CFG), encoding="utf-8")  # JSON is valid YAML
    pipe_cfg = tmp_path / "pipeline.yaml"
    pipe_cfg.write_text(json.dumps({"paths": {"marts_dir": str(marts)}}), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["attribution", "--config", str(attr_cfg), "--pipeline-config", str(pipe_cfg)])
    attribution.main()

    written = pd.read_parquet(marts / "synthetic_attribution.parquet")
    assert len(written) == 300
    synthetic_attribution_schema.validate(written, lazy=True)
