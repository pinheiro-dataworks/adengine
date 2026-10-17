"""The budget simulator consumes propensity scores as literal probabilities,
so the model must produce valid, discriminative, calibrated scores for the
test snapshot only (ADR-004). The Qini machinery must read zero when there is
no treatment effect and turn positive when there is one -- otherwise the
uplift demo (ADR-002) could not distinguish signal from noise.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from adengine import propensity
from adengine.propensity import (
    FEATURE_COLUMNS,
    _uplift_by_decile,
    fit_uplift_t_learner,
    prepare_features,
    qini_coefficient,
    qini_curve,
    train_propensity_models,
)

# Small, fast settings: the tests check behavior, not production accuracy.
CFG = {
    "random_state": 42,
    "propensity": {
        "logistic": {"class_weight": "balanced", "max_iter": 500},
        "hist_gbm": {
            "max_iter": 60, "early_stopping": False, "validation_fraction": 0.15,
            "learning_rate": 0.1, "max_depth": 3, "l2_regularization": 0.1,
        },
        "calibration": {"method": "isotonic", "cv": 3},
        "permutation_importance": {"n_repeats": 3, "scoring": "roc_auc"},
    },
}


def _customer_features(n: int, seed: int, id_prefix: str = "C") -> pd.DataFrame:
    """Feature snapshot whose conversion is driven mainly by recency, as RFM
    theory (and the real dataset's permutation importance) says it should be."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(0, 365, n)
    frequency = rng.integers(1, 25, n)
    single = frequency == 1
    logit = 1.5 - 0.012 * recency + 0.05 * frequency
    return pd.DataFrame({
        "customer_id": [f"{id_prefix}{i}" for i in range(n)],
        "recency_days": recency,
        "frequency": frequency,
        "monetary": rng.lognormal(6, 1, n),
        "product_diversity": rng.integers(1, 40, n),
        "avg_days_between_orders": np.where(single, np.nan, rng.uniform(5, 120, n)),
        "single_buyer": single,
        "spend_trend_slope": np.where(single, np.nan, rng.normal(0, 1, n)),
        "tenure_days": recency + rng.integers(0, 400, n),
        "is_uk": rng.random(n) < 0.9,
        "target_conversion": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8"),
    })


@pytest.fixture(scope="module")
def trained() -> tuple[pd.DataFrame, dict]:
    train_df = _customer_features(1500, seed=1, id_prefix="T")
    test_df = _customer_features(800, seed=2, id_prefix="V")
    return test_df, train_propensity_models(train_df, test_df, CFG)


def test_prepare_features_derives_aov_and_encodes_flags():
    df = _customer_features(20, seed=0)
    out = prepare_features(df)
    np.testing.assert_allclose(out["avg_order_value"], df["monetary"] / df["frequency"])
    assert out["single_buyer"].dtype.kind == "i" and out["is_uk"].dtype.kind == "i"
    assert set(FEATURE_COLUMNS) <= set(out.columns)
    assert df["single_buyer"].dtype == bool  # input frame is not mutated


def test_scores_are_probabilities_for_the_test_snapshot_only(trained):
    test_df, result = trained
    scored = result["scored_test"]
    # Scores belong to the T_test snapshot; no training customer leaks in.
    assert scored["customer_id"].tolist() == test_df["customer_id"].tolist()
    for col in ("score_baseline", "score_gbm_calibrated"):
        assert scored[col].between(0, 1).all(), col
    assert scored["target_conversion"].tolist() == test_df["target_conversion"].astype(int).tolist()


def test_models_learn_the_signal_and_report_every_metric(trained):
    _, result = trained
    metrics = result["metrics"]
    assert set(metrics) == {"baseline", "gbm_raw", "gbm_calibrated"}
    for name, m in metrics.items():
        assert set(m) == {"roc_auc", "pr_auc", "brier"}, name
        assert m["roc_auc"] > 0.7, name
        assert 0 < m["brier"] < 0.25, name  # 0.25 is the Brier score of a coin flip


def test_recency_dominates_permutation_importance(trained):
    _, result = trained
    importance = result["permutation_importance"]
    assert importance.iloc[0]["feature"] == "recency_days"
    assert importance["importance_mean"].is_monotonic_decreasing


def test_curves_are_serializable_for_the_dashboard(trained):
    _, result = trained
    roc, calib = result["roc_curve"], result["calibration_curve"]
    json.dumps({"roc": roc, "calibration": calib})  # must round-trip into propensity_summary.json
    assert roc["gbm"]["fpr"][0] == 0 and roc["gbm"]["tpr"][-1] == 1
    assert len(calib["calibrated"]["predicted"]) == len(calib["calibrated"]["observed"])


def test_qini_is_near_zero_without_a_treatment_effect():
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.binomial(1, 0.3, n)
    t = rng.binomial(1, 0.5, n)
    curve = qini_curve(y, rng.random(n), t)

    assert curve.iloc[0][["pct_targeted", "incremental_conversions"]].tolist() == [0.0, 0.0]
    assert curve["pct_targeted"].iloc[-1] == 100.0
    assert curve["random_reference"].iloc[-1] == pytest.approx(curve["incremental_conversions"].iloc[-1])
    # Observed |coefficient| <= 0.27 across seeds at this n: pure sampling noise.
    assert abs(qini_coefficient(curve, n)) < 0.5


def test_qini_is_positive_when_ranking_follows_a_real_effect():
    rng = np.random.default_rng(1)
    n = 20_000
    responsive = rng.random(n) < 0.3
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, np.where(responsive & (t == 1), 0.6, 0.1))
    score = responsive + rng.normal(0, 0.1, n)  # model that ranks responders first

    informative = qini_coefficient(qini_curve(y, score, t), n)
    shuffled = qini_coefficient(qini_curve(y, rng.random(n), t), n)
    # Same outcomes, same treatment: only the ranking differs.
    assert informative > 2.0
    assert abs(shuffled) < 0.5
    by_decile = _uplift_by_decile(score, y, t)
    assert by_decile["decile"].tolist() == list(range(1, 11))
    assert by_decile.iloc[0]["uplift"] > 0.3 > by_decile.iloc[-1]["uplift"]


def test_t_learner_recovers_a_heterogeneous_effect():
    features = _customer_features(4000, seed=3)
    rng = np.random.default_rng(4)
    exposed = rng.random(len(features)) < 0.5
    # Exposure lifts conversion only for recent customers.
    lift = np.where(exposed & (features["recency_days"] < 120), 0.5, 0.0)
    base = features["target_conversion"].to_numpy() * 0.3
    features["target_conversion"] = (rng.random(len(features)) < np.clip(base + lift, 0, 1)).astype("int8")
    attribution = pd.DataFrame({"customer_id": features["customer_id"], "exposed": exposed})

    result = fit_uplift_t_learner(features, attribution, CFG)
    assert result["qini_coefficient"] > 0
    assert result["uplift_by_decile"].iloc[0]["uplift"] > result["uplift_by_decile"].iloc[-1]["uplift"]


def test_main_writes_models_scores_and_uplift_artifacts(tmp_path, monkeypatch):
    marts = tmp_path / "marts"
    marts.mkdir()
    train_df = _customer_features(600, seed=5, id_prefix="T")
    test_df = _customer_features(400, seed=6, id_prefix="V")
    train_df.to_parquet(marts / "customer_features_train.parquet", index=False)
    test_df.to_parquet(marts / "customer_features_test.parquet", index=False)
    rng = np.random.default_rng(7)
    pd.DataFrame({"customer_id": test_df["customer_id"], "exposed": rng.random(400) < 0.5}).to_parquet(
        marts / "synthetic_attribution.parquet", index=False
    )
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(json.dumps(CFG), encoding="utf-8")
    pipe_cfg = tmp_path / "pipeline.yaml"
    pipe_cfg.write_text(json.dumps({"paths": {"marts_dir": str(marts)}}), encoding="utf-8")

    monkeypatch.chdir(tmp_path)  # main() writes models/ relative to the working directory
    monkeypatch.setattr(sys, "argv", ["propensity", "--config", str(model_cfg), "--pipeline-config", str(pipe_cfg)])
    propensity.main()

    assert (tmp_path / "models" / "logistic_baseline.joblib").exists()
    assert (tmp_path / "models" / "histgbm_calibrated.joblib").exists()
    summary = json.loads((marts / "propensity_summary.json").read_text(encoding="utf-8"))
    assert summary["feature_columns"] == FEATURE_COLUMNS
    assert len(pd.read_parquet(marts / "propensity_scores.parquet")) == 400
    assert "qini_coefficient" in json.loads((marts / "uplift_summary.json").read_text(encoding="utf-8"))
    assert (marts / "qini_curve.parquet").exists() and (marts / "uplift_by_decile.parquet").exists()
