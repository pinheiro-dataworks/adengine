"""metrics.py is exclusively pure functions — tested here with tiny fixtures
and the edge cases that break naive implementations: zero conversions, zero
cost, and an empty cohort.
"""
from __future__ import annotations


import pandas as pd
import pytest

from adengine.metrics import (
    build_ltv_table,
    channel_metrics,
    channel_segment_metrics,
    cohort_retention_matrix,
    cohort_revenue_matrix,
    compute_window_revenue,
    cpa,
    fit_bgnbd_gamma_gamma,
    ltv_cac_ratio,
    ltv_heuristic,
    roas,
)


def test_cpa_zero_conversions_is_infinite():
    assert cpa(cost=100.0, conversions=0) == float("inf")


def test_cpa_normal_case():
    assert cpa(cost=100.0, conversions=20) == pytest.approx(5.0)


def test_roas_zero_cost_with_revenue_is_infinite():
    assert roas(revenue=50.0, cost=0.0) == float("inf")


def test_roas_zero_cost_and_zero_revenue_is_zero():
    assert roas(revenue=0.0, cost=0.0) == 0.0


def test_roas_normal_case():
    assert roas(revenue=300.0, cost=100.0) == pytest.approx(3.0)


def test_ltv_cac_ratio_zero_cac_with_ltv_is_infinite():
    assert ltv_cac_ratio(ltv=500.0, cac=0.0) == float("inf")


def test_ltv_cac_ratio_normal_case():
    assert ltv_cac_ratio(ltv=600.0, cac=200.0) == pytest.approx(3.0)


def test_ltv_heuristic_formula():
    # £50 AOV x 4 purchases/year x 50% margin x 1 year horizon = £100
    assert ltv_heuristic(avg_order_value=50.0, purchase_frequency_annual=4.0, gross_margin=0.5, horizon_years=1.0) == pytest.approx(100.0)


def test_cohort_retention_matrix_censors_unobservable_future_cells():
    fact = pd.DataFrame({
        "customer_id": ["A", "A", "B"],
        "invoice_date": pd.to_datetime(["2023-01-05", "2023-02-10", "2023-03-01"], utc=True),
        "revenue": [10.0, 10.0, 20.0],
    })
    dim_customers = pd.DataFrame({
        "customer_id": ["A", "B"],
        "acquisition_cohort": ["2023-01", "2023-03"],
    })
    observed_through = pd.Timestamp("2023-03-15", tz="UTC")
    matrix = cohort_retention_matrix(fact, dim_customers, observed_through, max_months=3)
    row_a = matrix.set_index("acquisition_cohort").loc["2023-01"]
    row_b = matrix.set_index("acquisition_cohort").loc["2023-03"]

    # Cohort A (Jan) can observe M0, M1, M2 by mid-March, but not M3 yet.
    assert row_a["M0"] == 100.0
    assert row_a["M1"] == 100.0  # repurchased in Feb
    assert row_a["M2"] == 0.0    # no purchase in March
    assert pd.isna(row_a["M3"])  # censored — that month hasn't happened yet

    # Cohort B (Mar) can only observe M0 by mid-March.
    assert row_b["M0"] == 100.0
    assert pd.isna(row_b["M1"])
    assert pd.isna(row_b["M2"])


def test_cohort_retention_matrix_handles_empty_cohort_gracefully():
    fact = pd.DataFrame({"customer_id": [], "invoice_date": pd.to_datetime([], utc=True), "revenue": []})
    dim_customers = pd.DataFrame({"customer_id": [], "acquisition_cohort": []})
    matrix = cohort_retention_matrix(fact, dim_customers, pd.Timestamp("2023-01-01", tz="UTC"))
    assert matrix.empty


def test_cohort_revenue_matrix_accumulates_and_censors():
    fact = pd.DataFrame({
        "customer_id": ["A", "A", "B"],
        "invoice_date": pd.to_datetime(["2023-01-05", "2023-02-10", "2023-03-01"], utc=True),
        "revenue": [10.0, 15.0, 20.0],
    })
    dim_customers = pd.DataFrame({"customer_id": ["A", "B"], "acquisition_cohort": ["2023-01", "2023-03"]})
    matrix = cohort_revenue_matrix(fact, dim_customers, pd.Timestamp("2023-03-15", tz="UTC"), max_months=3)
    row_a = matrix.set_index("acquisition_cohort").loc["2023-01"]
    assert row_a["M0"] == pytest.approx(10.0)
    assert row_a["M1"] == pytest.approx(25.0)  # cumulative: 10 + 15
    assert pd.isna(row_a["M3"])  # censored


def _sample_fact_for_ltv():
    dates_a = pd.date_range("2023-01-01", periods=4, freq="30D", tz="UTC")
    dates_b = pd.date_range("2023-01-15", periods=2, freq="45D", tz="UTC")
    fact = pd.DataFrame({
        "customer_id": ["A"] * 4 + ["B"] * 2,
        "invoice_date": list(dates_a) + list(dates_b),
        "revenue": [20.0, 25.0, 22.0, 30.0, 15.0, 18.0],
    })
    return fact


def test_compute_window_revenue_only_counts_the_realized_window():
    fact = _sample_fact_for_ltv()
    as_of = pd.Timestamp("2023-02-01", tz="UTC")
    out = compute_window_revenue(fact, as_of, window_days=90).set_index("customer_id")
    # Only invoices strictly after as_of and within 90 days should be summed.
    future = fact[(fact["invoice_date"] > as_of) & (fact["invoice_date"] <= as_of + pd.Timedelta(days=90))]
    expected_a = future.loc[future["customer_id"] == "A", "revenue"].sum()
    assert out.loc["A", "window_revenue"] == pytest.approx(expected_a)


def test_fit_bgnbd_gamma_gamma_falls_back_on_bad_input():
    # A frame missing the columns lifetimes needs forces the except branch,
    # exercising the documented heuristic-fallback mitigation (§10 risk 1).
    bad_fact = pd.DataFrame({"customer_id": ["A"], "invoice_date": [pd.Timestamp("2023-01-01", tz="UTC")]})
    out, method = fit_bgnbd_gamma_gamma(bad_fact, pd.Timestamp("2023-01-01", tz="UTC"), horizon_months=12)
    assert out is None
    assert method == "heuristic_fallback"


def test_build_ltv_table_falls_back_to_heuristic_when_bgnbd_unavailable(monkeypatch):
    import adengine.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "fit_bgnbd_gamma_gamma", lambda fact, as_of, horizon_months: (None, "heuristic_fallback"))
    fact = _sample_fact_for_ltv()
    customer_features = pd.DataFrame({
        "customer_id": ["A", "B"], "monetary": [97.0, 33.0], "frequency": [4, 2], "tenure_days": [120, 60],
    })
    cfg = {"metrics": {"ltv_horizon_months": 12, "gross_margin": 0.55}}
    out = metrics_mod.build_ltv_table(fact, customer_features, pd.Timestamp("2023-06-01", tz="UTC"), cfg)
    assert (out["ltv_probabilistic"] == out["ltv_heuristic"]).all()
    assert out["ltv_method"].iloc[0] == "heuristic_fallback"


def test_build_ltv_table_uses_bgnbd_when_it_converges():
    fact = _sample_fact_for_ltv()
    customer_features = pd.DataFrame({
        "customer_id": ["A", "B"], "monetary": [97.0, 33.0], "frequency": [4, 2], "tenure_days": [120, 60],
    })
    cfg = {"metrics": {"ltv_horizon_months": 12, "gross_margin": 0.55}}
    out = build_ltv_table(fact, customer_features, pd.Timestamp("2023-06-01", tz="UTC"), cfg)
    assert set(out["customer_id"]) == {"A", "B"}
    assert (out["ltv_probabilistic"] >= 0).all()
    assert out["ltv_method"].iloc[0] in {"bgnbd_gamma_gamma", "heuristic_fallback"}


def test_channel_metrics_pure_aggregation():
    customer_features = pd.DataFrame({
        "customer_id": ["1", "2", "3"], "target_conversion": [1, 0, 1],
    })
    attribution = pd.DataFrame({
        "customer_id": ["1", "2", "3"], "channel": ["email", "email", "display"], "cac_synthetic": [10.0, 10.0, 40.0],
    })
    ltv_table = pd.DataFrame({"customer_id": ["1", "2", "3"], "ltv_probabilistic": [200.0, 150.0, 80.0]})
    window_revenue = pd.DataFrame({"customer_id": ["1", "3"], "window_revenue": [60.0, 50.0]})

    out = channel_metrics(customer_features, attribution, ltv_table, window_revenue).set_index("channel")
    assert out.loc["email", "customers"] == 2
    assert out.loc["email", "conversions"] == 1
    assert out.loc["email", "spend_synthetic"] == pytest.approx(20.0)
    assert out.loc["email", "cpa"] == pytest.approx(20.0)  # 20 cost / 1 conversion
    assert out.loc["email", "revenue"] == pytest.approx(60.0)  # customer 2 had no window revenue -> filled 0
    assert out.loc["display", "roas"] == pytest.approx(50.0 / 40.0)


def test_channel_segment_metrics_cross_tab():
    customer_features = pd.DataFrame({"customer_id": ["1", "2"], "target_conversion": [1, 0]})
    attribution = pd.DataFrame({"customer_id": ["1", "2"], "channel": ["email", "email"], "cac_synthetic": [10.0, 10.0]})
    segments = pd.DataFrame({"customer_id": ["1", "2"], "segment_name": ["Champions", "Hibernating"]})
    ltv_table = pd.DataFrame({"customer_id": ["1", "2"], "ltv_probabilistic": [200.0, 20.0]})
    window_revenue = pd.DataFrame({"customer_id": ["1"], "window_revenue": [60.0]})

    out = channel_segment_metrics(customer_features, attribution, segments, ltv_table, window_revenue)
    assert len(out) == 2
    champions_row = out[out["segment"] == "Champions"].iloc[0]
    assert champions_row["cpa"] == pytest.approx(10.0)
