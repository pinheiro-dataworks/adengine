"""Anti-leakage is the load-bearing guarantee of this module (ADR-001):
a transaction dated after `as_of` must never change a feature value, only
the (separately-computed) target. test_as_of_boundary_is_never_crossed
below is the mandatory test called for in adengine_escopo_tecnico.md §3.3.
"""
from __future__ import annotations

import pandas as pd
import pytest

from adengine.features import (
    build_customer_features,
    build_dim_customers,
    compute_behavioral,
    compute_rfm,
    compute_target,
)


def _fact(*records: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], utc=True)
    return df


@pytest.fixture
def fact_with_future_purchase() -> pd.DataFrame:
    return _fact(
        {"customer_id": "1007", "invoice": "5013", "stock_code": "10009", "invoice_date": "2023-01-18", "revenue": 22.0, "country": "United Kingdom"},
        {"customer_id": "1007", "invoice": "5014", "stock_code": "10009", "invoice_date": "2023-02-25", "revenue": 22.0, "country": "United Kingdom"},
    )


def test_as_of_boundary_is_never_crossed(fact_with_future_purchase):
    """Removing a post-as_of transaction must not change any feature value."""
    as_of = pd.Timestamp("2023-01-31", tz="UTC")

    full = fact_with_future_purchase
    truncated = full[full["invoice_date"] <= as_of]

    rfm_full = compute_rfm(full, as_of).set_index("customer_id")
    rfm_truncated = compute_rfm(truncated, as_of).set_index("customer_id")
    pd.testing.assert_frame_equal(rfm_full, rfm_truncated)

    assert rfm_full.loc["1007", "frequency"] == 1
    assert rfm_full.loc["1007", "recency_days"] == 13
    assert rfm_full.loc["1007", "monetary"] == pytest.approx(22.0)


def test_target_reflects_future_purchase_but_features_dont(fact_with_future_purchase):
    as_of = pd.Timestamp("2023-01-31", tz="UTC")
    target_with_future = compute_target(fact_with_future_purchase, as_of, window_days=90)
    assert target_with_future.set_index("customer_id").loc["1007", "target_conversion"] == 1

    truncated = fact_with_future_purchase[fact_with_future_purchase["invoice_date"] <= as_of]
    target_without_future = compute_target(truncated, as_of, window_days=90)
    assert target_without_future.set_index("customer_id").loc["1007", "target_conversion"] == 0


def test_target_window_is_exclusive_of_as_of_and_inclusive_of_horizon():
    as_of = pd.Timestamp("2023-01-31", tz="UTC")
    fact = _fact(
        {"customer_id": "A", "invoice": "1", "stock_code": "X", "invoice_date": "2023-01-31", "revenue": 5.0, "country": "UK"},
        {"customer_id": "A", "invoice": "2", "stock_code": "X", "invoice_date": "2023-01-31", "revenue": 5.0, "country": "UK"},
        {"customer_id": "B", "invoice": "3", "stock_code": "X", "invoice_date": "2023-01-31", "revenue": 5.0, "country": "UK"},
        {"customer_id": "B", "invoice": "4", "stock_code": "X", "invoice_date": "2023-05-01", "revenue": 5.0, "country": "UK"},
        {"customer_id": "C", "invoice": "5", "stock_code": "X", "invoice_date": "2023-01-31", "revenue": 5.0, "country": "UK"},
        {"customer_id": "C", "invoice": "6", "stock_code": "X", "invoice_date": "2023-05-02", "revenue": 5.0, "country": "UK"},
    )
    target = compute_target(fact, as_of, window_days=90).set_index("customer_id")
    # A: only ever bought on as_of itself -> no purchase strictly after as_of -> 0
    assert target.loc["A", "target_conversion"] == 0
    # B: repurchases exactly on the horizon boundary (as_of + 90d) -> included -> 1
    assert target.loc["B", "target_conversion"] == 1
    # C: repurchases one day past the horizon -> excluded -> 0
    assert target.loc["C", "target_conversion"] == 0


def test_single_buyer_gets_sentinel_not_imputed_mean():
    as_of = pd.Timestamp("2023-01-31", tz="UTC")
    fact = _fact(
        {"customer_id": "SOLO", "invoice": "1", "stock_code": "X", "invoice_date": "2023-01-10", "revenue": 10.0, "country": "UK"},
    )
    dim = build_dim_customers(fact)
    behavioral = compute_behavioral(fact, as_of, dim)
    row = behavioral.set_index("customer_id").loc["SOLO"]
    assert row["single_buyer"] == True  # noqa: E712
    assert pd.isna(row["avg_days_between_orders"])


def test_build_customer_features_end_to_end_schema(fact_with_future_purchase):
    as_of = pd.Timestamp("2023-01-31", tz="UTC")
    dim = build_dim_customers(fact_with_future_purchase[fact_with_future_purchase["invoice_date"] <= as_of])
    out = build_customer_features(fact_with_future_purchase, dim, as_of, window_days=90, include_target=True)
    assert set(out["customer_id"]) == {"1007"}
    assert out.iloc[0]["target_conversion"] == 1
