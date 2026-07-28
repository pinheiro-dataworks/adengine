"""Each DQ rule is tested in isolation against a minimal, hand-built fixture."""
from __future__ import annotations

import pandas as pd
import pytest

from adengine.cleaning import (
    drop_cancellations,
    drop_duplicate_line_items,
    drop_invalid_quantity_price,
    drop_missing_customer_id,
    drop_non_product_stock_codes,
    standardize,
)


def _rows(*records: dict) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def test_drop_cancellations_removes_c_prefixed_invoices():
    df = _rows(
        {"invoice": "5001", "stock_code": "X"},
        {"invoice": "C5002", "stock_code": "X"},
    )
    out, rec = drop_cancellations(df)
    assert list(out["invoice"]) == ["5001"]
    assert rec.rows_removed == 1
    assert rec.rows_before == 2


def test_drop_non_product_stock_codes():
    df = _rows({"stock_code": "10001"}, {"stock_code": "POST"}, {"stock_code": "M"})
    out, rec = drop_non_product_stock_codes(df, exclude_codes=["POST", "M"])
    assert list(out["stock_code"]) == ["10001"]
    assert rec.rows_removed == 2


def test_drop_invalid_quantity_price_catches_both_directions():
    df = _rows(
        {"quantity": 2, "unit_price": 5.0},
        {"quantity": 0, "unit_price": 5.0},
        {"quantity": 2, "unit_price": -1.0},
        {"quantity": -3, "unit_price": 5.0},
    )
    out, rec = drop_invalid_quantity_price(df)
    assert len(out) == 1
    assert rec.rows_removed == 3


def test_drop_missing_customer_id():
    df = _rows({"customer_id": 1001.0}, {"customer_id": float("nan")})
    out, rec = drop_missing_customer_id(df)
    assert len(out) == 1
    assert rec.rows_removed == 1


def test_drop_duplicate_line_items_keeps_one_copy():
    row = {"invoice": "5007", "stock_code": "10005", "invoice_date": pd.Timestamp("2023-01-12"),
           "customer_id": 1004.0, "quantity": 5, "unit_price": 8.0}
    df = pd.DataFrame.from_records([row, dict(row)])
    out, rec = drop_duplicate_line_items(df)
    assert len(out) == 1
    assert rec.rows_removed == 1


def test_standardize_computes_revenue_and_casts_types():
    df = _rows({
        "invoice": "5001", "stock_code": "10001", "description": "Widget",
        "quantity": 3, "unit_price": 2.5, "customer_id": 1001.0,
        "invoice_date": pd.Timestamp("2023-01-05"), "country": " United Kingdom ",
    })
    out, rec = standardize(df)
    assert out["revenue"].iloc[0] == pytest.approx(7.5)
    assert out["customer_id"].iloc[0] == "1001"
    assert isinstance(out["customer_id"].iloc[0], str)
    assert str(out["invoice_date"].dt.tz) == "UTC"
    assert out["country"].iloc[0] == "United Kingdom"
    assert rec.rows_removed == 0


def test_rule_order_matches_documented_pipeline():
    from adengine.cleaning import CLEANING_STEPS
    assert CLEANING_STEPS == [
        "cancellations_c_invoices",
        "non_product_stock_code",
        "invalid_quantity_or_price",
        "missing_customer_id",
        "duplicate_line_items",
        "standardize_types",
    ]
