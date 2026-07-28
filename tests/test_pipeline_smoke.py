"""End-to-end smoke test on a small, versioned fixture (not the real UCI
download) — proves bronze -> silver -> gold glues together and every
schema contract still holds, without touching the network or the 45MB
source file. Must stay fast and deterministic.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from adengine.cleaning import run_cleaning
from adengine.contracts import customer_features_schema, silver_schema
from adengine.features import build_customer_features, build_dim_customers

FIXTURE = Path(__file__).parent / "fixtures" / "sample_transactions.csv"

CFG = {
    "cleaning": {
        "non_product_stock_codes": ["POST", "DOT", "M", "D", "S", "B", "BANK CHARGES", "ADJUST", "AMAZONFEE", "PADS", "CRUK", "GIFT"],
    },
}


def _load_bronze() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE)
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["quantity"] = df["quantity"].astype(int)
    df["unit_price"] = df["unit_price"].astype(float)
    df["customer_id"] = df["customer_id"].astype(float)
    return df


def test_smoke_pipeline_bronze_to_gold():
    bronze = _load_bronze()
    assert len(bronze) == 15  # fixture row count — pins the fixture itself

    silver, records = run_cleaning(bronze, CFG)
    silver_schema.validate(silver, lazy=True)

    # Every documented rule fired at least once on this fixture.
    removed_by_rule = {r.rule: r.rows_removed for r in records}
    assert removed_by_rule["cancellations_c_invoices"] == 1
    assert removed_by_rule["non_product_stock_code"] == 1
    assert removed_by_rule["invalid_quantity_or_price"] == 2
    assert removed_by_rule["missing_customer_id"] == 1
    assert removed_by_rule["duplicate_line_items"] == 1

    dim_customers = build_dim_customers(silver)
    as_of = pd.Timestamp("2023-01-31", tz="UTC")
    features = build_customer_features(silver, dim_customers, as_of, window_days=90, include_target=True)
    customer_features_schema.validate(features, lazy=True)

    by_id = features.set_index("customer_id")
    # Customer 1001 bought twice by as_of and again in April -> converts.
    assert by_id.loc["1001", "target_conversion"] == 1
    assert by_id.loc["1001", "frequency"] == 2
    # Customer 1007's Feb repurchase must count toward target, not frequency.
    assert by_id.loc["1007", "target_conversion"] == 1
    assert by_id.loc["1007", "frequency"] == 1
    # Customer 1002 never repurchases -> does not convert.
    assert by_id.loc["1002", "target_conversion"] == 0
    # Customers whose only rows were invalid (1005, 1006) or missing an ID
    # never make it to silver, so they cannot appear in the feature store.
    assert "1005" not in by_id.index
    assert "1006" not in by_id.index
