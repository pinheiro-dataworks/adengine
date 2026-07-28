"""Pandera schemas enforced at every medallion boundary.

Each schema fails loudly and reports every violation at once (not just the
first), which is why pandera was chosen over hand-rolled assertions — see
adengine_escopo_tecnico.md §2.3. `strict=True` means an unexpected column is
itself a contract violation, not a silent pass-through.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema

bronze_schema = DataFrameSchema(
    {
        "invoice": Column(str, nullable=False),
        "stock_code": Column(str, nullable=False),
        "description": Column(str, nullable=True),
        "quantity": Column(int, nullable=False),
        "invoice_date": Column("datetime64[ns]", nullable=False),
        "unit_price": Column(float, nullable=False),
        "customer_id": Column(float, nullable=True),
        "country": Column(str, nullable=False),
    },
    strict=True,
    coerce=False,
)

silver_schema = DataFrameSchema(
    {
        "invoice": Column(str, nullable=False),
        "stock_code": Column(str, nullable=False),
        "description": Column(str, nullable=True),
        "quantity": Column(int, Check.gt(0)),
        "unit_price": Column(float, Check.gt(0)),
        "customer_id": Column(str, nullable=False),
        "invoice_date": Column("datetime64[ns, UTC]"),
        "revenue": Column(float, Check.gt(0)),
        "country": Column(str, nullable=False),
    },
    # A stock_code can legitimately repeat within one invoice at a different
    # price/quantity (e.g. a promo-priced top-up line) — the composite key
    # matches the exact one used by drop_duplicate_line_items in cleaning.py.
    unique=["invoice", "stock_code", "invoice_date", "customer_id", "quantity", "unit_price"],
    strict=True,
    coerce=False,
)

fact_transactions_schema = silver_schema

dim_customers_schema = DataFrameSchema(
    {
        "customer_id": Column(str, nullable=False, unique=True),
        "first_purchase_date": Column("datetime64[ns, UTC]"),
        "acquisition_cohort": Column(str, nullable=False),   # "YYYY-MM"
        "country": Column(str, nullable=False),
    },
    strict=True,
    coerce=False,
)

customer_features_schema = DataFrameSchema(
    {
        "customer_id": Column(str, nullable=False, unique=True),
        "as_of": Column("datetime64[ns, UTC]"),
        "recency_days": Column(int, Check.ge(0)),
        "frequency": Column(int, Check.ge(1)),
        "monetary": Column(float, Check.gt(0)),
        "product_diversity": Column(int, Check.ge(1)),
        "avg_days_between_orders": Column(float, Check.ge(0), nullable=True),
        "single_buyer": Column(bool, nullable=False),
        "spend_trend_slope": Column(float, nullable=True),
        "tenure_days": Column(int, Check.ge(0)),
        "is_uk": Column(bool, nullable=False),
        "target_conversion": Column(pa.Int8, Check.isin([0, 1]), nullable=True),
    },
    strict=True,
    coerce=False,
)

synthetic_attribution_schema = DataFrameSchema(
    {
        "customer_id": Column(str, nullable=False, unique=True),
        "channel": Column(str, Check.isin(["paid_search", "paid_social", "email", "display"])),
        "cac_synthetic": Column(float, Check.gt(0)),
        "exposed": Column(bool, nullable=False),
    },
    strict=True,
    coerce=False,
)
