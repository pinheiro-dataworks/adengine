"""Gold layer — dimensional model, RFM, behavioral features, and the target.

Every feature function takes an explicit `as_of: pd.Timestamp` and only ever
reads transactions with `invoice_date <= as_of`. This is the anti-leakage
contract from ADR-001: it is enforced here, not just documented, and
tests/test_features.py asserts that a transaction dated after `as_of` cannot
change any feature value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adengine.contracts import customer_features_schema, dim_customers_schema
from adengine.logging_conf import get_logger, log_step

logger = get_logger("features")


def build_dim_customers(fact: pd.DataFrame) -> pd.DataFrame:
    """One row per customer who ever transacted — a stable master dimension.

    Using each customer's true first purchase (not clipped to any as_of) is
    safe: this table is only ever joined onto customer_features rows that
    already require activity by as_of, so a customer's first purchase can
    never postdate the cutoff being used downstream.
    """
    g = fact.sort_values("invoice_date").groupby("customer_id")
    dim = g.agg(
        first_purchase_date=("invoice_date", "first"),
        country=("country", lambda s: s.mode().iat[0]),
    ).reset_index()
    dim["acquisition_cohort"] = dim["first_purchase_date"].dt.strftime("%Y-%m")
    dim_customers_schema.validate(dim, lazy=True)
    return dim


def compute_rfm(fact: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    hist = fact[fact["invoice_date"] <= as_of]
    rfm = (
        hist.groupby("customer_id")
        .agg(
            recency_days=("invoice_date", lambda s: (as_of - s.max()).days),
            frequency=("invoice", "nunique"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )
    return rfm


def compute_behavioral(fact: pd.DataFrame, as_of: pd.Timestamp, dim_customers: pd.DataFrame) -> pd.DataFrame:
    hist = fact[fact["invoice_date"] <= as_of].copy()

    diversity = hist.groupby("customer_id")["stock_code"].nunique().rename("product_diversity")

    inv_dates = hist.groupby(["customer_id", "invoice"])["invoice_date"].first().reset_index()
    inv_dates = inv_dates.sort_values(["customer_id", "invoice_date"])

    def gap_stats(dates: pd.Series) -> float:
        if len(dates) < 2:
            return np.nan
        diffs = dates.sort_values().diff().dropna().dt.days
        return float(diffs.mean())

    avg_gap = inv_dates.groupby("customer_id")["invoice_date"].apply(gap_stats).rename("avg_days_between_orders")
    n_invoices = inv_dates.groupby("customer_id")["invoice"].nunique()
    single_buyer = (n_invoices == 1).rename("single_buyer")

    # Spend-momentum: slope of monthly revenue over the trailing 6 calendar
    # months, zero-filled for months with no activity (a deliberate choice —
    # it makes ramping-up new customers read as positive-trend, see module docstring).
    window_start = (as_of - pd.DateOffset(months=6)).tz_localize(None).to_period("M")
    as_of_period = as_of.tz_localize(None).to_period("M")
    month_range = pd.period_range(window_start, as_of_period, freq="M")

    hist["month"] = hist["invoice_date"].dt.tz_localize(None).dt.to_period("M")
    monthly = hist.groupby(["customer_id", "month"])["revenue"].sum().reset_index()

    def slope_for(group: pd.DataFrame) -> float:
        series = group.set_index("month")["revenue"].reindex(month_range, fill_value=0.0)
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return float(np.polyfit(x, series.values, 1)[0])

    trend = monthly.groupby("customer_id").apply(slope_for, include_groups=False).rename("spend_trend_slope")

    tenure = dim_customers.set_index("customer_id")["first_purchase_date"]
    tenure = (as_of - tenure).dt.days.rename("tenure_days")

    is_uk = (dim_customers.set_index("customer_id")["country"] == "United Kingdom").rename("is_uk")

    out = pd.concat([diversity, avg_gap, single_buyer, trend, tenure, is_uk], axis=1).reset_index()
    out = out.rename(columns={"index": "customer_id"})
    return out


def compute_target(fact: pd.DataFrame, as_of: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """target_conversion = 1 iff the customer buys again in (as_of, as_of + window_days]."""
    horizon = as_of + pd.Timedelta(days=window_days)
    eligible = fact.loc[fact["invoice_date"] <= as_of, "customer_id"].unique()
    future = fact[(fact["invoice_date"] > as_of) & (fact["invoice_date"] <= horizon)]
    converted = set(future["customer_id"].unique())
    target = pd.DataFrame({"customer_id": eligible})
    target["target_conversion"] = target["customer_id"].isin(converted).astype("int8")
    return target


def build_customer_features(
    fact: pd.DataFrame,
    dim_customers: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    rfm = compute_rfm(fact, as_of)
    behavioral = compute_behavioral(fact, as_of, dim_customers)

    out = rfm.merge(behavioral, on="customer_id", how="left")
    out["as_of"] = pd.Series([as_of] * len(out), dtype="datetime64[ns, UTC]")

    if include_target:
        if window_days is None:
            raise ValueError("window_days is required when include_target=True")
        target = compute_target(fact, as_of, window_days)
        out = out.merge(target, on="customer_id", how="left")
    else:
        out["target_conversion"] = pd.array([None] * len(out), dtype="Int8")

    out["single_buyer"] = out["single_buyer"].astype(bool)
    out["is_uk"] = out["is_uk"].astype(bool)
    out["frequency"] = out["frequency"].astype(int)
    out["product_diversity"] = out["product_diversity"].astype(int)
    out["recency_days"] = out["recency_days"].astype(int)
    out["tenure_days"] = out["tenure_days"].astype(int)
    out["target_conversion"] = out["target_conversion"].astype("Int8") if include_target else out["target_conversion"]

    customer_features_schema.validate(out, lazy=True)
    return out


def main() -> None:
    import argparse

    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    staging_path = Path(cfg["paths"]["staging"])
    marts_dir = Path(cfg["paths"]["marts_dir"])
    marts_dir.mkdir(parents=True, exist_ok=True)

    fact = pd.read_parquet(staging_path)
    fact["invoice_date"] = pd.to_datetime(fact["invoice_date"], utc=True)

    with log_step(logger, "features.dim_customers") as rec:
        dim_customers = build_dim_customers(fact)
        dim_customers.to_parquet(marts_dir / "dim_customers.parquet", index=False)
        rec["rows_out"] = len(dim_customers)

    fact.to_parquet(marts_dir / "fact_transactions.parquet", index=False)

    temporal = cfg["temporal"]
    window_days = temporal["target_window_days"]
    t_train = pd.Timestamp(temporal["t_train"], tz="UTC")
    t_test = pd.Timestamp(temporal["t_test"], tz="UTC")
    t_full = pd.Timestamp(temporal["t_full"], tz="UTC")

    with log_step(logger, "features.customer_features_train") as rec:
        cf_train = build_customer_features(fact, dim_customers, t_train, window_days, include_target=True)
        cf_train.to_parquet(marts_dir / "customer_features_train.parquet", index=False)
        rec["rows_out"] = len(cf_train)
        rec["conversion_rate"] = float(cf_train["target_conversion"].mean())

    with log_step(logger, "features.customer_features_test") as rec:
        cf_test = build_customer_features(fact, dim_customers, t_test, window_days, include_target=True)
        cf_test.to_parquet(marts_dir / "customer_features_test.parquet", index=False)
        rec["rows_out"] = len(cf_test)
        rec["conversion_rate"] = float(cf_test["target_conversion"].mean())

    if t_full == t_test:
        cf_current = cf_test
    else:
        with log_step(logger, "features.customer_features_current") as rec:
            cf_current = build_customer_features(fact, dim_customers, t_full, window_days, include_target=True)
            rec["rows_out"] = len(cf_current)
    cf_current.to_parquet(marts_dir / "customer_features_current.parquet", index=False)

    logger.info('{"step": "features.done"}')


if __name__ == "__main__":
    main()
