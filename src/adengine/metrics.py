"""Media metrics engine — CPA, ROAS, LTV, cohort matrices.

Every function here is pure: a DataFrame or a handful of scalars in, a
DataFrame or scalar out, no I/O, no hidden state — which is what makes them
testable with 5-row fixtures covering the zero-conversion / zero-cost edge
cases (§6). The only stateful pieces are the BG/NBD + Gamma-Gamma fit
(inherently a fit/predict step) and its heuristic fallback.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from adengine.logging_conf import get_logger

logger = get_logger("metrics")


def cpa(cost: float, conversions: float) -> float:
    if conversions == 0:
        return float("inf")
    return cost / conversions


def roas(revenue: float, cost: float) -> float:
    if cost == 0:
        return float("inf") if revenue > 0 else 0.0
    return revenue / cost


def ltv_cac_ratio(ltv: float, cac: float) -> float:
    if cac == 0:
        return float("inf") if ltv > 0 else 0.0
    return ltv / cac


def ltv_heuristic(avg_order_value: float, purchase_frequency_annual: float, gross_margin: float, horizon_years: float = 1.0) -> float:
    """LTV = AOV x annual purchase frequency x gross margin x horizon.

    The documented fallback (§6) when BG/NBD + Gamma-Gamma cannot fit —
    e.g. too few repeat customers, or a `lifetimes` version incompatibility.
    """
    return avg_order_value * purchase_frequency_annual * gross_margin * horizon_years


def compute_window_revenue(fact: pd.DataFrame, as_of: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """Revenue actually realized in (as_of, as_of + window_days] per customer.

    This is a post-hoc, realized-outcome computation for reporting — not a
    model feature — so it is allowed to look past `as_of`; it must never be
    fed into features.py's anti-leakage feature set.
    """
    horizon = as_of + pd.Timedelta(days=window_days)
    future = fact[(fact["invoice_date"] > as_of) & (fact["invoice_date"] <= horizon)]
    return future.groupby("customer_id")["revenue"].sum().rename("window_revenue").reset_index()


def cohort_month_index(fact: pd.DataFrame, dim_customers: pd.DataFrame) -> pd.DataFrame:
    tx = fact[["customer_id", "invoice_date", "revenue"]].merge(
        dim_customers[["customer_id", "acquisition_cohort"]], on="customer_id", how="inner"
    )
    tx["cohort_period"] = pd.PeriodIndex(tx["acquisition_cohort"], freq="M")
    tx["txn_period"] = tx["invoice_date"].dt.tz_localize(None).dt.to_period("M")
    tx["months_since_acquisition"] = (
        (tx["txn_period"].dt.year - tx["cohort_period"].dt.year) * 12
        + (tx["txn_period"].dt.month - tx["cohort_period"].dt.month)
    )
    return tx


def cohort_retention_matrix(fact: pd.DataFrame, dim_customers: pd.DataFrame, observed_through: pd.Timestamp, max_months: int = 6) -> pd.DataFrame:
    """% of each acquisition cohort with >=1 order at each month-since-acquisition.

    Cells beyond what `observed_through` can possibly cover are left as NaN —
    that is censoring, not a zero-retention observation, and must not be
    imputed as such.
    """
    tx = cohort_month_index(fact, dim_customers)
    observed_period = observed_through.tz_localize(None).to_period("M") if observed_through.tz is not None else observed_through.to_period("M")

    cohort_sizes = dim_customers.groupby("acquisition_cohort")["customer_id"].nunique()
    cohorts = sorted(cohort_sizes.index)

    rows = []
    for cohort in cohorts:
        cohort_period = pd.Period(cohort, freq="M")
        max_observable = (observed_period.year - cohort_period.year) * 12 + (observed_period.month - cohort_period.month)
        row = {"acquisition_cohort": cohort, "cohort_size": int(cohort_sizes[cohort])}
        for m in range(max_months + 1):
            if m > max_observable:
                row[f"M{m}"] = np.nan
                continue
            active = tx[(tx["acquisition_cohort"] == cohort) & (tx["months_since_acquisition"] == m)]["customer_id"].nunique()
            row[f"M{m}"] = round(100 * active / row["cohort_size"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_revenue_matrix(fact: pd.DataFrame, dim_customers: pd.DataFrame, observed_through: pd.Timestamp, max_months: int = 6) -> pd.DataFrame:
    """Cumulative average revenue per acquired customer, by month since acquisition."""
    tx = cohort_month_index(fact, dim_customers)
    observed_period = observed_through.tz_localize(None).to_period("M") if observed_through.tz is not None else observed_through.to_period("M")
    cohort_sizes = dim_customers.groupby("acquisition_cohort")["customer_id"].nunique()
    cohorts = sorted(cohort_sizes.index)

    rows = []
    for cohort in cohorts:
        cohort_period = pd.Period(cohort, freq="M")
        max_observable = (observed_period.year - cohort_period.year) * 12 + (observed_period.month - cohort_period.month)
        row = {"acquisition_cohort": cohort, "cohort_size": int(cohort_sizes[cohort])}
        cum_revenue = 0.0
        for m in range(max_months + 1):
            if m > max_observable:
                row[f"M{m}"] = np.nan
                continue
            month_revenue = tx[(tx["acquisition_cohort"] == cohort) & (tx["months_since_acquisition"] == m)]["revenue"].sum()
            cum_revenue += month_revenue
            row[f"M{m}"] = round(cum_revenue / row["cohort_size"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_bgnbd_gamma_gamma(fact: pd.DataFrame, as_of: pd.Timestamp, horizon_months: int) -> tuple[pd.DataFrame, str]:
    """Returns (customer_id, ltv_probabilistic) and the method actually used.

    Falls back to the heuristic if `lifetimes` fails to import or fit — the
    documented mitigation for the "unmaintained dependency" risk (§10).
    """
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
        from lifetimes.utils import summary_data_from_transaction_data

        hist = fact[fact["invoice_date"] <= as_of].copy()
        hist["invoice_date"] = hist["invoice_date"].dt.tz_localize(None)
        as_of_naive = as_of.tz_localize(None) if as_of.tz is not None else as_of

        summary = summary_data_from_transaction_data(
            hist, "customer_id", "invoice_date", monetary_value_col="revenue",
            observation_period_end=as_of_naive, freq="D",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bgf = BetaGeoFitter(penalizer_coef=0.01)
            bgf.fit(summary["frequency"], summary["recency"], summary["T"])

            repeat = summary[summary["frequency"] > 0]
            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(repeat["frequency"], repeat["monetary_value"])

            clv = ggf.customer_lifetime_value(
                bgf, summary["frequency"], summary["recency"], summary["T"], summary["monetary_value"],
                time=horizon_months, freq="D", discount_rate=0.0,
            )
        out = pd.DataFrame({"customer_id": summary.index, "ltv_probabilistic": clv.reindex(summary.index).clip(lower=0).to_numpy()})
        return out, "bgnbd_gamma_gamma"
    except Exception as exc:
        logger.info(f'{{"step": "metrics.bgnbd_fallback", "reason": "{exc!r}"}}')
        return None, "heuristic_fallback"


def build_ltv_table(fact: pd.DataFrame, customer_features: pd.DataFrame, as_of: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    horizon_months = cfg["metrics"]["ltv_horizon_months"]
    margin = cfg["metrics"]["gross_margin"]

    probabilistic, method = fit_bgnbd_gamma_gamma(fact, as_of, horizon_months)

    out = customer_features[["customer_id", "monetary", "frequency", "tenure_days"]].copy()
    out["avg_order_value"] = out["monetary"] / out["frequency"]
    annual_freq = out["frequency"] / (out["tenure_days"].clip(lower=1) / 365.25)
    out["ltv_heuristic"] = out.apply(
        lambda r: ltv_heuristic(r["avg_order_value"], annual_freq[r.name], margin, horizon_months / 12), axis=1
    )
    out["ltv_historical"] = out["monetary"]

    if probabilistic is not None:
        out = out.merge(probabilistic, on="customer_id", how="left")
        out["ltv_probabilistic"] = out["ltv_probabilistic"].fillna(out["ltv_heuristic"])
    else:
        out["ltv_probabilistic"] = out["ltv_heuristic"]

    out["ltv_method"] = method
    return out


def channel_metrics(customer_features: pd.DataFrame, attribution: pd.DataFrame, ltv_table: pd.DataFrame, window_revenue: pd.DataFrame) -> pd.DataFrame:
    df = (
        customer_features.merge(attribution, on="customer_id")
        .merge(ltv_table[["customer_id", "ltv_probabilistic"]], on="customer_id")
        .merge(window_revenue, on="customer_id", how="left")
    )
    df["window_revenue"] = df["window_revenue"].fillna(0.0)
    rows = []
    for channel, g in df.groupby("channel"):
        cost = g["cac_synthetic"].sum()
        conversions = int(g["target_conversion"].sum())
        revenue = g["window_revenue"].sum()
        ltv = g["ltv_probabilistic"].mean()
        cac_mean = g["cac_synthetic"].mean()
        rows.append({
            "channel": channel,
            "customers": len(g),
            "spend_synthetic": round(cost, 2),
            "conversions": conversions,
            "revenue": round(revenue, 2),
            "cpa": round(cpa(cost, conversions), 2),
            "roas": round(roas(revenue, cost), 2),
            "avg_cac": round(cac_mean, 2),
            "avg_ltv": round(ltv, 2),
            "ltv_cac": round(ltv_cac_ratio(ltv, cac_mean), 2),
        })
    return pd.DataFrame(rows).sort_values("roas", ascending=False).reset_index(drop=True)


def channel_segment_metrics(customer_features: pd.DataFrame, attribution: pd.DataFrame, segment_assignments: pd.DataFrame, ltv_table: pd.DataFrame, window_revenue: pd.DataFrame) -> pd.DataFrame:
    df = (
        customer_features.merge(attribution, on="customer_id")
        .merge(segment_assignments[["customer_id", "segment_name"]], on="customer_id")
        .merge(ltv_table[["customer_id", "ltv_probabilistic"]], on="customer_id")
        .merge(window_revenue, on="customer_id", how="left")
    )
    df["window_revenue"] = df["window_revenue"].fillna(0.0)
    rows = []
    for (channel, segment), g in df.groupby(["channel", "segment_name"]):
        cost = g["cac_synthetic"].sum()
        conversions = int(g["target_conversion"].sum())
        revenue = g["window_revenue"].sum()
        rows.append({
            "channel": channel,
            "segment": segment,
            "customers": len(g),
            "cpa": round(cpa(cost, conversions), 2),
            "roas": round(roas(revenue, cost), 2),
            "avg_ltv": round(g["ltv_probabilistic"].mean(), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from adengine.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe_cfg = load_config(args.pipeline_config)
    marts_dir = Path(pipe_cfg["paths"]["marts_dir"])

    fact = pd.read_parquet(marts_dir / "fact_transactions.parquet")
    fact["invoice_date"] = pd.to_datetime(fact["invoice_date"], utc=True)
    dim_customers = pd.read_parquet(marts_dir / "dim_customers.parquet")
    customer_features = pd.read_parquet(marts_dir / "customer_features_current.parquet")
    attribution = pd.read_parquet(marts_dir / "synthetic_attribution.parquet")
    segment_assignments = pd.read_parquet(marts_dir / "segment_assignments.parquet")

    t_test = pd.Timestamp(pipe_cfg["temporal"]["t_test"], tz="UTC")

    retention = cohort_retention_matrix(fact, dim_customers, t_test)
    revenue_matrix = cohort_revenue_matrix(fact, dim_customers, t_test)
    retention.to_parquet(marts_dir / "cohort_retention_matrix.parquet", index=False)
    revenue_matrix.to_parquet(marts_dir / "cohort_revenue_matrix.parquet", index=False)

    ltv_table = build_ltv_table(fact, customer_features, t_test, cfg)
    ltv_table.to_parquet(marts_dir / "ltv_table.parquet", index=False)

    window_days = pipe_cfg["temporal"]["target_window_days"]
    window_revenue = compute_window_revenue(fact, t_test, window_days)

    ch_metrics = channel_metrics(customer_features, attribution, ltv_table, window_revenue)
    ch_metrics.to_parquet(marts_dir / "channel_metrics.parquet", index=False)

    ch_seg_metrics = channel_segment_metrics(customer_features, attribution, segment_assignments, ltv_table, window_revenue)
    ch_seg_metrics.to_parquet(marts_dir / "channel_segment_metrics.parquet", index=False)

    logger.info(json.dumps({
        "step": "metrics.done",
        "ltv_method": ltv_table["ltv_method"].iloc[0],
        "channels": ch_metrics.to_dict("records"),
    }, default=str))


if __name__ == "__main__":
    main()
