"""Silver layer — auditable cleaning with a per-rule quality report.

Every rule is a pure function `(DataFrame) -> (DataFrame, DQRecord)`. The
orchestrator applies them in a fixed, config-documented order and accumulates
the records into `data/staging/dq_report.json`. Nothing is dropped silently —
that is the entire point of this module (adengine_escopo_tecnico.md §3.2).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from adengine.contracts import silver_schema
from adengine.logging_conf import get_logger, log_step

logger = get_logger("cleaning")


@dataclass(frozen=True)
class DQRecord:
    rule: str
    rationale: str
    rows_before: int
    rows_removed: int

    @property
    def pct(self) -> float:
        return round(100 * self.rows_removed / max(self.rows_before, 1), 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pct_of_input"] = self.pct
        return d


def drop_cancellations(df: pd.DataFrame) -> tuple[pd.DataFrame, DQRecord]:
    mask = df["invoice"].str.startswith("C")
    rec = DQRecord(
        "cancellations_c_invoices",
        "Invoices prefixed 'C' are cancellations, not completed conversions.",
        len(df), int(mask.sum()),
    )
    return df.loc[~mask].copy(), rec


def drop_non_product_stock_codes(df: pd.DataFrame, exclude_codes: list[str]) -> tuple[pd.DataFrame, DQRecord]:
    mask = df["stock_code"].isin(exclude_codes)
    rec = DQRecord(
        "non_product_stock_code",
        "Postage, discounts, bank charges, and other non-product line items (see configs/pipeline.yaml).",
        len(df), int(mask.sum()),
    )
    return df.loc[~mask].copy(), rec


def drop_invalid_quantity_price(df: pd.DataFrame) -> tuple[pd.DataFrame, DQRecord]:
    mask = (df["quantity"] <= 0) | (df["unit_price"] <= 0)
    rec = DQRecord(
        "invalid_quantity_or_price",
        "Non-positive quantity or price: manual adjustments and data-entry artifacts, not valid sales.",
        len(df), int(mask.sum()),
    )
    return df.loc[~mask].copy(), rec


def drop_missing_customer_id(df: pd.DataFrame) -> tuple[pd.DataFrame, DQRecord]:
    mask = df["customer_id"].isna()
    rec = DQRecord(
        "missing_customer_id",
        "No stable key for customer-level modeling — cannot segment or score an anonymous transaction.",
        len(df), int(mask.sum()),
    )
    return df.loc[~mask].copy(), rec


def drop_duplicate_line_items(df: pd.DataFrame) -> tuple[pd.DataFrame, DQRecord]:
    mask = df.duplicated(subset=["invoice", "stock_code", "invoice_date", "customer_id", "quantity", "unit_price"])
    rec = DQRecord(
        "duplicate_line_items",
        "Exact duplicate rows are ingestion artifacts from the source export.",
        len(df), int(mask.sum()),
    )
    return df.loc[~mask].copy(), rec


def standardize(df: pd.DataFrame) -> tuple[pd.DataFrame, DQRecord]:
    """Schema conformance only — no row loss, but logged for a complete audit trail."""
    out = df.copy()
    out["customer_id"] = out["customer_id"].astype("int64").astype(str)
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.tz_localize("UTC")
    out["revenue"] = out["quantity"] * out["unit_price"]
    out["country"] = out["country"].str.strip()
    rec = DQRecord(
        "standardize_types",
        "customer_id -> string, invoice_date -> UTC, revenue = quantity * unit_price, country trimmed.",
        len(out), 0,
    )
    return out, rec


CLEANING_STEPS = [
    "cancellations_c_invoices",
    "non_product_stock_code",
    "invalid_quantity_or_price",
    "missing_customer_id",
    "duplicate_line_items",
    "standardize_types",
]


def run_cleaning(bronze: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[DQRecord]]:
    rows_raw = len(bronze)
    records: list[DQRecord] = []
    df = bronze

    with log_step(logger, "cleaning.drop_cancellations") as rec_log:
        df, rec = drop_cancellations(df)
        records.append(rec)
        rec_log.update(rows_out=len(df), removed=rec.rows_removed)

    with log_step(logger, "cleaning.drop_non_product_stock_codes") as rec_log:
        df, rec = drop_non_product_stock_codes(df, cfg["cleaning"]["non_product_stock_codes"])
        records.append(rec)
        rec_log.update(rows_out=len(df), removed=rec.rows_removed)

    with log_step(logger, "cleaning.drop_invalid_quantity_price") as rec_log:
        df, rec = drop_invalid_quantity_price(df)
        records.append(rec)
        rec_log.update(rows_out=len(df), removed=rec.rows_removed)

    with log_step(logger, "cleaning.drop_missing_customer_id") as rec_log:
        df, rec = drop_missing_customer_id(df)
        records.append(rec)
        rec_log.update(rows_out=len(df), removed=rec.rows_removed)

    with log_step(logger, "cleaning.drop_duplicate_line_items") as rec_log:
        df, rec = drop_duplicate_line_items(df)
        records.append(rec)
        rec_log.update(rows_out=len(df), removed=rec.rows_removed)

    with log_step(logger, "cleaning.standardize") as rec_log:
        df, rec = standardize(df)
        records.append(rec)
        rec_log.update(rows_out=len(df))

    silver_schema.validate(df, lazy=True)

    rows_silver = len(df)
    logger.info(json.dumps({
        "step": "cleaning.summary",
        "rows_raw": rows_raw,
        "rows_silver": rows_silver,
        "retention_pct": round(100 * rows_silver / rows_raw, 2),
    }))
    return df, records


def write_dq_report(records: list[DQRecord], rows_raw: int, rows_silver: int, path: str | Path) -> dict:
    report = {
        "rows_raw": rows_raw,
        "rows_silver": rows_silver,
        "retention_pct": round(100 * rows_silver / rows_raw, 2),
        "rules": [r.to_dict() for r in records],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    from adengine.config import load_config
    from adengine.ingestion import build_bronze

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    bronze = build_bronze(cfg)
    silver, records = run_cleaning(bronze, cfg)

    staging_path = Path(cfg["paths"]["staging"])
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(staging_path, index=False)

    write_dq_report(records, len(bronze), len(silver), cfg["paths"]["dq_report"])
    logger.info(json.dumps({"step": "cleaning.done", "written_to": str(staging_path)}))


if __name__ == "__main__":
    main()
