"""Bronze layer — ingestion with zero business-rule transformation.

The only thing this module does beyond a literal read is coerce columns to
stable dtypes so the frame can round-trip through Parquet (the source .xlsx
mixes int/str within a column across its two sheets). That is storage
plumbing, not cleaning: no rows are dropped and no values are altered.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from adengine.logging_conf import get_logger, log_step

logger = get_logger("ingestion")

RAW_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def _safe_str(col: pd.Series) -> pd.Series:
    return col.apply(lambda x: x if pd.isna(x) else str(x))


def read_bronze_from_source(raw_xlsx: str | Path, sheets: list[str]) -> pd.DataFrame:
    """Reads every sheet of the source workbook and concatenates them as-is."""
    xl = pd.ExcelFile(raw_xlsx)
    frames = []
    for sheet in sheets:
        df = pd.read_excel(xl, sheet_name=sheet)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    bronze = pd.concat(frames, ignore_index=True)
    bronze = bronze.rename(columns=RAW_COLUMNS)
    for col in ("invoice", "stock_code", "description", "country"):
        bronze[col] = _safe_str(bronze[col])
    return bronze


def build_bronze(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Reads the source workbook once and caches it as Parquet for fast reuse."""
    bronze_path = Path(cfg["paths"]["bronze"])
    with log_step(logger, "ingestion.build_bronze", force=force) as rec:
        if bronze_path.exists() and not force:
            bronze = pd.read_parquet(bronze_path)
            rec["source"] = "cache"
        else:
            bronze = read_bronze_from_source(cfg["paths"]["raw_xlsx"], cfg["source"]["sheets"])
            bronze_path.parent.mkdir(parents=True, exist_ok=True)
            bronze.to_parquet(bronze_path, index=False)
            rec["source"] = "xlsx"
        rec["rows_out"] = len(bronze)
    return bronze
