"""Bronze is the lineage anchor for every number downstream: it must keep
every source row, normalize only column names and storage dtypes, and be
traceable to an exact source file through the manifest hash. These tests
build a tiny two-sheet workbook locally, so CI never touches the network.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from adengine.config import load_config
from adengine.ingestion import RAW_COLUMNS, build_bronze, read_bronze_from_source

SHEETS = ["Year 2009-2010", "Year 2010-2011"]


def _write_workbook(path: Path) -> None:
    # Mirrors the real export: padded headers, and StockCode mixing ints and
    # strings across sheets, which is what breaks a naive Parquet round-trip.
    sheet_a = pd.DataFrame({
        "Invoice": [489434, 489434],
        "StockCode": [85048, "79323P"],
        "Description ": ["LIGHT", "CANDLE"],
        "Quantity": [12, 6],
        "InvoiceDate": pd.to_datetime(["2009-12-01 07:45", "2009-12-01 07:45"]),
        "Price": [6.95, 1.25],
        "Customer ID": [13085.0, 13085.0],
        "Country": ["United Kingdom", "United Kingdom"],
    })
    sheet_b = pd.DataFrame({
        "Invoice": ["C536379", 536380],
        "StockCode": ["D", 22423],
        "Description ": ["Discount", "REGENCY CAKESTAND"],
        "Quantity": [-1, 2],
        "InvoiceDate": pd.to_datetime(["2010-12-01 09:41", "2010-12-01 09:45"]),
        "Price": [27.5, 12.75],
        "Customer ID": [14527.0, None],
        "Country": ["United Kingdom", "France"],
    })
    with pd.ExcelWriter(path) as writer:
        sheet_a.to_excel(writer, sheet_name=SHEETS[0], index=False)
        sheet_b.to_excel(writer, sheet_name=SHEETS[1], index=False)


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    path = tmp_path / "source.xlsx"
    _write_workbook(path)
    return path


def test_read_bronze_keeps_every_row_and_renames_columns(workbook):
    bronze = read_bronze_from_source(workbook, SHEETS)

    assert len(bronze) == 4  # no row dropped, including the cancellation and the missing ID
    assert list(bronze.columns) == list(RAW_COLUMNS.values())
    assert bronze["customer_id"].isna().sum() == 1
    assert (bronze["quantity"] < 0).sum() == 1  # values are not altered either


def test_read_bronze_coerces_mixed_type_columns_to_str(workbook):
    bronze = read_bronze_from_source(workbook, SHEETS)

    for col in ("invoice", "stock_code", "description", "country"):
        assert bronze[col].map(type).eq(str).all(), col
    assert set(bronze["stock_code"]) == {"85048", "79323P", "D", "22423"}


def test_build_bronze_caches_and_reuses_parquet(workbook, tmp_path):
    cfg = {
        "paths": {"raw_xlsx": str(workbook), "bronze": str(tmp_path / "raw" / "bronze.parquet")},
        "source": {"sheets": SHEETS},
    }
    first = build_bronze(cfg)
    assert (tmp_path / "raw" / "bronze.parquet").exists()

    # With the source gone, a second build must come from the cache, not the xlsx.
    workbook.unlink()
    cached = build_bronze(cfg)
    pd.testing.assert_frame_equal(first, cached)

    with pytest.raises(FileNotFoundError):
        build_bronze(cfg, force=True)


def test_load_config_parses_mapping_and_rejects_non_mapping(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("seed: 7\npaths:\n  marts_dir: data/marts\n", encoding="utf-8")
    assert load_config(good) == {"seed": 7, "paths": {"marts_dir": "data/marts"}}

    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        load_config(bad)


def _load_download_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_hash_detects_a_single_changed_byte(tmp_path):
    sha256_of = _load_download_script().sha256_of
    # Larger than the 1 MiB read chunk, so the streaming path is exercised.
    payload = bytes(range(256)) * 5000
    original = tmp_path / "source.bin"
    original.write_bytes(payload)
    assert sha256_of(original) == hashlib.sha256(payload).hexdigest()

    tampered = tmp_path / "tampered.bin"
    tampered.write_bytes(payload[:-1] + b"\x00")
    assert sha256_of(tampered) != sha256_of(original)
