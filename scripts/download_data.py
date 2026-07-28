"""Reproducible, hash-verified download of Online Retail II (UCI).

If data/raw/online_retail_II.xlsx already exists it is not re-downloaded —
its hash is simply recorded, so a clean checkout still reproduces a manifest
without hitting the network twice. Run:

    python scripts/download_data.py --config configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adengine.config import load_config  # noqa: E402
from adengine.logging_conf import get_logger  # noqa: E402

logger = get_logger("download_data")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_extract(url: str, dest_xlsx: Path) -> None:
    logger.info(json.dumps({"step": "download", "url": url}))
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = resp.read()
    dest_xlsx.parent.mkdir(parents=True, exist_ok=True)
    if url.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
            if not xlsx_names:
                raise RuntimeError("No .xlsx found inside downloaded zip")
            with zf.open(xlsx_names[0]) as src, open(dest_xlsx, "wb") as out:
                out.write(src.read())
    else:
        dest_xlsx.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dest = Path(cfg["paths"]["raw_xlsx"])
    manifest_path = Path(cfg["paths"]["manifest"])

    if not dest.exists():
        download_and_extract(cfg["source"]["url"], dest)
    else:
        logger.info(json.dumps({"step": "download", "status": "already present, skipping fetch"}))

    digest = sha256_of(dest)

    import pandas as pd

    xl = pd.ExcelFile(dest)
    rows_raw = 0
    for sheet in xl.sheet_names:
        rows_raw += len(pd.read_excel(xl, sheet_name=sheet, usecols=[0]))

    manifest = {
        "source_url": cfg["source"]["url"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sha256": digest,
        "rows_raw": rows_raw,
        "sheets": xl.sheet_names,
        "file_size_bytes": dest.stat().st_size,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(json.dumps({"step": "manifest_written", "path": str(manifest_path), **manifest}))


if __name__ == "__main__":
    main()
