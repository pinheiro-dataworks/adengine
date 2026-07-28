"""Structured JSON-lines logging, configured once per process.

Each pipeline step should log a single event via `log_step` carrying the
step name, row counts in/out, duration, and the config parameters used —
that record is what `dq_report.json` and the CI logs are built from.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("adengine")
    root.setLevel(level)
    root.handlers = [handler]
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"adengine.{name}")


@contextmanager
def log_step(logger: logging.Logger, step: str, **params: Any) -> Iterator[dict]:
    """Times a pipeline step and emits one JSON line with rows_in/rows_out/duration."""
    record: dict[str, Any] = {"step": step, "params": params}
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record["duration_s"] = round(time.perf_counter() - t0, 3)
        logger.info(json.dumps(record, default=str))
