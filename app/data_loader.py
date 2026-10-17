"""Cached artifact loaders — the dashboard reads only precomputed outputs from
data/marts/ and never re-runs the pipeline or loads a model (Streamlit
Community Cloud's ~1GB RAM budget makes in-app model training the most likely
cause of an out-of-memory crash on the free tier).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MARTS_DIR = ROOT_DIR / "data" / "marts"
CONFIGS_DIR = ROOT_DIR / "configs"


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(MARTS_DIR / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    return json.loads((MARTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_dq_report() -> dict:
    return json.loads((ROOT_DIR / "data" / "staging" / "dq_report.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_manifest() -> dict | None:
    path = ROOT_DIR / "data" / "raw" / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def data_available() -> bool:
    return (MARTS_DIR / "fact_transactions.parquet").exists()
