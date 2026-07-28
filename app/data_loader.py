"""Cached artifact loaders — the dashboard reads only from data/marts/ and
models/, never re-runs the pipeline (Streamlit Community Cloud's ~1GB RAM
budget makes in-app training the #1 cause of free-tier deploy crashes, per
adengine_escopo_tecnico.md §8).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import yaml

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MARTS_DIR = ROOT_DIR / "data" / "marts"
MODELS_DIR = ROOT_DIR / "models"
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


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return joblib.load(MODELS_DIR / f"{name}.joblib")


def data_available() -> bool:
    return (MARTS_DIR / "fact_transactions.parquet").exists()
