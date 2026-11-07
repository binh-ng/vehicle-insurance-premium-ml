"""Shared helpers for the Streamlit app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Reference palette (validated categorical order) and recessive inks
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e7e6e2"
COVERAGE_COLORS = {"Comprehensive": BLUE, "Liability only": ORANGE}
CURRENCY = "ETB"  # see data notes: amounts are most likely Ethiopian Birr

TEAM = [
    ("Binh Nguyen", "Bill"),
    ("Hui Tsz Chuen", "Alvin"),
    ("Natcha Kurupakorn", "Judy"),
]

DATA_CITATION = (
    "Terefe, E. (2023). *Vehicle Insurance data* (Version 1) [Data set]. Mendeley Data. "
    "https://doi.org/10.17632/34nfrk36dt.1. Licensed under CC BY 4.0."
)

pio.templates["vip"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK),
        colorway=[BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#6250d6", "#e34948"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK2)),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK2)),
        legend=dict(orientation="h", y=1.08, x=0, title=None),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor="white", font_size=12),
        bargap=0.25,
    )
)
pio.templates.default = "plotly_white+vip"


def team_line() -> str:
    return " · ".join(f"**{full} ({pref})**" for full, pref in TEAM)


@st.cache_data(show_spinner="Loading policies...")
def load_policies() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/processed/policies.parquet")
    df["rate_pct"] = (df["premium"] / df["insured_value"].where(df["insured_value"] > 0)) * 100
    return df


@st.cache_data
def load_result(name: str):
    p = ROOT / "results" / name
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return json.loads(p.read_text())


def fmt_money(x: float) -> str:
    return f"{x:,.0f} {CURRENCY}"


def footer():
    st.divider()
    st.caption(f"Data: {DATA_CITATION}")
