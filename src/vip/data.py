"""Load, clean and engineer features for the Ethiopian vehicle insurance data.

Source: Terefe, E. (2023). Vehicle Insurance data. Mendeley Data, V1.
https://doi.org/10.17632/34nfrk36dt.1 (CC BY 4.0)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILES = ["motor_data11-14lats.csv", "motor_data14-2018.csv"]

SEX_MAP = {0: "Legal entity", 1: "Male", 2: "Female"}
INSR_TYPE_MAP = {1201: "Private", 1202: "Commercial", 1204: "Motor trade road risk"}

# Features known when a policy is quoted (no same-period claim information).
CATEGORICAL = ["sex", "insurance_type", "coverage", "vehicle_type", "usage", "make", "season"]
NUMERIC = [
    "log_insured_value",
    "vehicle_age",
    "seats_num",
    "carrying_capacity",
    "ccm_ton",
    "duration_days",
    "prior_policies",
    "prior_claim",
    "customer_tenure",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "premium"
GROUP = "object_id"


def read_raw_file(raw_dir: str | Path, name: str) -> pd.DataFrame:
    """Read one raw Mendeley file: the original CSV if present, else the lossless parquet copy in git."""
    csv = Path(raw_dir) / name
    if csv.exists():
        return pd.read_csv(csv, dtype={"EFFECTIVE_YR": str})
    pq = csv.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    raise FileNotFoundError(f"{name} not found. Download it from https://data.mendeley.com/datasets/34nfrk36dt/1")


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    frames = []
    for f in RAW_FILES:
        df = read_raw_file(raw_dir, f)
        df["source_file"] = f
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _two_digit_year(s: pd.Series) -> pd.Series:
    y = pd.to_numeric(s, errors="coerce")
    return np.where(y <= 18, 2000 + y, 1900 + y)


def clean(raw: pd.DataFrame, top_makes: int = 30) -> pd.DataFrame:
    df = raw.drop(columns="source_file").drop_duplicates().copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"type_vehicle": "vehicle_type", "insr_type": "insurance_type"})

    df["insr_begin"] = pd.to_datetime(df["insr_begin"], format="%d-%b-%y", errors="coerce")
    df["insr_end"] = pd.to_datetime(df["insr_end"], format="%d-%b-%y", errors="coerce")

    # Target must exist and be positive; production year and make are needed for pricing.
    df = df[df["premium"].gt(0) & df["prod_year"].notna() & df["insr_begin"].notna()]
    df = df[df["vehicle_type"].ne("Trade plates")]  # 18 rows, not a vehicle

    df["sex"] = df["sex"].map(SEX_MAP)
    df["insurance_type"] = df["insurance_type"].map(INSR_TYPE_MAP)

    # Zero insured value = liability (third-party) only cover, priced on a national tariff.
    df["coverage"] = np.where(df["insured_value"] > 0, "Comprehensive", "Liability only")
    df["log_insured_value"] = np.log1p(df["insured_value"].clip(lower=0))

    df["begin_year"] = df["insr_begin"].dt.year
    df["vehicle_age"] = (df["begin_year"] - df["prod_year"]).clip(0, 60)
    df["duration_days"] = (df["insr_end"] - df["insr_begin"]).dt.days.clip(1, 400)
    df["season"] = df["insr_begin"].dt.month.map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )
    df["customer_tenure"] = (df["begin_year"] - _two_digit_year(df["effective_yr"])).clip(0, 60)

    df["seats_num"] = df["seats_num"].clip(0, 200)
    df["carrying_capacity"] = df["carrying_capacity"].clip(0, 100_000)
    df["ccm_ton"] = df["ccm_ton"].clip(0, 20_000)

    makes = df["make"].value_counts().head(top_makes).index
    df["make"] = df["make"].where(df["make"].isin(makes), "Other").fillna("Other")

    # Claim history from earlier policies of the same vehicle (a no-claim-discount proxy).
    df["had_claim"] = df["claim_paid"].fillna(0).gt(0).astype(int)
    df = df.sort_values(["object_id", "insr_begin"])
    g = df.groupby("object_id")
    df["prior_policies"] = g.cumcount()
    df["prior_claim"] = g["had_claim"].transform(lambda s: s.shift(1).fillna(0)).astype(int)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    keep = ["object_id", "insr_begin", "begin_year", "insured_value", "claim_paid", TARGET] + FEATURES
    return df[keep].reset_index(drop=True)


def build(raw_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    df = clean(load_raw(raw_dir))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")
    return df
