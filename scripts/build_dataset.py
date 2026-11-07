"""Rebuild data/processed/policies.parquet from the two raw Mendeley CSV files.

Usage: python scripts/build_dataset.py   (expects the CSVs in data/raw/, see data/README.md)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vip.data import build  # noqa: E402

if __name__ == "__main__":
    df = build(ROOT / "data/raw", ROOT / "data/processed/policies.parquet")
    print(f"Saved {len(df):,} policies on {df['object_id'].nunique():,} vehicles")
