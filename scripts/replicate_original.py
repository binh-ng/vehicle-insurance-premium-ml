"""Re-run the original DATA 611 tuned random forest and check its reported results.

1. Reproduce the tuned RF with the original preprocessing (2014-2018 file, 9 columns,
   drop_duplicates on those columns, 10% sample, random 80/20 split) and the best
   parameters printed in the final notebook.
2. Repeat with five different 10% samples to show how much one sample moves the score.
3. Report feature importance from the *tuned* model, with one-hot dummies summed back to
   their original variable (the original read importances from the untuned model).

Usage: python scripts/replicate_original.py   ->  results/replication.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vip.data import read_raw_file  # noqa: E402

# Best parameters printed by RandomizedSearchCV in the final group notebook
BEST = dict(n_estimators=300, min_samples_split=20, min_samples_leaf=3, max_samples=None,
            max_features=0.5, max_depth=15)
COLS = {"SEX": "sex", "SEATS_NUM": "seats_num", "CCM_TON": "ccm_ton", "INSR_TYPE": "insurance_type",
        "INSURED_VALUE": "insured_value", "PREMIUM": "premium", "PROD_YEAR": "production_year",
        "TYPE_VEHICLE": "vehicle_type", "USAGE": "usage"}
DUMMIES = ["sex", "insurance_type", "vehicle_type", "usage"]
REPORTED = {"train": {"r2": 0.783, "rmse": 6109.13, "mae": 3483.67},
            "test": {"r2": 0.742, "rmse": 6641.06, "mae": 3864.53}}


def scores(y, p):
    return {"r2": r2_score(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": mean_absolute_error(y, p)}


def main():
    raw = read_raw_file(ROOT / "data/raw", "motor_data14-2018.csv")
    base = raw[list(COLS)].rename(columns=COLS)
    n_raw = len(base)
    base = base.drop_duplicates().dropna(subset=["premium", "production_year"])
    base = base[pd.to_numeric(base["premium"], errors="coerce").gt(0)]
    out = {"reported": REPORTED, "rows_raw": n_raw, "rows_after_original_dedup": len(base),
           "note": "drop_duplicates on 9 columns removes renewals of the same vehicle with identical values", "runs": []}

    for seed in [42, 1, 2, 3, 4]:
        df = base.sample(frac=0.1, random_state=seed)
        y = df.pop("premium")
        X = pd.get_dummies(df, columns=DUMMIES, drop_first=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, **BEST).fit(X_tr, y_tr)
        run = {"sample_seed": seed, "train": scores(y_tr, rf.predict(X_tr)), "test": scores(y_te, rf.predict(X_te))}
        if seed == 42:
            imp = pd.Series(rf.feature_importances_, index=X.columns)
            grouped = imp.groupby(lambda c: next((d for d in DUMMIES if c.startswith(d + "_")), c)).sum()
            out["importance_tuned_dummies"] = imp.sort_values(ascending=False).head(8).round(4).to_dict()
            out["importance_tuned_grouped"] = grouped.sort_values(ascending=False).round(4).to_dict()
        out["runs"].append(run)
        print(seed, round(run["test"]["r2"], 3))

    t = [r["test"]["r2"] for r in out["runs"]]
    out["test_r2_range"] = [min(t), max(t)]
    (ROOT / "results/replication.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ["test_r2_range", "importance_tuned_grouped"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
