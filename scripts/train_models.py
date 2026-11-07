"""Train, tune and compare premium models.

Usage:  python scripts/train_models.py [--fast]

Outputs (results/ and models/):
  model_comparison.csv      test metrics for every model (vehicle-grouped split)
  segment_metrics.csv       metrics by coverage type
  temporal_holdout.csv      train on earlier policies, test on the most recent year
  leakage_check.csv         random split vs vehicle-grouped split
  ablation.csv              original feature set vs expanded feature set
  shap_*.parquet            LightGBM SHAP values for explainability
  best_params.json          tuned hyperparameters
  models/lgbm_*.txt         point + quantile LightGBM models for the Streamlit estimator
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, GammaRegressor, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vip.data import CATEGORICAL, FEATURES, GROUP, NUMERIC, TARGET  # noqa: E402
from vip.models import (LogTargetModel, linear_preprocessor, metrics,  # noqa: E402
                        segment_metrics, tree_preprocessor)

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
RES = ROOT / "results"
MOD = ROOT / "models"
RES.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def grouped_split(df, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, te = next(gss.split(df, groups=df[GROUP]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def subsample_groups(df, n_rows):
    if len(df) <= n_rows:
        return df
    groups = df[GROUP].unique()
    rng = np.random.default_rng(SEED)
    keep = rng.choice(groups, size=int(len(groups) * n_rows / len(df)), replace=False)
    return df[df[GROUP].isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------- tuning helpers
def cv_rmsle(make_and_fit, df, n_splits=3):
    """Mean log-scale RMSE over vehicle-grouped folds."""
    scores = []
    for tr, va in GroupKFold(n_splits=n_splits).split(df, groups=df[GROUP]):
        pred_log = make_and_fit(df.iloc[tr], df.iloc[va])
        scores.append(np.sqrt(np.mean((np.log(df[TARGET].values[va]) - pred_log) ** 2)))
    return float(np.mean(scores))


def lgb_params(trial):
    return {
        "objective": "regression",
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        "cat_smooth": trial.suggest_float("cat_smooth", 1, 50, log=True),
        "verbose": -1,
        "seed": SEED,
        "num_threads": 2,
    }


def fit_lgb(params, tr, va=None, rounds=2000):
    dtr = lgb.Dataset(tr[FEATURES], np.log(tr[TARGET]), categorical_feature=CATEGORICAL, free_raw_data=False)
    if va is None:
        return lgb.train(params, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(va[FEATURES], np.log(va[TARGET]), reference=dtr)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def xgb_params(trial):
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eta": trial.suggest_float("eta", 0.03, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
        "max_cat_to_onehot": 8,
        "seed": SEED,
        "nthread": 2,
    }


def xgb_matrix(d):
    return xgb.DMatrix(d[FEATURES], label=np.log(d[TARGET]), enable_categorical=True)


def fit_xgb(params, tr, va=None, rounds=2000):
    dtr = xgb_matrix(tr)
    if va is None:
        return xgb.train(params, dtr, num_boost_round=rounds)
    return xgb.train(params, dtr, num_boost_round=rounds, evals=[(xgb_matrix(va), "va")],
                     early_stopping_rounds=100, verbose_eval=False)


class BoosterWrap:
    """Make LightGBM / XGBoost boosters look like LogTargetModel."""

    def __init__(self, booster, kind, smear):
        self.b, self.kind, self.smear_ = booster, kind, smear

    def predict_log(self, d):
        if self.kind == "lgb":
            return self.b.predict(d[FEATURES])
        return self.b.predict(xgb.DMatrix(d[FEATURES], enable_categorical=True))

    def predict(self, d):
        return np.exp(self.predict_log(d)) * self.smear_


def smear(train, pred_log):
    return float(np.mean(np.exp(np.log(train[TARGET].values) - pred_log)))


# ---------------------------------------------------------------- main
def main(fast=False):
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data/processed/policies.parquet")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    train, test = grouped_split(df)
    log(f"train {len(train):,} rows / test {len(test):,} rows (split by vehicle)")
    tune = subsample_groups(train, 40_000 if fast else 100_000)
    n_trials = 6 if fast else 40
    budget = 120 if fast else 700  # seconds per Optuna study (2-CPU workspace)
    X_tr, y_tr = train[FEATURES], train[TARGET]
    results, preds, best_params = [], {}, {}

    def record(name, family, model, fit_seconds, notes=""):
        p = model.predict(test if isinstance(model, BoosterWrap) else test[FEATURES])
        preds[name] = p
        m = metrics(test[TARGET], p)
        m.update(model=name, family=family, fit_seconds=round(fit_seconds, 1), notes=notes)
        results.append(m)
        log(f"{name:<28} R2={m['R2']:.3f} R2_log={m['R2_log']:.3f} MAE={m['MAE']:,.0f} MdAPE={m['MdAPE']:.3f}")
        pd.DataFrame(results).to_csv(RES / "model_comparison.csv", index=False)

    # 1. Domain baseline: tariff-style lookup (median premium per vehicle type x usage x coverage)
    t = time.time()
    keys = ["vehicle_type", "usage", "coverage"]
    lut = train.groupby(keys, observed=True)[TARGET].median().rename("p").reset_index()
    p = test[keys].merge(lut, on=keys, how="left")["p"].fillna(train[TARGET].median()).values
    preds["Tariff lookup (baseline)"] = p
    m = metrics(test[TARGET], p)
    m.update(model="Tariff lookup (baseline)", family="Baseline", fit_seconds=round(time.time() - t, 1),
             notes="Median premium by vehicle type x usage x coverage")
    results.append(m)

    # 2. Linear models on log premium
    lin_sub = subsample_groups(train, 50_000)
    for name, est in [
        ("OLS (log)", LinearRegression()),
        ("Ridge (log)", RidgeCV(alphas=np.logspace(-4, 3, 15))),
        ("Lasso (log)", LassoCV(alphas=np.logspace(-5, -1, 9), cv=3, max_iter=2000, tol=1e-3, selection="random", random_state=SEED)),
        ("Elastic Net (log)", ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=np.logspace(-5, -1, 9), cv=3,
                                           max_iter=2000, tol=1e-3, selection="random", random_state=SEED)),
    ]:
        t = time.time()
        pipe = Pipeline([("prep", linear_preprocessor()), ("est", est)])
        if name in ("Lasso (log)", "Elastic Net (log)"):
            pipe.fit(lin_sub[FEATURES], np.log(lin_sub[TARGET]))  # choose alpha on a subsample
            a = pipe.named_steps["est"].alpha_
            best_params[name] = {"alpha": a, "l1_ratio": getattr(pipe.named_steps["est"], "l1_ratio_", 1.0)}
            from sklearn.linear_model import ElasticNet
            pipe = Pipeline([("prep", linear_preprocessor()),
                             ("est", ElasticNet(alpha=a, l1_ratio=best_params[name]["l1_ratio"], max_iter=2000, tol=1e-3, selection="random", random_state=SEED))])
        if name in ("Lasso (log)", "Elastic Net (log)"):
            big = subsample_groups(train, 150_000)  # coordinate descent is slow on 640k dense rows
            model = LogTargetModel(pipe, name).fit(big[FEATURES], big[TARGET])
            best_params[name]["final_fit_rows"] = len(big)
        else:
            model = LogTargetModel(pipe, name).fit(X_tr, y_tr)
        if name == "Ridge (log)":
            best_params[name] = {"alpha": float(pipe.named_steps["est"].alpha_)}
        record(name, "Linear", model, time.time() - t)

    # 3. Gamma GLM with log link (actuarial standard for premium / severity)
    t = time.time()
    glm = Pipeline([("prep", linear_preprocessor()), ("est", GammaRegressor(alpha=1e-4, max_iter=1000))])
    glm.fit(X_tr, y_tr)
    record("Gamma GLM (log link)", "GLM", glm, time.time() - t)

    # 4. Decision tree (tuned)
    t = time.time()
    dt_search = RandomizedSearchCV(
        Pipeline([("prep", tree_preprocessor()), ("est", DecisionTreeRegressor(random_state=SEED))]),
        {"est__max_depth": [8, 12, 16, 20, 25, None], "est__min_samples_leaf": [5, 10, 20, 50, 100]},
        n_iter=6 if fast else 15, cv=GroupKFold(3), scoring="neg_root_mean_squared_error", random_state=SEED)
    dt_search.fit(tune[FEATURES], np.log(tune[TARGET]), groups=tune[GROUP])
    best_params["Decision Tree"] = {k.replace("est__", ""): v for k, v in dt_search.best_params_.items()}
    model = LogTargetModel(dt_search.best_estimator_, "Decision Tree").fit(X_tr, y_tr)
    record("Decision Tree", "Tree", model, time.time() - t)

    # 5. Random forest (tuned; same search space as the original project, plus log target)
    t = time.time()
    rf_tune = subsample_groups(train, 20_000 if fast else 60_000)
    rf_search = RandomizedSearchCV(
        Pipeline([("prep", tree_preprocessor()), ("est", RandomForestRegressor(random_state=SEED, n_jobs=2))]),
        {"est__n_estimators": [100, 200], "est__max_depth": [None, 15, 20],
         "est__min_samples_leaf": [3, 5, 10], "est__max_features": ["sqrt", 0.3, 0.5],
         "est__max_samples": [0.6, 0.8, None]},
        n_iter=4 if fast else 12, cv=GroupKFold(3), scoring="neg_root_mean_squared_error", random_state=SEED)
    rf_search.fit(rf_tune[FEATURES], np.log(rf_tune[TARGET]), groups=rf_tune[GROUP])
    bp = {k.replace("est__", ""): v for k, v in rf_search.best_params_.items()}
    best_params["Random Forest"] = bp
    rf_final = Pipeline([("prep", tree_preprocessor()), ("est", RandomForestRegressor(
        random_state=SEED, n_jobs=2, **{**bp, "n_estimators": 150,
                                        "max_samples": min(bp["max_samples"] or 1.0, 0.3)}))])
    model = LogTargetModel(rf_final, "Random Forest").fit(X_tr, y_tr)
    record("Random Forest", "Tree ensemble", model, time.time() - t,
           "Final fit uses <=30% bootstrap samples per tree to fit in 2 CPU cores")
    del model, rf_final

    # 6. HistGradientBoosting (tuned with Optuna)
    t = time.time()
    hgb_tune = tune.copy()

    def hgb_make(trial):
        return Pipeline([("prep", tree_preprocessor()), ("est", HistGradientBoostingRegressor(
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 31, 255, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
            max_iter=800, early_stopping=True, categorical_features=list(range(len(CATEGORICAL))),
            random_state=SEED))])

    def hgb_obj(trial):
        def f(tr, va):
            pipe = hgb_make(trial)
            pipe.fit(tr[FEATURES], np.log(tr[TARGET]))
            return pipe.predict(va[FEATURES])
        return cv_rmsle(f, hgb_tune)

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(hgb_obj, n_trials=max(4, n_trials // 2), timeout=budget // 2)
    best_params["HistGradientBoosting"] = {**st.best_params, "optuna_trials": len(st.trials)}
    model = LogTargetModel(hgb_make(optuna.trial.FixedTrial(st.best_params)), "HGB").fit(X_tr, y_tr)
    record("HistGradientBoosting", "Boosting", model, time.time() - t)

    # 7. XGBoost (tuned with Optuna)
    t = time.time()
    rounds = {}

    def xgb_obj(trial):
        params, its = xgb_params(trial), []

        def f(tr, va):
            b = fit_xgb(params, tr, va)
            its.append(b.best_iteration)
            return b.predict(xgb.DMatrix(va[FEATURES], enable_categorical=True), iteration_range=(0, b.best_iteration + 1))
        s = cv_rmsle(f, tune)
        rounds[trial.number] = int(np.mean(its) * 1.1) + 1
        return s

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(xgb_obj, n_trials=n_trials, timeout=budget)
    xp = xgb_params(optuna.trial.FixedTrial(st.best_params))
    nround = rounds[st.best_trial.number]
    best_params["XGBoost"] = {**st.best_params, "num_boost_round": nround, "optuna_trials": len(st.trials)}
    b = fit_xgb(xp, train, rounds=nround)
    wx = BoosterWrap(b, "xgb", 1.0)
    wx.smear_ = smear(train, wx.predict_log(train))
    record("XGBoost", "Boosting", wx, time.time() - t)

    # 8. LightGBM (tuned with Optuna)
    t = time.time()

    def lgb_obj(trial):
        params, its = lgb_params(trial), []

        def f(tr, va):
            b = fit_lgb(params, tr, va)
            its.append(b.best_iteration)
            return b.predict(va[FEATURES], num_iteration=b.best_iteration)
        s = cv_rmsle(f, tune)
        rounds[("lgb", trial.number)] = int(np.mean(its) * 1.1) + 1
        return s

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(lgb_obj, n_trials=n_trials, timeout=budget)
    lp = lgb_params(optuna.trial.FixedTrial(st.best_params))
    lround = rounds[("lgb", st.best_trial.number)]
    best_params["LightGBM"] = {**st.best_params, "num_boost_round": lround, "optuna_trials": len(st.trials)}
    lgb_model = fit_lgb(lp, train, rounds=lround)
    wl = BoosterWrap(lgb_model, "lgb", 1.0)
    wl.smear_ = smear(train, wl.predict_log(train))
    record("LightGBM", "Boosting", wl, time.time() - t)

    # 9. Neural network (MLP) on a subsample
    t = time.time()
    mlp_sub = subsample_groups(train, 60_000 if fast else 200_000)
    mlp = Pipeline([("prep", linear_preprocessor()), ("est", MLPRegressor(
        hidden_layer_sizes=(128, 64), alpha=1e-4, learning_rate_init=1e-3, batch_size=512,
        early_stopping=True, max_iter=150, random_state=SEED))])
    model = LogTargetModel(mlp, "MLP").fit(mlp_sub[FEATURES], mlp_sub[TARGET])
    best_params["Neural Network (MLP)"] = {"hidden_layer_sizes": [128, 64], "alpha": 1e-4, "train_rows": len(mlp_sub)}
    record("Neural Network (MLP)", "Neural net", model, time.time() - t, f"Trained on {len(mlp_sub):,} rows")

    # 10. Simple ensemble of the two strongest boosters
    p = np.sqrt(preds["LightGBM"] * preds["XGBoost"])
    preds["Ensemble (LightGBM + XGBoost)"] = p
    m = metrics(test[TARGET], p)
    m.update(model="Ensemble (LightGBM + XGBoost)", family="Ensemble", fit_seconds=0.0,
             notes="Geometric mean of the two boosted models")
    results.append(m)
    comp = pd.DataFrame(results).sort_values("R2_log", ascending=False)
    comp.to_csv(RES / "model_comparison.csv", index=False)
    (RES / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    # Segment metrics for every model
    seg = []
    for name, p in preds.items():
        s = segment_metrics(test, p)
        s["model"] = name
        seg.append(s)
    pd.concat(seg).to_csv(RES / "segment_metrics.csv", index=False)

    # Test-set predictions sample for app plots
    samp = test.sample(20_000, random_state=SEED).index
    out = test.loc[samp, ["premium", "coverage", "vehicle_type", "usage"]].copy()
    for name in ["LightGBM", "Random Forest", "Ridge (log)", "Gamma GLM (log link)", "Tariff lookup (baseline)"]:
        out[name] = preds[name][samp]
    out.to_parquet(RES / "test_predictions_sample.parquet", index=False)

    # ---------------------------------------------------- leakage check (random vs grouped split)
    log("leakage check")
    rows = []
    sub = subsample_groups(df, 300_000)
    for split in ["Random rows (original approach)", "Grouped by vehicle (this repo)"]:
        if split.startswith("Random"):
            tr, te = train_test_split(sub, test_size=0.2, random_state=SEED)
        else:
            tr, te = grouped_split(sub)
        b = fit_lgb(lp, tr, rounds=lround)
        w = BoosterWrap(b, "lgb", 1.0)
        w.smear_ = smear(tr, w.predict_log(tr))
        m = metrics(te[TARGET], w.predict(te))
        m["split"] = split
        m["shared_vehicles_pct"] = float(te[GROUP].isin(set(tr[GROUP])).mean())
        rows.append(m)
    pd.DataFrame(rows).to_csv(RES / "leakage_check.csv", index=False)

    # ---------------------------------------------------- zero insured value: impute vs coverage flag
    log("imputation check")

    def impute_like_original(d, ref):
        """Original approach: replace 0 insured value by the median of same make, vehicle type, production year."""
        d = d.copy()
        ref = ref[ref["coverage"] == "Comprehensive"].assign(py=lambda x: x["begin_year"] - x["vehicle_age"])
        med = ref.groupby(["make", "vehicle_type", "py"], observed=True)["log_insured_value"].median()
        key = pd.MultiIndex.from_arrays([d["make"], d["vehicle_type"], d["begin_year"] - d["vehicle_age"]])
        fill = pd.Series(med.reindex(key).values, index=d.index)
        z = d["coverage"] == "Liability only"
        d.loc[z, "log_insured_value"] = fill[z]
        return d[d["log_insured_value"].notna()]

    rows = []
    no_cov = [f for f in FEATURES if f != "coverage"]
    variants = {
        "Original: impute zeros with group median": (impute_like_original(train, train),
                                                     impute_like_original(test, train), no_cov),
        "Expanded: keep zeros + coverage flag": (train, test, FEATURES),
    }
    for label, (tr_v, te_v, feats) in variants.items():
        cats = [c for c in feats if c in CATEGORICAL]
        nums = [c for c in feats if c in NUMERIC]
        r = LogTargetModel(Pipeline([("prep", linear_preprocessor(cats, nums)),
                                     ("est", RidgeCV(alphas=np.logspace(-4, 3, 15)))]))
        r.fit(tr_v[feats], tr_v[TARGET])
        m = metrics(te_v[TARGET], r.predict(te_v[feats]))
        m.update(approach=label, model="Ridge (log)", test_rows=len(te_v))
        rows.append(m)
    pd.DataFrame(rows).to_csv(RES / "imputation_check.csv", index=False)

    # ---------------------------------------------------- feature-set ablation
    log("ablation")
    orig_feats = ["sex", "seats_num", "ccm_ton", "insurance_type", "log_insured_value", "vehicle_age",
                  "vehicle_type", "usage"]
    sets = {
        "Original RF features": orig_feats,
        "+ coverage type": orig_feats + ["coverage"],
        "+ policy duration & make": orig_feats + ["coverage", "duration_days", "make"],
        "All features (incl. claim history)": FEATURES,
    }
    rows = []
    for label, feats in sets.items():
        cats = [c for c in feats if c in CATEGORICAL]
        dtr = lgb.Dataset(train[feats], np.log(train[TARGET]), categorical_feature=cats)
        b = lgb.train(lp, dtr, num_boost_round=lround)
        pl = b.predict(train[feats])
        s = float(np.mean(np.exp(np.log(train[TARGET]) - pl)))
        m = metrics(test[TARGET], np.exp(b.predict(test[feats])) * s)
        m["feature_set"] = label
        m["n_features"] = len(feats)
        rows.append(m)
    pd.DataFrame(rows).to_csv(RES / "ablation.csv", index=False)

    # ---------------------------------------------------- temporal holdout
    log("temporal holdout")
    cut = pd.Timestamp("2017-07-01")
    tr_t, te_t = df[df["insr_begin"] < cut], df[df["insr_begin"] >= cut]
    rows = []
    keys = ["vehicle_type", "usage", "coverage"]
    lut = tr_t.groupby(keys, observed=True)[TARGET].median().rename("p").reset_index()
    p = te_t[keys].merge(lut, on=keys, how="left")["p"].fillna(tr_t[TARGET].median()).values
    rows.append({**metrics(te_t[TARGET], p), "model": "Tariff lookup (baseline)"})
    g = Pipeline([("prep", linear_preprocessor()), ("est", GammaRegressor(alpha=1e-4, max_iter=1000))])
    g.fit(tr_t[FEATURES], tr_t[TARGET])
    rows.append({**metrics(te_t[TARGET], g.predict(te_t[FEATURES])), "model": "Gamma GLM (log link)"})
    r = LogTargetModel(Pipeline([("prep", linear_preprocessor()), ("est", RidgeCV(alphas=np.logspace(-4, 3, 15)))]))
    r.fit(tr_t[FEATURES], tr_t[TARGET])
    rows.append({**metrics(te_t[TARGET], r.predict(te_t[FEATURES])), "model": "Ridge (log)"})
    b = fit_xgb(xp, tr_t, rounds=nround)
    w = BoosterWrap(b, "xgb", 1.0)
    w.smear_ = smear(tr_t, w.predict_log(tr_t))
    rows.append({**metrics(te_t[TARGET], w.predict(te_t)), "model": "XGBoost"})
    b = fit_lgb(lp, tr_t, rounds=lround)
    w = BoosterWrap(b, "lgb", 1.0)
    w.smear_ = smear(tr_t, w.predict_log(tr_t))
    rows.append({**metrics(te_t[TARGET], w.predict(te_t)), "model": "LightGBM"})
    tt = pd.DataFrame(rows)
    tt["train_rows"], tt["test_rows"] = len(tr_t), len(te_t)
    tt["cutoff"] = str(cut.date())
    tt.to_csv(RES / "temporal_holdout.csv", index=False)

    # ---------------------------------------------------- explainability (SHAP via LightGBM)
    log("SHAP")
    xs = test.sample(4000, random_state=SEED)
    contrib = lgb_model.predict(xs[FEATURES], pred_contrib=True)
    shap_df = pd.DataFrame(contrib[:, :-1], columns=FEATURES)
    shap_df.to_parquet(RES / "shap_values.parquet", index=False)
    xs[FEATURES + ["premium"]].reset_index(drop=True).to_parquet(RES / "shap_features.parquet", index=False)
    imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(shap_df.values).mean(0),
                        "gain": lgb_model.feature_importance("gain") / lgb_model.feature_importance("gain").sum()})
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(RES / "feature_importance.csv", index=False)

    # ---------------------------------------------------- estimator models for the app
    log("final estimator models")
    full_lgb = fit_lgb(lp, df, rounds=lround)
    full_lgb.save_model(str(MOD / "lgbm_point.txt"))
    w = BoosterWrap(full_lgb, "lgb", 1.0)
    meta = {"smear": smear(df, w.predict_log(df)), "features": FEATURES, "categorical": CATEGORICAL,
            "categories": {c: list(df[c].cat.categories) for c in CATEGORICAL}, "num_boost_round": lround,
            "trained_rows": len(df)}
    for alpha in (0.1, 0.9):
        qp = {**lp, "objective": "quantile", "alpha": alpha}
        qb = fit_lgb(qp, df, rounds=lround)
        qb.save_model(str(MOD / f"lgbm_q{int(alpha * 100)}.txt"))
    (MOD / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small tuning budget for a quick smoke test")
    main(**vars(ap.parse_args()))
