import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import BLUE, CURRENCY, GREY, ORANGE, ROOT, footer, load_policies

st.title("🧮 Premium estimator")
st.markdown(
    "Describe a vehicle and policy. The LightGBM model (trained on all 800k policies) estimates the annual "
    "premium, and two quantile models give a range where 80% of similar policies fall. "
    "**This is a teaching demo based on 2011 to 2018 Ethiopian data, not a real quote.**"
)


@st.cache_resource
def load_models():
    meta = json.loads((ROOT / "models/meta.json").read_text())
    point = lgb.Booster(model_file=str(ROOT / "models/lgbm_point.txt"))
    q10 = lgb.Booster(model_file=str(ROOT / "models/lgbm_q10.txt"))
    q90 = lgb.Booster(model_file=str(ROOT / "models/lgbm_q90.txt"))
    return meta, point, q10, q90


@st.cache_data
def typical_values():
    df = load_policies()
    num = df.groupby("vehicle_type", observed=True)[["seats_num", "carrying_capacity", "ccm_ton"]].median()
    use = df.groupby("vehicle_type", observed=True)["usage"].agg(lambda s: s.value_counts().index[0])
    make = df.groupby("vehicle_type", observed=True)["make"].agg(lambda s: s.value_counts().index[0])
    iv = df[df.coverage == "Comprehensive"].groupby("vehicle_type", observed=True)["insured_value"].median()
    keys = ["vehicle_type", "usage", "coverage"]
    lut = df.groupby(keys, observed=True)["premium"].median()
    return num, use, make, iv, lut


meta, point, q10, q90 = load_models()
cats = meta["categories"]
num, use, make, iv_med, lut = typical_values()


def frame(rows: list[dict]) -> pd.DataFrame:
    X = pd.DataFrame(rows)[meta["features"]]
    for c in meta["categorical"]:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X


def predict(X):
    p = np.exp(point.predict(X)) * meta["smear"]
    lo, hi = np.exp(q10.predict(X)), np.exp(q90.predict(X))
    return p, np.minimum(lo, p), np.maximum(hi, p)


with st.form("quote"):
    c1, c2, c3 = st.columns(3)
    vt = c1.selectbox("Vehicle type", cats["vehicle_type"],
                      index=cats["vehicle_type"].index("Automobile"))
    coverage = c2.radio("Coverage", cats["coverage"], horizontal=True)
    usage = c3.selectbox("Usage", cats["usage"], index=cats["usage"].index(use.get(vt, "Private")))

    c4, c5, c6 = st.columns(3)
    default_iv = float(round(iv_med.get(vt, 500_000), -3))
    insured_value = c4.number_input(f"Insured value ({CURRENCY})", min_value=0.0, value=default_iv, step=10_000.0,
                                    disabled=(coverage == "Liability only"),
                                    help="Liability-only cover does not insure the vehicle, so the value is 0.")
    vehicle_age = c5.slider("Vehicle age at policy start (years)", 0, 40, 6)
    mk = c6.selectbox("Make", cats["make"], index=cats["make"].index(make.get(vt, "TOYOTA")))

    with st.expander("More details (pre-filled with typical values for this vehicle type)"):
        d1, d2, d3 = st.columns(3)
        seats = d1.number_input("Seats", 0, 200, int(num.loc[vt, "seats_num"]))
        ccm = d2.number_input("Engine size (cc) or tonnage", 0, 20_000, int(num.loc[vt, "ccm_ton"]))
        cap = d3.number_input("Carrying capacity", 0, 100_000,
                              int(0 if np.isnan(num.loc[vt, "carrying_capacity"]) else num.loc[vt, "carrying_capacity"]))
        d4, d5, d6 = st.columns(3)
        ins_type = d4.selectbox("Insurance type", cats["insurance_type"],
                                index=cats["insurance_type"].index("Commercial" if usage != "Private" else "Private"))
        sex = d5.selectbox("Policyholder", cats["sex"], index=cats["sex"].index("Legal entity" if usage != "Private" else "Male"))
        duration = d6.slider("Policy length (days)", 30, 365, 364)
        d7, d8, d9 = st.columns(3)
        prior_pol = d7.number_input("Earlier policies with this insurer", 0, 7, 1)
        prior_claim = d8.radio("Claim on previous policy?", ["No", "Yes"], horizontal=True) == "Yes"
        tenure = d9.number_input("Years as a customer", 0, 40, 3)
        season = st.selectbox("Start season", cats["season"])
    submitted = st.form_submit_button("Estimate premium", type="primary")

iv = 0.0 if coverage == "Liability only" else insured_value
row = dict(sex=sex, insurance_type=ins_type, coverage=coverage, vehicle_type=vt, usage=usage, make=mk,
           season=season, log_insured_value=np.log1p(iv), vehicle_age=vehicle_age, seats_num=seats,
           carrying_capacity=cap, ccm_ton=ccm, duration_days=duration, prior_policies=prior_pol,
           prior_claim=int(prior_claim), customer_tenure=tenure)
X = frame([row])
p, lo, hi = (v[0] for v in predict(X))
tariff = lut.get((vt, usage, coverage), np.nan)

k1, k2, k3 = st.columns(3)
k1.metric("Estimated premium", f"{p:,.0f} {CURRENCY}")
k2.metric("Likely range (10th to 90th percentile)", f"{lo:,.0f} to {hi:,.0f}")
if coverage == "Comprehensive" and iv > 0:
    k3.metric("Implied rate (% of insured value)", f"{p / iv:.2%}")
elif not np.isnan(tariff):
    k3.metric("Median premium for this type and usage", f"{tariff:,.0f} {CURRENCY}")

c1, c2 = st.columns(2)
with c1:
    contrib = point.predict(X, pred_contrib=True)[0]
    labels = {"log_insured_value": "Insured value", "ccm_ton": "Engine size", "vehicle_age": "Vehicle age",
              "seats_num": "Seats", "carrying_capacity": "Capacity", "duration_days": "Policy length",
              "prior_policies": "Earlier policies", "prior_claim": "Previous claim", "customer_tenure": "Tenure",
              "insurance_type": "Insurance type", "vehicle_type": "Vehicle type", "sex": "Policyholder"}
    s = pd.Series(contrib[:-1], index=[labels.get(f, f.replace("_", " ").capitalize()) for f in meta["features"]])
    s = s[s.abs().sort_values(ascending=False).index][:8][::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[BLUE if v > 0 else ORANGE for v in s.values],
                           hovertemplate="%{y}: %{x:+.2f} on log scale<extra></extra>"))
    fig.add_vline(x=0, line_color=GREY, line_width=1)
    fig.update_layout(title="Why this estimate? (top 8 factors)", xaxis_title="Push on log premium vs average policy",
                      height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Blue bars raise the estimate, orange bars lower it.")
with c2:
    if coverage == "Comprehensive":
        grid = np.geomspace(max(iv / 5, 20_000), max(iv * 3, 100_000), 40)
        xs = frame([{**row, "log_insured_value": np.log1p(g)} for g in grid])
        title, xlab, xv = "What if the insured value changes?", f"Insured value ({CURRENCY})", grid
    else:
        grid = np.arange(0, 31)
        xs = frame([{**row, "vehicle_age": g} for g in grid])
        title, xlab, xv = "What if the vehicle were older or newer?", "Vehicle age (years)", grid
    pp, ll, hh = predict(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[xv, xv[::-1]], y=np.r_[hh, ll[::-1]], fill="toself", fillcolor="rgba(42,120,214,0.15)",
                             line=dict(width=0), hoverinfo="skip", name="80% range"))
    fig.add_trace(go.Scatter(x=xv, y=pp, mode="lines", line=dict(color=BLUE, width=2), name="Estimate",
                             hovertemplate=f"%{{x:,.0f}} → %{{y:,.0f}} {CURRENCY}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=f"Premium ({CURRENCY})", height=380,
                      xaxis_type="log" if coverage == "Comprehensive" else "linear")
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Range comes from LightGBM quantile models at the 10th and 90th percentiles. Amounts are in the dataset's "
    "currency (most likely Ethiopian Birr) at 2011 to 2018 price levels."
)
footer()
