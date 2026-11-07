import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import BLUE, GREY, ORANGE, footer, load_result

st.title("🔍 What drives premiums")

LABELS = {
    "log_insured_value": "Insured value", "coverage": "Coverage type", "usage": "Usage",
    "vehicle_type": "Vehicle type", "ccm_ton": "Engine size / tonnage", "vehicle_age": "Vehicle age",
    "seats_num": "Seats", "carrying_capacity": "Carrying capacity", "duration_days": "Policy length (days)",
    "make": "Make", "insurance_type": "Insurance type", "sex": "Policyholder type",
    "prior_policies": "Earlier policies with insurer", "prior_claim": "Claim on previous policy",
    "customer_tenure": "Years as customer", "season": "Start season",
}

imp = load_result("feature_importance.csv")
shap = load_result("shap_values.parquet")
feats = load_result("shap_features.parquet")

st.markdown(
    "We explain the LightGBM model with **SHAP values** (Lundberg & Lee, 2017). For each policy, a SHAP value "
    "says how much a feature pushed the predicted log premium up or down compared with the average policy. "
    "Averaging their size across 4,000 test policies ranks the features."
)

c1, c2 = st.columns([1, 1])
with c1:
    d = imp.assign(label=imp.feature.map(LABELS)).sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(x=d["mean_abs_shap"], y=d["label"], orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Mean |SHAP| (log premium)", height=520, xaxis_title="Average impact on log premium")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.markdown("**Original DataCon top 5 (Random Forest impurity importance)**")
    orig = pd.DataFrame({"Feature": ["Insured value", "Engine capacity", "Production year",
                                     "Usage: Fare Paying Passengers*", "Number of seats"],
                         "Score": [0.658, 0.112, 0.057, 0.043, 0.032]})
    st.dataframe(orig, hide_index=True, use_container_width=True)
    st.caption(
        "*Reported as \"Vehicle Usage\", but the score belongs to one usage category only. These scores also "
        "came from a working notebook that read importances from the untuned forest; the final notebook had the "
        "corrected values but the report kept the old ones (see the changelog)."
    )
    st.markdown(
        """
**What changed in the story**

- Insured value still matters most, but much of its importance was really telling
  **comprehensive from liability-only** cover apart. With coverage as its own feature, that is explicit.
- **Usage and vehicle type** carry most of the remaining signal, which matches how tariffs are built.
- **Claim on the previous policy** and **years as a customer** matter less than expected. The insurer applies a
  no-claim discount at renewal (Terefe, 2023b), but our proxy only sees claims recorded in this dataset, and the
  discount is capped by an age loading, so its signal is weak here.
"""
    )

st.subheader("How one feature moves the prediction")
num_feats = [f for f in imp.feature if pd.api.types.is_numeric_dtype(feats[f])]
cat_feats = [f for f in imp.feature if f not in num_feats]
f = st.selectbox("Feature", list(imp.feature), format_func=lambda x: LABELS.get(x, x))
x = feats[f]
y = shap[f]
if f in num_feats:
    xx = np.expm1(x) if f == "log_insured_value" else x
    fig = px.scatter(x=xx, y=y, color=feats["coverage"].astype(str), opacity=0.45, log_x=(f == "log_insured_value"),
                     color_discrete_map={"Comprehensive": BLUE, "Liability only": ORANGE},
                     labels={"x": LABELS[f], "y": "SHAP value (log premium)", "color": "Coverage"})
    fig.update_traces(marker=dict(size=8, line=dict(width=0)))
else:
    g = pd.DataFrame({"level": x.astype(str), "shap": y}).groupby("level")["shap"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= 15].sort_values("mean")
    fig = go.Figure(go.Bar(x=g["mean"], y=g["level"], orientation="h",
                           marker_color=[ORANGE if v < 0 else BLUE for v in g["mean"]],
                           customdata=g["count"], hovertemplate="%{y}: %{x:+.3f} (n=%{customdata})<extra></extra>"))
    fig.update_layout(xaxis_title="Average SHAP value (log premium)", height=max(320, 26 * len(g)))
fig.add_hline(y=0, line_color=GREY, line_width=1) if f in num_feats else fig.add_vline(x=0, line_color=GREY, line_width=1)
fig.update_layout(title=f"Effect of {LABELS[f].lower()} on the predicted premium")
st.plotly_chart(fig, use_container_width=True)
st.caption("A SHAP value of +0.1 on the log scale means roughly a 10% higher premium; -0.1 roughly 10% lower.")
footer()
