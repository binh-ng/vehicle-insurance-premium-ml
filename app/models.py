import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import BLUE, COVERAGE_COLORS, CURRENCY, GREY, ORANGE, footer, load_result

st.title("🏁 Model comparison")

comp = load_result("model_comparison.csv")
seg = load_result("segment_metrics.csv")

METRICS = {
    "MdAPE": ("Median % error", "lower is better", True),
    "Within_20pct": ("Share of predictions within 20%", "higher is better", False),
    "R2_log": ("R² on log premium", "higher is better", False),
    "R2": ("R² on premium", "higher is better", False),
    "MAE": (f"Mean absolute error ({CURRENCY})", "lower is better", True),
    "RMSE": (f"RMSE ({CURRENCY})", "lower is better", True),
}

tab1, tab2, tab3, tab4 = st.tabs(["Expanded models", "Original DataCon results", "Robustness checks", "Predicted vs actual"])

with tab1:
    st.markdown(
        "All models use the same cleaned data, the same features and the **same test set: 20% of vehicles "
        "the models never saw during training**. Tree and neural models were tuned with cross-validation "
        "grouped by vehicle."
    )
    c1, c2 = st.columns([1, 1])
    metric = c1.selectbox("Metric", list(METRICS), format_func=lambda k: f"{METRICS[k][0]} ({METRICS[k][1]})")
    segment = c2.radio("Policies", ["All", "Comprehensive", "Liability only"], horizontal=True)
    if segment == "All":
        d = comp.copy()
    else:
        d = seg[seg.coverage == segment].copy()
    label, hint, lower = METRICS[metric]
    d = d.sort_values(metric, ascending=not lower)
    best = d.iloc[-1]["model"]
    colors = [BLUE if m == best else GREY for m in d["model"]]
    pct = metric in ("MdAPE", "Within_20pct")
    fig = go.Figure(go.Bar(
        x=d[metric], y=d["model"], orientation="h", marker_color=colors,
        text=[f"{v:.1%}" if pct else (f"{v:.3f}" if "R2" in metric else f"{v:,.0f}") for v in d[metric]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{text}<extra></extra>"))
    fig.update_layout(title=f"{label} · {segment.lower()} policies · best highlighted", height=480,
                      xaxis_title=label, yaxis_title="")
    if pct:
        fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    show = comp[["model", "family", "R2", "R2_log", "MdAPE", "Within_20pct", "MAE", "RMSE", "fit_seconds"]]
    st.dataframe(
        show.sort_values("MdAPE"), hide_index=True, use_container_width=True,
        column_config={
            "R2": st.column_config.NumberColumn("R²", format="%.3f"),
            "R2_log": st.column_config.NumberColumn("R² (log)", format="%.3f"),
            "MdAPE": st.column_config.NumberColumn("Median % error", format="percent"),
            "Within_20pct": st.column_config.NumberColumn("Within 20%", format="percent"),
            "MAE": st.column_config.NumberColumn("MAE", format="%.0f"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.0f"),
            "fit_seconds": st.column_config.NumberColumn("Fit time (s)", format="%.0f"),
        })
    st.caption(
        "Why several metrics? Premiums range from under 100 to millions, so RMSE and R² on the raw scale are "
        "driven by a few very large fleet policies. Median % error and R² on the log scale describe the typical policy."
    )
    with st.expander("Tuned hyperparameters"):
        st.json(load_result("best_params.json"))

with tab2:
    st.markdown(
        "These are the results we presented at DataCon 2025, exactly as reported, with one correction: the "
        "report's comparison table showed the polynomial model's scores in the Ridge row (46.15% / 28.66%). "
        "The Ridge scores were 46.16% / 42.24%, as on the poster."
    )
    orig = pd.DataFrame([
        ["Random Forest (baseline)", 0.906, 0.716, 4009.18, 6976.30, 2084.79, 4045.05],
        ["Random Forest (tuned)", 0.783, 0.742, 6109.13, 6641.06, 3483.67, 3864.53],
        ["MLR additive", 0.1837, 0.3983, 15100.52, 8762.19, 4728.66, 4648.09],
        ["MLR without influential points", 0.4616, 0.4224, 8004.63, 8585.18, 4330.44, 4554.07],
        ["MLR + interactions", 0.5221, -0.5598, 7541.89, 14107.61, 4024.34, 4331.47],
        ["MLR + polynomial", 0.4615, 0.2866, 8005.67, 9540.70, 4246.61, 4512.21],
        ["Ridge (final MLR)", 0.4616, 0.4224, 8004.64, 8585.17, 4330.44, 4554.07],
    ], columns=["Model", "Train R²", "Test R²", "Train RMSE", "Test RMSE", "Train MAE", "Test MAE"])
    st.dataframe(orig, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Train R²", "Test R²"]})
    rep = load_result("replication.json")
    lo_r2, hi_r2 = rep["test_r2_range"]
    st.markdown(
        f"**Re-run check.** We re-ran the tuned random forest with the original code path and best parameters "
        f"(`scripts/replicate_original.py`). Over five different 10% samples, test R² ranged from **{lo_r2:.2f} to "
        f"{hi_r2:.2f}**, so the reported 0.742 is reproducible, but a single 10% sample moves the score by several points."
    )
    g = pd.Series(rep["importance_tuned_grouped"]).sort_values()
    names = {"insured_value": "Insured value", "ccm_ton": "Engine capacity", "usage": "Usage (all categories)",
             "vehicle_type": "Vehicle type (all)", "production_year": "Production year", "seats_num": "Seats",
             "sex": "Policyholder type", "insurance_type": "Insurance type"}
    fig = go.Figure(go.Bar(x=g.values, y=[names.get(k, k) for k in g.index], orientation="h", marker_color=BLUE,
                           text=[f"{v:.1%}" for v in g.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(title="Tuned random forest importance, dummies summed per variable (corrected)",
                      xaxis_tickformat=".0%", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The report's top-5 list came from the untuned forest and counted one usage category as \"Vehicle Usage\". "
               "Summed over all categories, usage is the third most important variable.")
    st.warning(
        "These numbers are not directly comparable with the expanded models. The Random Forest used a 10% sample "
        "and the MLR used a filtered, imputed dataset with a different test set, and both used a random row split "
        "in which the same vehicle can appear in both training and test data."
    )
    fig = go.Figure()
    fig.add_bar(name="Original (DataCon 2025)", x=["Random Forest (tuned)", "Ridge / MLR"], y=[0.742, 0.4224],
                marker_color=GREY, text=["74.2%", "42.2%"], textposition="outside")
    new_rf = comp.loc[comp.model == "Random Forest", "R2"].iloc[0]
    new_ridge = comp.loc[comp.model == "Ridge (log)", "R2"].iloc[0]
    fig.add_bar(name="Expanded version", x=["Random Forest (tuned)", "Ridge / MLR"], y=[new_rf, new_ridge],
                marker_color=BLUE, text=[f"{new_rf:.1%}", f"{new_ridge:.1%}"], textposition="outside")
    fig.update_layout(barmode="group", title="Test R² on premium: original vs expanded pipeline",
                      yaxis_tickformat=".0%", yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("1. Does the way we split the data change the score?")
    lk = load_result("leakage_check.csv")
    st.markdown(
        f"With a random row split, **{lk.iloc[0]['shared_vehicles_pct']:.0%} of test policies belong to vehicles "
        "already seen in training** (the same car renewed in another year). Splitting by vehicle removes that overlap."
    )
    st.dataframe(lk[["split", "R2", "R2_log", "MdAPE", "MAE", "shared_vehicles_pct"]], hide_index=True,
                 use_container_width=True,
                 column_config={"MdAPE": st.column_config.NumberColumn("Median % error", format="percent"),
                                "shared_vehicles_pct": st.column_config.NumberColumn("Test rows from seen vehicles", format="percent"),
                                "R2": st.column_config.NumberColumn("R²", format="%.3f"),
                                "R2_log": st.column_config.NumberColumn("R² (log)", format="%.3f"),
                                "MAE": st.column_config.NumberColumn(format="%.0f")})

    st.caption("Here the scores barely move, because renewal premiums change from year to year anyway. The grouped "
               "split is still the right design: it measures how well the model prices a vehicle it has never seen.")

    st.subheader("2. Train on the past, test on the future")
    tt = load_result("temporal_holdout.csv")
    st.markdown(
        f"Models trained on policies starting before **{tt['cutoff'].iloc[0]}** "
        f"({tt['train_rows'].iloc[0]:,} rows) and tested on later policies ({tt['test_rows'].iloc[0]:,} rows)."
    )
    t = tt.sort_values("MdAPE", ascending=False)
    fig = go.Figure(go.Bar(x=t["MdAPE"], y=t["model"], orientation="h",
                           marker_color=[BLUE if i == len(t) - 1 else GREY for i in range(len(t))],
                           text=[f"{v:.1%}" for v in t["MdAPE"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{text}<extra></extra>"))
    fig.update_layout(title="Median % error on future policies (lower is better)", xaxis_tickformat=".0%", height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Impute zero insured values, or keep them?")
    ic = load_result("imputation_check.csv")
    fig = go.Figure(go.Bar(x=ic["approach"], y=ic["MdAPE"], marker_color=[GREY, BLUE],
                           text=[f"{v:.1%}" for v in ic["MdAPE"]], textposition="outside",
                           hovertemplate="%{x}: %{text}<extra></extra>"))
    fig.update_layout(title="Ridge (log) median % error on the same test vehicles (lower is better)",
                      yaxis_tickformat=".0%", yaxis_range=[0, ic["MdAPE"].max() * 1.25], height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Filling the zeros with a median vehicle value (the original approach) tells the model that a "
               "liability-only policy insures an average car, which it does not.")

    st.subheader("4. Which features add value?")
    ab = load_result("ablation.csv")
    fig = go.Figure(go.Bar(x=ab["feature_set"], y=ab["MdAPE"], marker_color=BLUE,
                           text=[f"{v:.1%}" for v in ab["MdAPE"]], textposition="outside",
                           hovertemplate="%{x}: %{text}<extra></extra>"))
    fig.update_layout(title="LightGBM median % error as features are added (lower is better)",
                      yaxis_tickformat=".0%", yaxis_range=[0, ab["MdAPE"].max() * 1.25])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Starts from the feature set of the original Random Forest, all evaluated on the vehicle-grouped split. "
               "The coverage flag adds little for trees, which can already split on insured value = 0; it matters for "
               "linear models (check 3). Policy length and make give the biggest gain.")

with tab4:
    pr = load_result("test_predictions_sample.parquet")
    models = [c for c in pr.columns if c not in ("premium", "coverage", "vehicle_type", "usage")]
    mdl = st.selectbox("Model", models)
    s = pr.sample(6000, random_state=1)
    fig = px.scatter(s, x="premium", y=mdl, color="coverage", log_x=True, log_y=True, opacity=0.4,
                     color_discrete_map=COVERAGE_COLORS, hover_data=["vehicle_type", "usage"],
                     labels={"premium": f"Actual premium ({CURRENCY})", mdl: f"Predicted ({CURRENCY})",
                             "coverage": "Coverage"},
                     title=f"{mdl}: predicted vs actual (6,000 test policies)")
    lo, hi = s["premium"].min(), s["premium"].max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#52514e", dash="dash", width=1.5),
                             name="Perfect prediction", hoverinfo="skip"))
    fig.update_traces(marker=dict(size=8, line=dict(width=0)), selector=dict(mode="markers"))
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
footer()
