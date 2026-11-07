import numpy as np
import plotly.express as px
import streamlit as st

from app.common import BLUE, COVERAGE_COLORS, CURRENCY, footer, load_policies

st.title("📊 Explore the data")
df = load_policies()

# Filters in one row above the charts
f1, f2, f3, f4 = st.columns([1.2, 1, 1.4, 1.4])
years = f1.slider("Policy start year", int(df.begin_year.min()), int(df.begin_year.max()),
                  (int(df.begin_year.min()), int(df.begin_year.max())))
cov = f2.multiselect("Coverage", list(COVERAGE_COLORS), default=list(COVERAGE_COLORS))
vt_all = df["vehicle_type"].value_counts().index.tolist()
vt = f3.multiselect("Vehicle type", vt_all, placeholder="All vehicle types")
us_all = df["usage"].value_counts().index.tolist()
us = f4.multiselect("Usage", us_all, placeholder="All usages")

m = df["begin_year"].between(*years) & df["coverage"].isin(cov)
if vt:
    m &= df["vehicle_type"].isin(vt)
if us:
    m &= df["usage"].isin(us)
d = df[m]
if d.empty:
    st.warning("No policies match these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Policies", f"{len(d):,}")
k2.metric("Median premium", f"{d.premium.median():,.0f} {CURRENCY}")
k3.metric("Share liability only", f"{(d.coverage == 'Liability only').mean():.0%}")
k4.metric("Policies with a claim paid", f"{d.claim_paid.fillna(0).gt(0).mean():.1%}")

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(d, x=np.log10(d["premium"]), color="coverage", nbins=80, barmode="overlay",
                       opacity=0.75, color_discrete_map=COVERAGE_COLORS,
                       title="Premium distribution (log scale)",
                       labels={"x": f"Premium ({CURRENCY}, log10)", "coverage": "Coverage"})
    fig.update_xaxes(tickvals=[1, 2, 3, 4, 5, 6], ticktext=["10", "100", "1k", "10k", "100k", "1M"])
    fig.update_yaxes(title="Policies")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Spikes in the liability-only curve are tariff prices that many vehicles share.")
with c2:
    trend = (d.groupby(["begin_year", "coverage"], observed=True)["premium"].median()
             .reset_index())
    fig = px.line(trend, x="begin_year", y="premium", color="coverage", markers=True,
                  color_discrete_map=COVERAGE_COLORS, title="Median premium by start year",
                  labels={"begin_year": "Policy start year", "premium": f"Median premium ({CURRENCY})",
                          "coverage": "Coverage"})
    fig.update_traces(line_width=2, marker_size=8)
    st.plotly_chart(fig, use_container_width=True)

order = d.groupby("vehicle_type", observed=True)["premium"].median().sort_values().index.tolist()
fig = px.box(d.sample(min(len(d), 60_000), random_state=0), y="vehicle_type", x="premium", color="coverage",
             category_orders={"vehicle_type": order}, log_x=True, points=False,
             color_discrete_map=COVERAGE_COLORS, title="Premium by vehicle type",
             labels={"premium": f"Premium ({CURRENCY}, log scale)", "vehicle_type": "", "coverage": "Coverage"})
fig.update_layout(height=520)
st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    comp = d[d.coverage == "Comprehensive"]
    if len(comp):
        s = comp.sample(min(len(comp), 8000), random_state=0)
        fig = px.scatter(s, x="insured_value", y="premium", log_x=True, log_y=True, opacity=0.35,
                         hover_data=["vehicle_type", "usage", "vehicle_age"],
                         title="Comprehensive: premium vs insured value",
                         labels={"insured_value": f"Insured value ({CURRENCY})", "premium": f"Premium ({CURRENCY})"})
        fig.update_traces(marker=dict(size=8, color=BLUE, line=dict(width=0)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Close to a straight line on log axes: premium is mostly a rate applied to the vehicle's value.")
with c4:
    cnt = d["usage"].value_counts().head(10).sort_values().reset_index()
    fig = px.bar(cnt, x="count", y="usage", orientation="h", title="Policies by usage (top 10)",
                 labels={"count": "Policies", "usage": ""})
    fig.update_traces(marker_color=BLUE, hovertemplate="%{y}: %{x:,}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Show a sample of the filtered data"):
    st.dataframe(d.drop(columns=["object_id"]).head(500), use_container_width=True)
footer()
