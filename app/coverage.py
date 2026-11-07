import plotly.express as px
import streamlit as st

from app.common import BLUE, COVERAGE_COLORS, CURRENCY, ORANGE, footer, load_policies

st.title("🛡️ Coverage insight: zero insured value is not missing data")
df = load_policies()

st.markdown(
    """
In the original project, about 231,000 zero insured values were treated as missing and filled with a median
value. Looking at them through an insurance lens tells a different story.

In Ethiopia, liability (third-party) insurance is mandatory, and its price is **nationally fixed for each car
type** (Terefe, 2023b). A liability-only policy does not insure the vehicle itself, so there is no insured value
to record. That matches what we see below: zero insured value policies cluster on a handful of exact prices.
"""
)

liab = df[df.coverage == "Liability only"]
top = liab["premium"].round(2).value_counts().head(12).reset_index()
top.columns = ["premium", "policies"]
top["share"] = top["policies"] / len(liab)
c1, c2 = st.columns([1.1, 1])
with c1:
    t = top.sort_values("policies")
    t["label"] = t["premium"].map(lambda v: f"{v:,.2f}")
    fig = px.bar(t, x="policies", y="label", orientation="h",
                 title="Most common exact premiums, liability-only policies",
                 labels={"policies": "Policies", "label": f"Premium ({CURRENCY})"},
                 custom_data=["share"])
    fig.update_traces(marker_color=ORANGE, hovertemplate="%{y}: %{x:,} policies (%{customdata[0]:.1%})<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.metric("Liability-only share of policies", f"{len(liab) / len(df):.1%}")
    st.metric("Share of liability-only policies on the 12 most common prices", f"{top['share'].sum():.0%}")
    comp = df[df.coverage == "Comprehensive"]
    st.metric("Share of comprehensive policies on their 12 most common prices",
              f"{comp['premium'].round(2).value_counts().head(12).sum() / len(comp):.0%}")
    st.caption("Repeated exact prices are the fingerprint of a tariff.")

st.subheader("For comprehensive cover, look at the rate, not the raw premium")
st.markdown(
    "Dividing the premium by the insured value gives the **premium rate**. It removes the size of the vehicle's "
    "value and shows the risk loading that usage, vehicle type and age add."
)
comp = df[(df.coverage == "Comprehensive") & df.rate_pct.between(0.05, 10)]
by = st.radio("Break the rate down by", ["usage", "vehicle_type", "insurance_type", "sex"], horizontal=True,
              format_func=lambda s: s.replace("_", " ").capitalize())
g = comp.groupby(by, observed=True)["rate_pct"].agg(["median", "count"]).reset_index()
g = g[g["count"] >= 200].sort_values("median")
fig = px.bar(g, x="median", y=by, orientation="h", custom_data=["count"],
             title=f"Median premium rate by {by.replace('_', ' ')} (comprehensive policies)",
             labels={"median": "Median premium rate (% of insured value)", by: ""})
fig.update_traces(marker_color=BLUE, hovertemplate="%{y}: %{x:.2f}% (n=%{customdata[0]:,})<extra></extra>")
fig.update_layout(height=max(320, 32 * len(g)))
st.plotly_chart(fig, use_container_width=True)

age = comp.assign(age_band=(comp.vehicle_age // 3 * 3).clip(upper=30)).groupby("age_band")["rate_pct"].median().reset_index()
fig = px.line(age, x="age_band", y="rate_pct", markers=True, title="Median premium rate by vehicle age",
              labels={"age_band": "Vehicle age (years, 3-year bands)", "rate_pct": "Median rate (% of insured value)"})
fig.update_traces(line_color=BLUE, line_width=2, marker_size=8)
st.plotly_chart(fig, use_container_width=True)
st.caption("The dataset author notes that an age loading applies after a vehicle's first three years (Terefe, 2023b).")

st.info(
    "**Advisor's note on the original consumer tip.** \"Choose a lower insured value\" does lower the premium, "
    "but insuring a vehicle below its market value usually means a smaller payout after a total loss or a "
    "proportional cut on partial claims. Insure at a realistic value, then compare deductibles and usage class."
)
footer()
