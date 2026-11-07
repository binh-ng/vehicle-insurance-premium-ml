import streamlit as st

from app.common import ROOT, TEAM, footer

st.title("📚 Changes & references")

st.subheader("Team")
st.markdown("\n".join(
    f"- **{full} ({pref})**: {role}" for (full, pref), role in zip(TEAM, [
        "Data cleaning, multiple linear regression, assumption testing and regularization; expanded pipeline, GitHub showcase and Streamlit demo",
        "Random forest modelling, hyperparameter tuning and feature importance",
        "Exploratory data analysis and interactive Plotly visualisations",
    ])
))
st.caption("Master of Data Science and Analytics, University of Calgary. DATA 611 Predictive Analytics, Winter 2025.")

st.subheader("What changed since DataCon 2025")
st.markdown((ROOT / "CHANGES.md").read_text().split("\n", 2)[2])

st.subheader("References")
st.markdown((ROOT / "REFERENCES.md").read_text().split("\n", 2)[2])
footer()
