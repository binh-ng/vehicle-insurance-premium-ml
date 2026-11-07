import streamlit as st

from app.common import footer, load_policies, load_result, team_line

st.title("Predicting Vehicle Insurance Premiums with Machine Learning")
st.markdown(
    f"{team_line()}  \n"
    "Master of Data Science and Analytics, University of Calgary · DATA 611 Group 6 · "
    "Showcased at **DataCon 2025**"
)

df = load_policies()
comp = load_result("model_comparison.csv")
best = comp.sort_values("R2_log", ascending=False).iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Policies analysed", f"{len(df):,}")
c2.metric("Vehicles", f"{df['object_id'].nunique():,}")
c3.metric("Models compared", f"{len(comp)}")
c4.metric(f"Median error, {best['model']}", f"{best['MdAPE']:.1%}",
          help="Median absolute percentage error on vehicles the model never saw")

st.subheader("The question")
st.markdown(
    """
How much should a vehicle's insurance cost, and which factors drive that price? Households want to know
whether a quote is fair. Fleet owners want to budget. Insurers want pricing that is consistent and explainable.

We used about 800,000 motor policies (2011 to 2018) from the Ethiopian Insurance Corporation, published
openly on Mendeley Data, to find the drivers of the premium and to build models that predict it.
"""
)

st.subheader("What we found")
st.markdown(
    """
- **Two kinds of policies hide in the data.** Almost half of the policies have an insured value of zero.
  They are liability (third-party) only covers, priced on a nationally fixed tariff by vehicle type and use,
  not missing values. Keeping the zeros instead of imputing them cuts the linear model's median error from
  52% to 29%.
- **For comprehensive cover, premium is roughly a rate times the insured value.** The median rate is about
  1.3% of the vehicle's value; usage, vehicle type and age move that rate up or down.
- **Tree ensembles win, and the top four are practically tied.** XGBoost, LightGBM, the tuned random forest and
  HistGradientBoosting all explain about 95% of the variation in log premium on unseen vehicles. A Gamma GLM,
  the actuarial standard, is less accurate but far easier to explain.
- **Honest evaluation matters.** The same vehicle appears in several years. Splitting by vehicle (not by row)
  and testing on a later time period gives scores you can trust.
"""
)

st.subheader("How to use this demo")
st.markdown(
    """
| Page | What you can do |
|---|---|
| 📊 Explore the data | Filter by year, coverage, vehicle type and usage and see how premiums behave |
| 🛡️ Coverage insight | See why zero insured value means liability-only cover |
| 🏁 Model comparison | Compare 13 models, the original DataCon results, and robustness checks |
| 🔍 What drives premiums | SHAP explanations of the best model |
| 🧮 Premium estimator | Describe a vehicle and get an estimated premium with a likely range |
| 📚 Changes & references | What we revised since the showcase, and every source we used |
"""
)
footer()
