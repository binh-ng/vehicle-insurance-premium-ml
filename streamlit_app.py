"""Entry point for Streamlit Community Cloud: `streamlit run streamlit_app.py`."""
import streamlit as st

st.set_page_config(page_title="Vehicle Insurance Premium ML", page_icon="🚗", layout="wide")

pages = [
    st.Page("app/home.py", title="Overview", icon="🏠", default=True),
    st.Page("app/explore.py", title="Explore the data", icon="📊"),
    st.Page("app/coverage.py", title="Coverage insight", icon="🛡️"),
    st.Page("app/models.py", title="Model comparison", icon="🏁"),
    st.Page("app/drivers.py", title="What drives premiums", icon="🔍"),
    st.Page("app/estimator.py", title="Premium estimator", icon="🧮"),
    st.Page("app/about.py", title="Changes & references", icon="📚"),
]
st.navigation(pages).run()
