"""Streamlit entry point: project hub navigation."""

import streamlit as st

home = st.Page("pages/home.py", title="All projects", icon="🏠", default=True)
deploy = st.Page(
    "pages/deployment.py",
    title="How this app is deployed",
    icon="🚀",
)
home_credit = st.Page(
    "pages/home_credit.py",
    title="Home Credit default risk",
    icon="💳",
)

pg = st.navigation(
    {
        "Hub": [home, deploy],
        "Projects": [home_credit],
    }
)
pg.run()
