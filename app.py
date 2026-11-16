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
fraud = st.Page("pages/fraud.py", title="Credit card fraud", icon="🚨")
rag = st.Page("pages/rag.py", title="CFPB complaints RAG", icon="📄")
bank = st.Page("pages/bank.py", title="Bank term-deposit propensity", icon="🏦")
segmentation = st.Page(
    "pages/segmentation.py",
    title="Credit-card customer segmentation",
    icon="🧩",
)
lending = st.Page(
    "pages/lending_club.py",
    title="Lending Club interest rate",
    icon="📈",
)

pg = st.navigation(
    {
        "Hub": [home, deploy],
        "Projects": [home_credit, fraud, rag, bank, segmentation, lending],
    }
)
pg.run()
