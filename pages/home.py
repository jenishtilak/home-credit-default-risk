"""Hub home: intro and links to each project and the deployment page."""

import streamlit as st

st.set_page_config(page_title="ML project hub", layout="wide")

st.title("ML project hub")
st.write(
    "Six Kaggle practice projects prepared for a Muthoot FinCorp Data Scientist "
    "interview. Each project page has a small live demo and a **How it was built** "
    "tab you can walk through out loud. The same GitHub repo also serves the Home "
    "Credit model as a FastAPI API on Render."
)

st.page_link(
    "pages/deployment.py",
    label="How this app is deployed — flow, live links, and JD vs free alternatives",
    icon="🚀",
)

st.subheader("Projects")

projects = [
    {
        "title": "1. Home Credit default risk",
        "blurb": "Application scorecard: predict whether a loan applicant will default. Tuned LightGBM, validation AUC 0.7804, SHAP reasons, approve / review / reject bands.",
        "page": "pages/home_credit.py",
        "status": "Live demo",
    },
    {
        "title": "2. Credit card fraud",
        "blurb": "Detect rare fraudulent transactions with XGBoost, compared with Isolation Forest. Class imbalance, precision / recall and SHAP.",
        "page": None,
        "status": "Demo being added next",
    },
    {
        "title": "3. CFPB complaints RAG",
        "blurb": "Retrieve similar consumer complaints and draft a short answer. This demo uses TF-IDF retrieval only; production would add an LLM such as Bedrock.",
        "page": None,
        "status": "Demo being added next",
    },
    {
        "title": "4. Bank term-deposit propensity",
        "blurb": "Who is likely to subscribe to a term deposit after a campaign call. Model comparison (logistic regression, trees, XGBoost) and a business cut-off.",
        "page": None,
        "status": "Demo being added next",
    },
    {
        "title": "5. Credit-card customer segmentation",
        "blurb": "K-Means segments on CC General spending behaviour, with silhouette to choose k and PCA only to plot the clusters.",
        "page": None,
        "status": "Demo being added next",
    },
    {
        "title": "6. Lending Club interest-rate regression",
        "blurb": "Predict the interest rate on a personal loan. XGBoost regressor, RMSE / MAE / R², and the drivers of a higher rate.",
        "page": None,
        "status": "Demo being added next",
    },
]

for i in range(0, len(projects), 2):
    cols = st.columns(2)
    for col, project in zip(cols, projects[i : i + 2]):
        with col:
            st.markdown(f"### {project['title']}")
            st.write(project["blurb"])
            if project["page"]:
                st.page_link(project["page"], label="Open this project", icon="➡️")
            else:
                st.caption(project["status"])

st.divider()
st.caption(
    "Repo: [jenishtilak/home-credit-default-risk](https://github.com/jenishtilak/home-credit-default-risk)  ·  "
    "API docs: [Render /docs](https://home-credit-default-risk-api-q0qa.onrender.com/docs)"
)
