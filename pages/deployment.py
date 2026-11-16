"""How this app is deployed: flow, request path, and JD vs free alternatives."""

import streamlit as st

st.set_page_config(page_title="How this app is deployed", layout="wide")

GITHUB = "https://github.com/jenishtilak/home-credit-default-risk"
API = "https://home-credit-default-risk-api-q0qa.onrender.com"
API_DOCS = f"{API}/docs"
APP = "https://jenishtilak-home-credit-default-risk-app-jboaro.streamlit.app/"

st.title("How this app is deployed")
st.write(
    "The same tuned LightGBM model (validation AUC **0.7804**, logistic baseline "
    "**0.7600**) is versioned on GitHub, served as a FastAPI `POST /predict` "
    "endpoint on Render, and shown to a credit officer in this Streamlit app. "
    "The 1.8 MB `joblib` file holds the model, the feature list, the training "
    "medians and the cut-off. Experiments were tracked in MLflow in the notebook "
    "(local SQLite). A hosted DagsHub server was the planned free next step and "
    "is not live yet."
)

st.info(
    "Both free hosts sleep when idle. Open the API docs and this app about "
    "**10 minutes before 2:00 PM IST**. The first request can take about a minute."
)

st.markdown(
    f"- GitHub repo: [{GITHUB.replace('https://', '')}]({GITHUB})  \n"
    f"- Render API docs: [{API_DOCS.replace('https://', '')}]({API_DOCS})  \n"
    f"- This Streamlit app: [{APP.replace('https://', '')}]({APP})"
)

st.subheader("End-to-end flow")
st.write(
    "Walk an interviewer from the Kaggle data to a live decision. The notebook "
    "does the science; GitHub holds the code and the saved model; two hosts serve "
    "the same scorer — Render for machines, Streamlit Cloud for a person."
)

st.graphviz_chart(
    """
    digraph {
        rankdir=LR
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#d6eaf8", fillcolor="#eaf4fb"]
        edge [color="#5d6d7e"]

        data [label="1. Kaggle data\\n307,511 applications\\n8.07% default rate"]
        nb   [label="2. Notebook\\njoins, EDA, features\\nLightGBM, SHAP, bands"]
        mlf  [label="3. MLflow tracking\\nrun: tuned_lgbm\\nlocal sqlite mlflow.db"]
        job  [label="4. joblib model file\\n1.8 MB bundle\\nmodel, features, medians"]
        git  [label="5. GitHub repo\\napi.py, app.py, scoring.py\\nhome_credit_lgbm.joblib"]
        api  [label="6. FastAPI on Render\\nPOST /predict\\ninteractive docs at /docs"]
        ui   [label="7. Streamlit Cloud\\nUI + SHAP chart"]
        user [label="8. Credit officer / user\\nprobability, band, reasons"]

        data -> nb -> mlf -> job -> git
        git -> api -> user
        git -> ui -> user
    }
    """
)
st.caption(
    "Same scorer, two doors: the API is for other systems; the Streamlit app is "
    "for a person. Both load `home_credit_lgbm.joblib` and call the same helpers "
    "in `scoring.py`."
)

st.subheader("What happens on one request")
st.write(
    "One click of **Predict**, or one `POST /predict`. The officer types 10 fields. "
    "Everything else is filled from the training medians, and the derived features "
    "are rebuilt the same way as in the notebook."
)

st.graphviz_chart(
    """
    digraph {
        rankdir=LR
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#d5f5e3", fillcolor="#eafaf1"]
        edge [color="#5d6d7e"]

        a [label="A. INPUT\\n10 fields\\nJSON body or sidebar"]
        b [label="B. FEATURES\\nbuild_features\\nmedians + derived ratios"]
        c [label="C. SCORE\\npredict_proba\\nP(default)"]
        d [label="D. DECIDE\\n0.4132 approve\\n0.6460 reject"]
        e [label="E. EXPLAIN\\nSHAP reasons\\nAPI: top 3  ·  App: top 8"]
        f [label="F. OUTPUT\\nprobability,\\ndecision, reasons"]

        a -> b -> c -> d -> e -> f
    }
    """
)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
**A–B. Inputs and features**

- Ten fields: external scores 1–3 (0–1, higher is safer), age, years employed,
  income, credit, annuity, gender, and education.
- `GET /` is a health check and returns `status: ok`.
- `build_features` starts from training medians, writes the ten inputs, sets
  goods price equal to credit, and rebuilds `CREDIT_INCOME_RATIO`,
  `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `EXT_SOURCE_MEAN` and `DEBT_TO_CREDIT`.
        """
    )
with c2:
    st.markdown(
        """
**D–F. Bands, SHAP, result**

- **Approve** if score < **0.4132** (bottom 60%).
- **Review** if 0.4132 ≤ score < **0.6460** (next 25%).
- **Reject** if score ≥ **0.6460** (top 15%).
- A separate flag at **0.5912** marks the riskiest 20% (the business cut-off).
- SHAP: red bars push default risk up; green bars pull it down. The API returns
  the top 3 as plain sentences; the app plots the top 8.

Tested live: typical applicant **34.05% Approve**; high-risk example **79.32% Reject**.
        """
    )

st.subheader("JD / production (AWS) vs what we used now")
st.write(
    "The job description asks for deploying models using APIs, MLflow, Git and AWS. "
    "GenAI / RAG is a JD theme too, but it lives in the CFPB project, not in the "
    "scorecard. The line to remember: this is a free, bounded demo, and the "
    "production path on AWS is clear."
)

st.dataframe(
    {
        "Piece": [
            "Code and versioning",
            "Experiment tracking",
            "Model storage",
            "Packaging",
            "API serving",
            "Business UI",
            "Monitoring",
            "Security",
            "GenAI / RAG",
        ],
        "JD / production (AWS)": [
            "Git, hosted on GitHub (or CodeCommit).",
            "MLflow on a managed server, or SageMaker Experiments.",
            "Versioned object store in S3, plus a SageMaker Model Registry entry.",
            "A Docker image, pushed to Amazon ECR.",
            "SageMaker endpoint, or FastAPI on ECS Fargate, with API Gateway in front.",
            "An internal app on ECS or App Runner, behind the company login.",
            "CloudWatch for latency and errors, plus a scheduled PSI (drift) job.",
            "IAM roles, a private VPC, Secrets Manager, and no public PII.",
            "Amazon Bedrock as the generator, with retrieval over your own documents.",
        ],
        "What we used now (free alternative)": [
            "The same: Git and GitHub. One public repo holds the API, the app, scoring code and the model file.",
            "MLflow in the notebook, stored in a local SQLite file (mlflow.db), run tuned_lgbm. A free hosted DagsHub server is the planned next step and is not live yet.",
            "The 1.8 MB joblib file in the GitHub repo, loaded at startup by both the API and the app.",
            "requirements.txt plus Render’s native Python build (render.yaml, Python 3.11). Streamlit Cloud installs from the same file. No container in the demo.",
            "FastAPI on Render’s free web service. The test page is /docs. The free plan sleeps when idle; the first request takes about a minute.",
            "Streamlit Community Cloud, deploying this app from the same GitHub repo, with the SHAP chart.",
            "PSI was computed in the notebook: train vs validation scores 0.0003 (stable). Host logs on Render / Streamlit; no live drift job yet.",
            "A public demo with no real applicant PII — only made-up or typical values. Do not put live customer data on these hosts.",
            "Not in the Home Credit scorecard. The CFPB page in this hub uses TF-IDF retrieval only. In production, Bedrock would generate the answer.",
        ],
    },
    hide_index=True,
    use_container_width=True,
)

st.subheader("The 30-second answer")
st.markdown(
    """
> The model is versioned in GitHub, served as a REST API with FastAPI on Render,
> and as a Streamlit app for business users. Experiments were tracked in MLflow
> locally — a hosted DagsHub server is the next free step. In production at
> Muthoot I’d containerise it with Docker, store the model in S3, run it on
> SageMaker or ECS behind API Gateway, and monitor it with CloudWatch and PSI
> drift checks. RAG is a separate project: the CFPB assistant, with Bedrock
> replacing the generator in production.
    """
)

st.caption(
    "Close with: “Same joblib file, same scoring.py, two doors — the API for "
    "systems, the app for a credit officer.”"
)
