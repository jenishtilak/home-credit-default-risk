"""Home Credit default-risk page — score an applicant and the 12 training steps."""

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from scoring import (
    APPROVE_MAX,
    EDUCATION_COLUMNS,
    FLAG_CUTOFF,
    FRIENDLY_NAMES,
    REVIEW_MAX,
    build_features,
    decide,
)

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


NOTEBOOK_GITHUB = (
    "https://github.com/jenishtilak/home-credit-default-risk/blob/main/"
    "notebooks/home_credit_default_risk.ipynb"
)
NOTEBOOK_KAGGLE = (
    "https://www.kaggle.com/code/jenishtilakt/"
    "loan-default-credit-risk-scorecard-lightgbm"
)

# Typical applicant = training medians. Age 43.1 years, employed 4.5 years.
TYPICAL = {
    "ext1": 0.51,
    "ext2": 0.57,
    "ext3": 0.54,
    "age": 43,
    "income": 147150,
    "credit": 513531,
    "annuity": 24903,
    "years_employed": 4.5,
    "gender": "Female",
    "education": "Secondary / secondary special",
}

LOW_RISK = {
    "ext1": 0.78,
    "ext2": 0.82,
    "ext3": 0.80,
    "age": 52,
    "income": 250000,
    "credit": 400000,
    "annuity": 20000,
    "years_employed": 15.0,
    "gender": "Female",
    "education": "Higher education",
}

HIGH_RISK = {
    "ext1": 0.16,
    "ext2": 0.18,
    "ext3": 0.15,
    "age": 28,
    "income": 110000,
    "credit": 650000,
    "annuity": 38000,
    "years_employed": 1.0,
    "gender": "Male",
    "education": "Secondary / secondary special",
}


@st.cache_resource
def load_bundle():
    return joblib.load(APP_DIR / "home_credit_lgbm.joblib")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


def shap_chart(explainer, X, features):
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    values = np.array(values).reshape(-1)

    reasons = pd.Series(values, index=features)
    top = reasons.reindex(reasons.abs().sort_values(ascending=False).head(8).index)
    labels = [FRIENDLY_NAMES.get(name, name) for name in top.index]
    colors = ["#c0392b" if v > 0 else "#1e8449" for v in top.values]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(labels[::-1], top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (positive = higher default risk)")
    ax.set_title("Top reasons for this applicant")
    fig.tight_layout()
    return fig


def load_low_risk():
    st.session_state.update(LOW_RISK)


def load_high_risk():
    st.session_state.update(HIGH_RISK)


def render_score_tab(model, features, medians, explainer):
    st.write(
        "Enter a few applicant details and click **Predict**. "
        "Every other feature is filled with the training-set median, "
        "and the derived features (average external score, loan term, income ratios) "
        "are computed the same way as in the training notebook."
    )

    st.sidebar.header("Applicant")
    st.sidebar.button("Load a low-risk example", on_click=load_low_risk, use_container_width=True)
    st.sidebar.button("Load a high-risk example", on_click=load_high_risk, use_container_width=True)
    st.sidebar.caption(
        "External scores are like bureau scores, scaled from 0 to 1. Higher is safer."
    )

    st.sidebar.slider("External score 1", 0.0, 1.0, step=0.01, key="ext1")
    st.sidebar.slider("External score 2", 0.0, 1.0, step=0.01, key="ext2")
    st.sidebar.slider("External score 3", 0.0, 1.0, step=0.01, key="ext3")
    st.sidebar.number_input("Age (years)", min_value=21, max_value=70, step=1, key="age")
    st.sidebar.number_input(
        "Annual income", min_value=10000, max_value=2_000_000, step=1000, key="income"
    )
    st.sidebar.number_input(
        "Credit amount", min_value=10000, max_value=3_000_000, step=1000, key="credit"
    )
    st.sidebar.number_input(
        "Loan annuity", min_value=1000, max_value=200_000, step=500, key="annuity"
    )
    st.sidebar.number_input(
        "Years employed", min_value=0.0, max_value=40.0, step=0.5, key="years_employed"
    )
    st.sidebar.radio("Gender", ["Female", "Male"], key="gender")
    st.sidebar.selectbox("Education", list(EDUCATION_COLUMNS.keys()), key="education")
    clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)

    if clicked:
        X = build_features(
            st.session_state.ext1,
            st.session_state.ext2,
            st.session_state.ext3,
            st.session_state.age,
            st.session_state.income,
            st.session_state.credit,
            st.session_state.annuity,
            st.session_state.years_employed,
            st.session_state.gender,
            st.session_state.education,
            features,
            medians,
        )
        # Feature order must match the saved model.
        assert list(X.columns) == list(features)

        prob = float(model.predict_proba(X)[0, 1])
        band, message, color = decide(prob)

        left, mid, right = st.columns(3)
        left.metric("Default probability", f"{prob:.1%}")
        mid.markdown(
            f"<div style='padding:8px 12px;border-radius:8px;background:{color};"
            f"color:white;font-size:1.4rem;font-weight:600;text-align:center'>{band}</div>",
            unsafe_allow_html=True,
        )
        right.write(message)

        if prob >= FLAG_CUTOFF:
            st.warning(
                "This score is at or above the 0.5912 cut-off, so it is in the riskiest 20% of applicants."
            )
        else:
            st.info(
                "This score is below the 0.5912 cut-off that flags the riskiest 20% of applicants."
            )

        st.caption(
            f"Bands: Approve if under {APPROVE_MAX} (bottom 60% of validation scores), "
            f"Review if under {REVIEW_MAX} (next 25%), Reject at or above {REVIEW_MAX} (top 15%)."
        )

        fig = shap_chart(explainer, X, features)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption("Red bars push the default risk up. Green bars pull it down.")
    else:
        st.info("Fill in the applicant details in the sidebar, or load an example, then click Predict.")

    with st.expander("About the model"):
        st.markdown(
            """
- **Model:** LightGBM (tuned), trained on **307,511** Home Credit applications.
- **Validation:** AUC **0.7804**, Gini **0.5608**, KS **0.4252**.
- **Cut-off 0.5912:** flags the riskiest 20% of applicants. Precision 22.5%, recall 55.7%.
- **Decision bands** (illustrative 60% / 25% / 15% split of the validation scores):
  - Approve if score < **0.4132** — 3.0% defaulted in this band.
  - Review if **0.4132 ≤ score < 0.6460** — 9.9% defaulted.
  - Reject if score ≥ **0.6460** — 25.3% defaulted.
- The officer edits 10 fields. The remaining features (bureau history, previous
  applications, and so on) stay at the training medians. Goods price is set equal
  to the credit amount.
- SHAP explains *this* applicant: why the model moved the score up or down.
            """
        )


def render_build_tab():
    st.write(
        "This app scores new applicants with a tuned LightGBM model trained on the "
        "Kaggle Home Credit Default Risk data. The 12 steps below follow the training "
        "notebook, from the business problem to monitoring. Open each step as you explain it."
    )
    st.markdown(
        f"- Training notebook on [GitHub]({NOTEBOOK_GITHUB}) (with outputs)  \n"
        f"- Same notebook on [Kaggle]({NOTEBOOK_KAGGLE})"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Validation AUC", "0.7804")
    m2.metric("Gini", "0.5608")
    m3.metric("KS", "0.4252")
    m4.metric("Default rate", "8.07%")

    st.subheader("Model comparison")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": [
                    "Logistic regression (baseline)",
                    "LightGBM (default)",
                    "Tuned LightGBM (final)",
                ],
                "AUC": ["0.7600", "0.7784", "0.7804"],
                "Gini": ["0.5200", "0.5569", "0.5608"],
                "KS": ["0.3914", "0.4226", "0.4252"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("All metrics are on the same 20% validation set (61,503 applicants).")

    st.subheader("The 12 training steps")

    with st.expander("Step 1 — Problem statement and the target"):
        st.markdown(
            """
- Framed this as supervised binary classification: predict the probability that an
  applicant defaults, so the credit team can approve, review or reject at sanction.
- `TARGET = 1` means the client had payment difficulties; `0` means they repaid.
- Loaded **307,511** applications: **282,686** repaid and **24,825** defaulted
  (**8.07%**). That is about **11.4** repaid for every default, so we judge the
  model on AUC, Gini and KS, not accuracy.
            """
        )

    with st.expander("Step 2 — Data collection and the join"):
        st.markdown(
            """
- Mapped seven data tables. History tables are one-to-many (on average 5.6 bureau
  records and 4.9 previous applications per applicant).
- Summarised bureau, previous applications and instalments with `groupby` /
  aggregation, then LEFT JOINed on `SK_ID_CURR`. Rows stayed 307,511; columns
  went from **122 to 134**.
- 14.31% of applicants have no bureau history and default at **10.12%**, versus
  **7.73%** for those with history. That "thin file" signal is worth keeping.
            """
        )

    with st.expander("Step 3 — Exploratory data analysis"):
        st.markdown(
            """
- **3.2 Welch t-test** on EXT_SOURCE 1, 2 and 3, AMT_CREDIT and age. Defaulters
  have lower external scores (EXT_SOURCE_3 **0.391 vs 0.521**) and are younger
  (**40.8 vs 44.2** years). All five p-values were below 0.05.
- **3.3 Chi-square** on contract type, gender, education and income type. Cash
  loans default at 8.35% vs 5.48% for revolving; lower secondary is riskiest at
  10.93%, higher education safer at 5.36%.
- **3.5 Correlation:** strongest signals are the external scores (EXT_SOURCE_3
  **−0.1789**). Joined features such as late-payment rate also made the top 10.
- The tests confirmed which features carry signal and guided feature engineering.
  We did **not** drop columns based on p-values: with 300k rows almost everything
  is significant, so we relied on IV, model importance and validation AUC instead.
            """
        )

    with st.expander("Step 4 — Data cleaning"):
        st.markdown(
            """
- `DAYS_EMPLOYED = 365243` (about 1,000 years) is a placeholder, mostly pensioners.
  We turned it into missing and added a flag, `DAYS_EMPLOYED_ANOM`.
- Zero-filled missing history counts, added a `NO_BUREAU_HISTORY` flag, dropped
  **17** columns that were more than 60% missing, and filled the rest with the
  median (or `"Missing"` for categories).
- Final shape: **(307,511, 119)** with no missing values, ready for logistic regression.
            """
        )

    with st.expander("Step 5 — Feature engineering and WoE / IV"):
        st.markdown(
            """
- Built **7** business features: age and years employed, credit-to-income,
  annuity-to-income (like FOIR), loan term, average external score, and
  bureau debt-to-credit.
- Ranked 8 numeric features with Weight of Evidence and Information Value.
  Top IV: **EXT_SOURCE_MEAN 0.618**, CREDIT_TERM 0.144, YEARS_EMPLOYED 0.107.
- Nothing was dropped. IV above 0.5 is usually a leakage warning, but this is
  an external credit score known at application time, so it is strong rather
  than leaking.
            """
        )

    with st.expander("Step 6 — Encoding and the train / validation split"):
        st.markdown(
            """
- One-hot encoded the categorical columns with `get_dummies`, producing **250**
  features.
- Split **80/20**, stratified on TARGET: 246,008 train and 61,503 validation.
  The default rate stayed **8.07%** in both sets, so the evaluation is honest.
            """
        )

    with st.expander("Step 7 — Baseline logistic regression"):
        st.markdown(
            """
- Scaled logistic regression, the explainable industry-standard scorecard:
  AUC **0.7600**, Gini **0.5200**, KS **0.3914**.
- VIF check: AMT_CREDIT **41.35** and AMT_GOODS_PRICE **38.98**, because the
  loan amount almost equals the goods price. Both were kept for this version;
  in production we would keep one or replace them with a ratio.
            """
        )

    with st.expander("Step 8 — LightGBM"):
        st.markdown(
            """
- Trained LightGBM with `scale_pos_weight` **11.39** (so each default counts
  about 11 times as much) and early stopping. The run chose **328** trees.
- Validation: AUC **0.7784**, Gini **0.5569**, KS **0.4226**. Gini improved by
  **+0.0369** over the logistic baseline.
            """
        )

    with st.expander("Step 9 — Hyperparameter tuning"):
        st.markdown(
            """
- Ran a small `RandomizedSearchCV` on a 20% sample of the training data
  (10 random combinations × 3-fold CV), then retrained on the full training set.
- Tuned LightGBM: AUC **0.7804**, Gini **0.5608**, KS **0.4252**, with 1,003
  trees. This became the final model (`tuned_lgbm`).
- The tuning gain is small (0.7784 to 0.7804). Simpler trees (15 leaves, depth 5)
  did slightly better; more feature work would likely add more than more tuning.
            """
        )

    with st.expander("Step 10 — Evaluation, cut-off, bands and SHAP"):
        st.markdown(
            """
- Cut-off **0.5912** flags the riskiest 20% of applicants: precision **22.47%**,
  recall **55.67%**. Confusion matrix: TP **2,764**, FN **2,201**, FP **9,537**,
  TN **47,001**.
- Decision bands (an illustrative 60% / 25% / 15% split of the validation scores):
  Approve if under **0.4132** (3.01% defaulted), Review if under **0.6460**
  (9.85%), Reject at or above **0.6460** (25.34%).
- SHAP explains each score. The top feature by gain is EXT_SOURCE_MEAN (**31.98%**).
            """
        )

    with st.expander("Step 11 — Error analysis"):
        st.markdown(
            """
- Done on the **validation set** at the 0.5912 cut-off, not on a separate test set.
- Missed defaulters (the 2,201 false negatives) looked safer on paper: higher
  EXT_SOURCE_MEAN (**0.505 vs 0.371**), older, longer employed, fewer past refusals.
  Their default is driven by things the application data does not show.
- Weakest large segment: pensioners, AUC **0.7481** (11,228 applicants). That
  group might need its own features, model or policy rule.
            """
        )

    with st.expander("Step 12 — MLflow, saving the model and PSI monitoring"):
        st.markdown(
            """
- Tracked experiments with MLflow using a **local SQLite** file (`mlflow.db`)
  inside the notebook session. That file is not in this GitHub repo; in production
  we would point the same code at a shared MLflow server.
- The logged run is `tuned_lgbm`. The app loads `home_credit_lgbm.joblib`
  (the model, feature list, medians and cut-off).
- Score PSI between train and validation is **0.0003** (stable; below 0.1 is fine).
            """
        )


def render():
    st.set_page_config(page_title="Loan Default Risk Checker", layout="wide")
    bundle = load_bundle()
    model = bundle["model"]
    features = bundle["features"]
    medians = bundle["medians"]
    explainer = load_explainer(model)

    for key, value in TYPICAL.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.title("Loan Default Risk Checker")

    score_tab, build_tab = st.tabs(["Score an applicant", "How the model was built"])
    with score_tab:
        render_score_tab(model, features, medians, explainer)
    with build_tab:
        render_build_tab()


render()
