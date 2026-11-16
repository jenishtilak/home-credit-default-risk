"""Loan Default Risk Checker — Home Credit LightGBM model."""

from pathlib import Path

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

APP_DIR = Path(__file__).parent

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
                f"This score is at or above the 0.5912 cut-off, so it is in the riskiest 20% of applicants."
            )
        else:
            st.info(
                f"This score is below the 0.5912 cut-off that flags the riskiest 20% of applicants."
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


if __name__ == "__main__":
    render()
