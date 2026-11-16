"""Lending Club interest-rate regression — live prediction plus how it was built."""

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hub_lib import concept, notebook_links

KAGGLE = "https://www.kaggle.com/code/jenishtilakt/lending-club-interest-rate-regression-xgboost"

HOME = ["MORTGAGE", "RENT", "OWN", "OTHER", "NONE"]
VERIFY = ["Not Verified", "Source Verified", "Verified"]
PURPOSE = [
    "car", "credit_card", "debt_consolidation", "educational", "home_improvement",
    "house", "major_purchase", "medical", "moving", "other", "renewable_energy",
    "small_business", "vacation", "wedding",
]
APP_TYPE = ["INDIVIDUAL", "JOINT"]
NUM_COLS = [
    "loan_amnt", "term", "emp_length", "annual_inc", "dti", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
]


@st.cache_resource
def load_bundle():
    bundle = joblib.load(ROOT / "models" / "lending_club_int_rate.joblib")
    # Some XGBoost versions drop the trained intercept when loading this pickle
    # and fall back to 0.5, which makes every predicted rate about 13 pp too low.
    cfg = json.loads(bundle["model"].get_booster().save_config())
    current = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    if abs(current - 0.5) < 1e-6:
        bundle["rate_offset"] = float(bundle["mean_rate"]) - 0.5
    else:
        bundle["rate_offset"] = 0.0
    return bundle


@st.cache_resource
def load_examples():
    return json.loads((ROOT / "data" / "examples.json").read_text())


def set_example(name):
    ints = {
        "loan_amnt", "term", "emp_length", "annual_inc", "open_acc", "pub_rec",
        "revol_bal", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    }
    for k, v in load_examples()[name].items():
        st.session_state[f"lc_{k}"] = int(v) if k in ints else v


def build_features(bundle):
    X = pd.DataFrame(0.0, index=[0], columns=bundle["features"])
    for col in NUM_COLS:
        X.loc[0, col] = float(st.session_state[f"lc_{col}"])
    home = st.session_state["lc_home_ownership"]
    col = f"home_ownership_{home}"
    if col in X.columns:
        X.loc[0, col] = 1
    ver = st.session_state["lc_verification_status"]
    col = f"verification_status_{ver}"
    if col in X.columns:
        X.loc[0, col] = 1
    purpose = st.session_state["lc_purpose"]
    col = f"purpose_{purpose}"
    if col in X.columns:
        X.loc[0, col] = 1
    app = st.session_state["lc_application_type"]
    col = f"application_type_{app}"
    if col in X.columns:
        X.loc[0, col] = 1
    return X


def render_demo(bundle):
    st.write(
        "Predict the interest rate a Lending Club-style personal loan would have "
        "been charged, from application features. **Grade, sub-grade and instalment "
        "are not used** — they leak the price decision itself."
    )
    if "lc_loan_amnt" not in st.session_state:
        set_example("lc_typical")

    st.sidebar.header("Loan application")
    st.sidebar.button(
        "Load a typical 36-month loan",
        on_click=set_example,
        args=("lc_typical",),
        use_container_width=True,
    )
    st.sidebar.button(
        "Load a higher-rate example",
        on_click=set_example,
        args=("lc_high",),
        use_container_width=True,
    )
    st.sidebar.number_input("Loan amount", min_value=1000, max_value=40000, step=500, key="lc_loan_amnt")
    st.sidebar.selectbox("Term (months)", [36, 60], key="lc_term")
    st.sidebar.number_input("Employment length (years, 0–10)", min_value=0, max_value=10, step=1, key="lc_emp_length")
    st.sidebar.number_input("Annual income", min_value=5000, max_value=500000, step=1000, key="lc_annual_inc")
    st.sidebar.number_input("DTI", min_value=0.0, max_value=80.0, step=0.5, key="lc_dti")
    st.sidebar.number_input("Revolving utilisation (%)", min_value=0.0, max_value=150.0, step=1.0, key="lc_revol_util")
    st.sidebar.number_input("Revolving balance", min_value=0, max_value=500000, step=500, key="lc_revol_bal")
    st.sidebar.number_input("Open accounts", min_value=0, max_value=80, step=1, key="lc_open_acc")
    st.sidebar.number_input("Total accounts", min_value=0, max_value=120, step=1, key="lc_total_acc")
    st.sidebar.number_input("Mortgage accounts", min_value=0, max_value=40, step=1, key="lc_mort_acc")
    st.sidebar.number_input("Public records", min_value=0, max_value=20, step=1, key="lc_pub_rec")
    st.sidebar.number_input("Public-record bankruptcies", min_value=0, max_value=10, step=1, key="lc_pub_rec_bankruptcies")
    st.sidebar.selectbox("Home ownership", HOME, key="lc_home_ownership")
    st.sidebar.selectbox("Verification status", VERIFY, key="lc_verification_status")
    st.sidebar.selectbox("Purpose", PURPOSE, key="lc_purpose")
    st.sidebar.selectbox("Application type", APP_TYPE, key="lc_application_type")
    clicked = st.sidebar.button("Predict rate", type="primary", use_container_width=True)

    if not clicked:
        st.info("Load an example in the sidebar, or edit the fields, then predict.")
        return

    pred = float(bundle["model"].predict(build_features(bundle))[0]) + bundle["rate_offset"]
    mean_rate = float(bundle["mean_rate"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted interest rate", f"{pred:.2f}%")
    c2.metric("Training-set mean rate", f"{mean_rate:.2f}%")
    c3.metric("Vs mean", f"{pred - mean_rate:+.2f} pp")
    st.caption(
        "Term is the strongest driver (60-month loans are priced higher). "
        "This model is learning a historical rate card, not a 'correct' or "
        "regulatory price, and should not be used on its own to set a legally "
        "binding customer rate."
    )


def render_build():
    st.write(
        "Five models were compared on a 100,000-row sample of Lending Club loans. "
        "XGBoost won on validation RMSE and is the model in the live demo."
    )
    notebook_links("lending_club_interest_rate.ipynb", KAGGLE, "Kaggle")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test RMSE", "3.2711")
    m2.metric("Test MAE", "2.5929")
    m3.metric("Test R²", "0.4667")
    m4.metric("Mean rate", "13.65%")

    st.subheader("Model comparison (test set)")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": [
                    "Mean baseline",
                    "Linear regression / Ridge",
                    "Random Forest",
                    "XGBoost (saved model)",
                ],
                "MAE": ["3.5812", "2.8192", "2.6917", "2.5929"],
                "RMSE": ["4.4793", "3.5313", "3.3906", "3.2711"],
                "R²": ["0.0000", "0.3785", "0.4270", "0.4667"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    concept(
        "RMSE, MAE and R²",
        "MAE is the average absolute error in percentage points (2.59 pp). RMSE "
        "penalises large misses more (3.27 pp) and was the metric we used to pick "
        "the winner. R² is the share of rate variation the model explains (0.47) "
        "— a lot is left unexplained because we deliberately removed grade.",
    )
    concept(
        "Leakage (grade and instalment)",
        "Grade and sub-grade *are* the price decision, and instalment is computed "
        "from the rate. Including them would make the model look excellent and "
        "useless, because it would be reading the answer off the rate card.",
    )
    concept(
        "Why term dominates",
        "60-month loans had a mean rate of 17.13 vs 12.58 for 36-month loans. "
        "XGBoost puts 59% of its importance on term. That is a pricing policy "
        "pattern, not a mysterious effect.",
    )

    st.subheader("The 15 training steps")
    with st.expander("Steps 1–2 — Sample, parse and fill"):
        st.markdown(
            """
- 396,030 loans in the full file, then a **100,000-row** sample. Sample mean
  rate **13.67**, median 13.33, min 5.32, max 30.99.
- Parsed percent strings, term (`36 months` → 36) and employment length, then
  median-filled employment length, utilisation, mortgage accounts and bankruptcies.
            """
        )
    with st.expander("Step 3 — EDA and ANOVA on grade"):
        st.markdown(
            """
- Mean rate rises from grade A **7.39** to grade G **25.85** (one-way ANOVA
  F = 169,114.5, p ≈ 0). Strongest correlations with the rate: term 0.433,
  revol_util 0.294, dti 0.176, loan_amnt 0.173.
            """
        )
    with st.expander("Step 4 — Drop leakage columns"):
        st.markdown(
            """
- Dropped `grade`, `sub_grade` and `installment` so the model has to price from
  application features, not from the decision we are trying to study.
            """
        )
    with st.expander("Steps 5–6 — Encoding and the split"):
        st.markdown(
            """
- One-hot encoded home ownership, verification, purpose and application type
  (drop_first) to **33** features. 60 / 20 / 20 split; scale only for the linear models.
            """
        )
    with st.expander("Steps 7–12 — Models and the winner"):
        st.markdown(
            """
- Mean baseline RMSE **4.4793**. Linear / Ridge 3.5313. Random Forest 3.3906.
- XGBoost: MAE **2.5929**, RMSE **3.2711**, R² **0.4667**, **299** trees.
  Winner by validation RMSE (3.2645). Residual mean 0.0171.
            """
        )
    with st.expander("Steps 13–15 — Importance, MLflow and the saved model"):
        st.markdown(
            """
- Top XGBoost features: term **59.05%**, purpose_credit_card 4.67%,
  verification_status_Verified 4.30%, revol_util 3.45%, pub_rec 2.82%.
- All five models were logged in MLflow. The winner is `lending_club_int_rate.joblib`.
            """
        )


st.set_page_config(page_title="Lending Club interest rate", layout="wide")
bundle = load_bundle()
st.title("Lending Club interest rate")
demo, build = st.tabs(["Predict a rate", "How the model was built"])
with demo:
    render_demo(bundle)
with build:
    render_build()
