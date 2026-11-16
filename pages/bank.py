"""Bank term-deposit propensity — live score plus how the model was built."""

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

KAGGLE = "https://www.kaggle.com/code/jenishtilakt/bank-term-deposit-propensity-model-comparison"

JOBS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
    "self-employed", "services", "student", "technician", "unemployed", "unknown",
]
MARITAL = ["divorced", "married", "single", "unknown"]
EDU = [
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
]
YESNO = ["no", "yes", "unknown"]
CONTACT = ["cellular", "telephone"]
MONTHS = ["apr", "aug", "dec", "jul", "jun", "mar", "may", "nov", "oct", "sep"]
DAYS = ["fri", "mon", "thu", "tue", "wed"]
POUTCOME = ["failure", "nonexistent", "success"]
CAT_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUM_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "bank_marketing_propensity.joblib")


@st.cache_resource
def load_examples():
    return json.loads((ROOT / "data" / "examples.json").read_text())


def set_example(name):
    ints = {"age", "campaign", "pdays", "previous"}
    for k, v in load_examples()[name].items():
        st.session_state[f"bk_{k}"] = int(v) if k in ints else v


def build_features(bundle):
    row = {c: st.session_state[f"bk_{c}"] for c in CAT_COLS + NUM_COLS}
    frame = pd.DataFrame([row])
    frame = pd.get_dummies(frame, columns=CAT_COLS, drop_first=True, dtype=int)
    X = pd.DataFrame(0.0, index=[0], columns=bundle["features"])
    for col in frame.columns:
        if col in X.columns:
            X.loc[0, col] = frame.loc[0, col]
    return X


def render_demo(bundle):
    st.write(
        "Score a campaign contact: the probability they subscribe to a term deposit. "
        "`duration` (how long the call lasted) is **not** used — that would be leakage, "
        "because it is only known after the call."
    )
    examples = load_examples()
    if "bk_age" not in st.session_state:
        set_example("bank_typical")

    st.sidebar.header("Client")
    st.sidebar.button(
        "Load a typical client",
        on_click=set_example,
        args=("bank_typical",),
        use_container_width=True,
    )
    st.sidebar.button(
        "Load a likely subscriber",
        on_click=set_example,
        args=("bank_likely",),
        use_container_width=True,
    )
    st.sidebar.number_input("Age", min_value=18, max_value=100, step=1, key="bk_age")
    st.sidebar.selectbox("Job", JOBS, key="bk_job")
    st.sidebar.selectbox("Marital status", MARITAL, key="bk_marital")
    st.sidebar.selectbox("Education", EDU, key="bk_education")
    st.sidebar.selectbox("Credit in default", YESNO, key="bk_default")
    st.sidebar.selectbox("Housing loan", YESNO, key="bk_housing")
    st.sidebar.selectbox("Personal loan", YESNO, key="bk_loan")
    st.sidebar.selectbox("Contact", CONTACT, key="bk_contact")
    st.sidebar.selectbox("Month", MONTHS, key="bk_month")
    st.sidebar.selectbox("Day of week", DAYS, key="bk_day_of_week")
    st.sidebar.number_input("Contacts this campaign", min_value=1, max_value=50, step=1, key="bk_campaign")
    st.sidebar.number_input(
        "Days since last contact (999 = never)", min_value=0, max_value=999, step=1, key="bk_pdays"
    )
    st.sidebar.number_input("Previous contacts", min_value=0, max_value=20, step=1, key="bk_previous")
    st.sidebar.selectbox("Previous outcome", POUTCOME, key="bk_poutcome")
    st.sidebar.caption("Macro features stay at the loaded example (2008–2010 Portuguese campaign).")
    for col in ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]:
        if f"bk_{col}" not in st.session_state:
            st.session_state[f"bk_{col}"] = examples["bank_typical"][col]
    clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)

    if not clicked:
        st.info("Load an example in the sidebar, or edit the fields, then click Predict.")
        return

    X = build_features(bundle)
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    cutoff = float(bundle["threshold"])
    call = prob >= cutoff

    c1, c2, c3 = st.columns(3)
    c1.metric("Subscription score", f"{prob:.1%}")
    c2.metric("Cut-off (max F1)", f"{cutoff:.2f}")
    if call:
        c3.markdown(
            "<div style='padding:8px 12px;border-radius:8px;background:#1e8449;"
            "color:white;font-size:1.3rem;font-weight:600;text-align:center'>Call</div>",
            unsafe_allow_html=True,
        )
    else:
        c3.markdown(
            "<div style='padding:8px 12px;border-radius:8px;background:#7f8c8d;"
            "color:white;font-size:1.3rem;font-weight:600;text-align:center'>Do not call first</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "The 0.71 cut-off maximised F1 on the validation set. Scores are pushed up "
        "by scale_pos_weight, so they rank well but are not calibrated probabilities. "
        "Previous campaign success and the macro features (especially nr.employed) "
        "are the strongest drivers."
    )


def render_build():
    st.write(
        "Four models were compared on the UCI Bank Marketing data. XGBoost won on "
        "validation ROC-AUC and is the model in the live demo."
    )
    notebook_links("bank_term_deposit.ipynb", KAGGLE, "Kaggle")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", "0.8100")
    m2.metric("Test PR-AUC", "0.4717")
    m3.metric("Top-20% capture", "66.3%")
    m4.metric("Subscription rate", "11.27%")

    st.subheader("Model comparison (test set)")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": [
                    "Logistic regression (baseline)",
                    "Decision tree (max_depth 4)",
                    "Random Forest",
                    "XGBoost (saved model)",
                ],
                "ROC-AUC": ["0.7987", "0.7845", "0.8125", "0.8100"],
                "PR-AUC": ["0.4473", "0.4035", "0.4695", "0.4717"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "XGBoost was chosen for the slightly higher validation ROC-AUC (0.8147). "
        "On the test set, Random Forest is a very close 0.8125 vs 0.8100."
    )

    concept(
        "Leakage (duration)",
        "Call duration is only known after you have already called the client. "
        "Subscribers talked for 553 seconds vs 221 for the rest, so including it "
        "would inflate the score and make the model useless for choosing whom to call.",
    )
    concept(
        "Propensity and lift",
        "A propensity model ranks who is most likely to say yes. If we call the "
        "top 20% of scores we reach 66.3% of subscribers (lift 3.31 vs a random list).",
    )
    concept(
        "scale_pos_weight",
        "Each subscriber was weighted as 7.88 non-subscribers in XGBoost. That "
        "improves ranking on an 11% target, but it pushes raw scores up, so we "
        "chose a 0.71 cut-off rather than 0.50.",
    )

    st.subheader("The 14 training steps")
    with st.expander("Steps 1–2 — Problem and the target rate"):
        st.markdown(
            """
- Predict who will subscribe to a term deposit after a campaign call, so the
  call centre can spend capacity on the best leads.
- 41,176 clients after dropping 12 duplicates; **4,639** subscribers (**11.27%**).
            """
        )
    with st.expander("Step 3 — EDA (chi-square and a t-test on age)"):
        st.markdown(
            """
- Previous success: **65.1%** subscribe vs **8.8%** when there was no previous
  campaign. Cellular 14.7% vs telephone 5.2%.
- Age gap is significant but small (40.91 vs 39.91 years, Welch p = 1.843e-06).
            """
        )
    with st.expander("Step 4 — Drop duration (leakage)"):
        st.markdown(
            """
- Duration was shown only to illustrate the leak, then dropped. A targeting
  model can only use information known before the call starts.
            """
        )
    with st.expander("Steps 5–6 — Encoding and the split"):
        st.markdown(
            """
- One-hot encoded the categoricals (`get_dummies`, drop_first) to **52** features.
- Stratified train / validation / test, then scale only for logistic regression.
  XGBoost does not need scaling (`needs_scaling = False` in the saved bundle).
            """
        )
    with st.expander("Steps 7–10 — Four models"):
        st.markdown(
            """
- Logistic regression with class_weight, a depth-4 tree the business can read,
  Random Forest, and XGBoost with `scale_pos_weight` **7.88** and **116** trees.
            """
        )
    with st.expander("Steps 11–12 — Comparison, cut-off and lift"):
        st.markdown(
            """
- Cut-off **0.71** (validation-max F1): test precision **48.29%**, recall
  **54.85%**, F1 **51.36%**.
- Calling the top 20% reaches **66.3%** of subscribers (response rate 37.3%,
  lift 3.31).
            """
        )
    with st.expander("Steps 13–14 — SHAP, MLflow and the saved model"):
        st.markdown(
            """
- Top XGBoost features: nr.employed **39.36%**, cons.conf.idx 5.06%,
  emp.var.rate 4.45%, poutcome_success 3.26%, euribor3m 2.81%.
- All four models were logged in MLflow. The winner is `bank_marketing_propensity.joblib`.
            """
        )


st.set_page_config(page_title="Bank term-deposit propensity", layout="wide")
bundle = load_bundle()
st.title("Bank term-deposit propensity")
demo, build = st.tabs(["Score a client", "How the model was built"])
with demo:
    render_demo(bundle)
with build:
    render_build()
