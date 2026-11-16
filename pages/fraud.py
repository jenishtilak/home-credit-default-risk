"""Credit card fraud — live score plus how the model was built."""

import json
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hub_lib import concept, notebook_links

KAGGLE = "https://www.kaggle.com/code/jenishtilakt/credit-card-fraud-xgboost-isolation-forest"
KEY_FEATURES = ["Time", "Amount", "V14", "V10", "V4", "V12", "V17", "V8"]


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "creditcard_xgb.joblib")


@st.cache_resource
def load_examples():
    return json.loads((ROOT / "data" / "examples.json").read_text())


def set_example(name):
    examples = load_examples()
    st.session_state.update({f"fr_{k}": float(v) for k, v in examples[name].items()})


def current_row(features):
    examples = load_examples()
    base = dict(examples["fraud_genuine"])
    for name in features:
        key = f"fr_{name}"
        if key in st.session_state:
            base[name] = float(st.session_state[key])
        elif name not in base:
            base[name] = 0.0
    return base


def render_demo(bundle):
    st.write(
        "Score one credit-card transaction with the saved XGBoost model. "
        "V1 to V28 are anonymous PCA components from the bank, so they are not "
        "readable as merchant or location. Time and Amount are scaled with the "
        "training scaler before scoring, just as in the notebook."
    )
    st.sidebar.header("Transaction")
    st.sidebar.button(
        "Load a genuine example",
        on_click=set_example,
        args=("fraud_genuine",),
        use_container_width=True,
    )
    st.sidebar.button(
        "Load a fraud example",
        on_click=set_example,
        args=("fraud_fraud",),
        use_container_width=True,
    )

    features = bundle["features"]
    row = current_row(features)
    for name in features:
        if f"fr_{name}" not in st.session_state:
            st.session_state[f"fr_{name}"] = float(row[name])

    st.sidebar.number_input("Time (seconds from first transaction)", step=1.0, key="fr_Time")
    st.sidebar.number_input("Amount", min_value=0.0, step=1.0, key="fr_Amount")
    for name in ["V14", "V10", "V4", "V12", "V17", "V8"]:
        st.sidebar.number_input(name, step=0.1, format="%.4f", key=f"fr_{name}")
    st.sidebar.caption("The other PCA components stay at the loaded example's values.")
    clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)

    if not clicked:
        st.info("Load an example in the sidebar, or edit the fields, then click Predict.")
        return

    row = current_row(features)
    X = pd.DataFrame([row], columns=features)
    X[bundle["scale_cols"]] = bundle["scaler"].transform(X[bundle["scale_cols"]])
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    cutoff = float(bundle["threshold"])
    flagged = prob >= cutoff

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud score", f"{prob:.1%}")
    c2.metric("Cut-off", f"{cutoff:.2f}")
    if flagged:
        c3.markdown(
            "<div style='padding:8px 12px;border-radius:8px;background:#c0392b;"
            "color:white;font-size:1.3rem;font-weight:600;text-align:center'>Flag for review</div>",
            unsafe_allow_html=True,
        )
        st.warning("This score is at or above the 0.51 cut-off used in the notebook.")
    else:
        c3.markdown(
            "<div style='padding:8px 12px;border-radius:8px;background:#1e8449;"
            "color:white;font-size:1.3rem;font-weight:600;text-align:center'>Pass</div>",
            unsafe_allow_html=True,
        )
        st.info("This score is below the 0.51 cut-off used in the notebook.")

    dmatrix = xgb.DMatrix(X, feature_names=list(features))
    contrib = bundle["model"].get_booster().predict(dmatrix, pred_contribs=True)[0]
    values = np.array(contrib[:-1])  # last value is the bias term
    reasons = pd.Series(values, index=features)
    top = reasons.reindex(reasons.abs().sort_values(ascending=False).head(8).index)
    colors = ["#c0392b" if v > 0 else "#1e8449" for v in top.values]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(list(top.index[::-1]), top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (positive = higher fraud score)")
    ax.set_title("Top reasons for this transaction")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(
        "Red bars push the fraud score up. Green bars pull it down. "
        "V1–V28 cannot be translated into business language because they are PCA components."
    )


def render_build():
    st.write(
        "This page uses the XGBoost model trained in the Credit Card Fraud notebook. "
        "Open each step as you explain it."
    )
    notebook_links("credit_card_fraud.ipynb", KAGGLE, "Kaggle")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test PR-AUC", "0.7507")
    m2.metric("Test ROC-AUC", "0.9674")
    m3.metric("Recall at 0.51", "82.11%")
    m4.metric("Fraud rate", "0.1667%")

    st.subheader("Model comparison")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": [
                    "Logistic regression (baseline)",
                    "XGBoost (main model)",
                    "Isolation Forest (unsupervised)",
                ],
                "PR-AUC": ["0.6918", "0.7507", "0.0908"],
                "ROC-AUC": ["0.9645", "0.9674", "0.9429"],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    concept(
        "Class imbalance",
        "Fraud is only 0.17% of transactions, so a model that never flags fraud is "
        "still 99.8% accurate. We use PR-AUC, recall and precision@k, not accuracy, "
        "and we up-weight the rare class with scale_pos_weight.",
    )
    concept(
        "PCA components (V1–V28)",
        "PCA squeezes many original columns into new uncorrelated components. "
        "The bank already applied it to hide sensitive details, which is why we "
        "cannot say what V14 means even though it is the strongest fraud signal.",
    )
    concept(
        "scale_pos_weight vs SMOTE",
        "scale_pos_weight tells XGBoost each fraud is worth 598 genuine rows. "
        "That is a class-weight, not synthetic oversampling. SMOTE would invent "
        "new fraud rows; we did not use it here.",
    )

    st.subheader("The 10 training steps")
    with st.expander("Step 1 — Problem statement and the data"):
        st.markdown(
            """
- Supervised binary classification: predict the probability that a card transaction
  is fraud (`Class = 1`), so the fraud team can review the riskiest payments.
- 283,726 transactions after dropping 1,081 duplicates; **473** frauds, a fraud
  rate of **0.1667%**.
            """
        )
    with st.expander("Step 2 — Class imbalance"):
        st.markdown(
            """
- About **599** genuine payments for every fraud. Accuracy would look excellent
  for a model that never raises an alert, so we planned PR-AUC, precision, recall
  and precision@k from the start.
- The same genuine-to-fraud ratio is the starting value for `scale_pos_weight`.
            """
        )
    with st.expander("Step 3 — EDA (Amount, Time, Mann-Whitney)"):
        st.markdown(
            """
- Fraud Amount has a lower median (**9.82 vs 22.00**) but a higher mean
  (**123.87 vs 88.41**). Mann-Whitney p = **2.686e-05**.
- Strongest correlations with fraud: V17, V14, V12 and V10.
            """
        )
    with st.expander("Step 4 — Stratified split and scaling"):
        st.markdown(
            """
- 60 / 20 / 20 train / validation / test, stratified so the tiny fraud rate
  stays the same in each split.
- Only Time and Amount are scaled. V1 to V28 are already PCA features.
            """
        )
    with st.expander("Step 5 — Logistic regression baseline"):
        st.markdown(
            """
- Logistic regression with `class_weight='balanced'`: test PR-AUC **0.6918**,
  ROC-AUC **0.9645**. A strong, explainable floor to beat.
            """
        )
    with st.expander("Step 6 — XGBoost (main model)"):
        st.markdown(
            """
- `scale_pos_weight` **598.42**, early stopping on validation `aucpr`, **69** trees.
- Test PR-AUC **0.7507**, ROC-AUC **0.9674**. This is the model in the live demo.
            """
        )
    with st.expander("Step 7 — Isolation Forest (unsupervised comparison)"):
        st.markdown(
            """
- Useful when labels arrive late (chargebacks). Test PR-AUC **0.0908**, ROC-AUC
  **0.9429**. ROC looks fine because genuine cases are easy; PR-AUC shows it
  does not compete with the supervised model when labels exist.
            """
        )
    with st.expander("Step 8 — Evaluation and the 0.51 cut-off"):
        st.markdown(
            """
- Cut-off **0.51** (an illustrative cost assumption). Test precision **37.14%**,
  recall **82.11%**, F1 **51.15%**.
- Precision@100: **73%** (73 of the 95 test frauds sit in the top 100 scores).
            """
        )
    with st.expander("Step 9 — Feature importance and SHAP"):
        st.markdown(
            """
- Top features by gain: V14 **37.42%**, V10 7.40%, V4 4.67%, V12 3.69%, V8 3.58%.
- SHAP explains one transaction, but the reasons stay in PCA language.
            """
        )
    with st.expander("Step 10 — MLflow and the saved model"):
        st.markdown(
            """
- Experiments were logged in MLflow (local SQLite in the notebook). The app
  loads `creditcard_xgb.joblib`: the model, feature list, scaler and cut-off.
            """
        )


st.set_page_config(page_title="Credit card fraud", layout="wide")
bundle = load_bundle()
st.title("Credit card fraud")
demo, build = st.tabs(["Score a transaction", "How the model was built"])
with demo:
    render_demo(bundle)
with build:
    render_build()
