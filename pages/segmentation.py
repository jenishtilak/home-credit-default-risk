"""Credit-card customer segmentation — assign a cluster plus how it was built."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hub_lib import concept, notebook_links

KAGGLE = "https://www.kaggle.com/code/jenishtilakt/credit-card-customer-segmentation-kmeans-clusters"

ACTIONS = {
    0: "Low-activity customers — invite them to use the card with a simple spend offer, and do not push cash advances.",
    1: "Cash-advance users — high utilisation and almost never pay in full. Watch risk, price cash advances clearly, and offer a structured repayment plan.",
    2: "Revolvers — high spend and high balances on the highest limits. Good revenue, but set limits and collections rules carefully.",
    3: "Transactors — frequent purchases, the highest full-payment rate, little cash advance. Reward spend; they are the healthier users of the book.",
}

INPUTS = [
    ("BALANCE", "Balance", 0.0, 20000.0, 50.0),
    ("PURCHASES", "Purchases", 0.0, 30000.0, 50.0),
    ("ONEOFF_PURCHASES", "One-off purchases", 0.0, 20000.0, 50.0),
    ("INSTALLMENTS_PURCHASES", "Instalment purchases", 0.0, 15000.0, 50.0),
    ("CASH_ADVANCE", "Cash advance", 0.0, 20000.0, 50.0),
    ("CREDIT_LIMIT", "Credit limit", 50.0, 30000.0, 100.0),
    ("PAYMENTS", "Payments", 0.0, 30000.0, 50.0),
    ("MINIMUM_PAYMENTS", "Minimum payments", 0.0, 20000.0, 50.0),
    ("PURCHASES_FREQUENCY", "Purchases frequency (0–1)", 0.0, 1.0, 0.05),
    ("ONEOFF_PURCHASES_FREQUENCY", "One-off frequency (0–1)", 0.0, 1.0, 0.05),
    ("PURCHASES_INSTALLMENTS_FREQUENCY", "Instalment frequency (0–1)", 0.0, 1.0, 0.05),
    ("CASH_ADVANCE_FREQUENCY", "Cash-advance frequency (0–1)", 0.0, 1.0, 0.05),
    ("PRC_FULL_PAYMENT", "Full-payment rate (0–1)", 0.0, 1.0, 0.05),
    ("TENURE", "Tenure (months)", 6.0, 12.0, 1.0),
    ("BALANCE_FREQUENCY", "Balance frequency (0–1)", 0.0, 1.0, 0.05),
    ("CASH_ADVANCE_TRX", "Cash-advance transactions", 0.0, 80.0, 1.0),
    ("PURCHASES_TRX", "Purchase transactions", 0.0, 200.0, 1.0),
]


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "cc_general_kmeans.joblib")


@st.cache_resource
def load_examples():
    return json.loads((ROOT / "data" / "examples.json").read_text())


def set_cluster_example(label):
    row = load_examples()["cc_clusters"][str(label)]
    for k, v in row.items():
        st.session_state[f"cc_{k}"] = float(v)


def render_demo(bundle):
    st.write(
        "Assign a credit-card customer to one of four K-Means segments. "
        "Skewed amount and count columns are log-transformed (`log1p`) and then "
        "scaled, exactly as in the notebook. PCA is **not** used to assign the "
        "cluster — only to draw the 2D picture in the training notebook."
    )
    if "cc_BALANCE" not in st.session_state:
        set_cluster_example(3)

    st.sidebar.header("Customer")
    names = bundle["cluster_names"]
    for k in range(4):
        st.sidebar.button(
            f"Load a {names[k]} example",
            on_click=set_cluster_example,
            args=(k,),
            use_container_width=True,
        )
    for key, label, lo, hi, step in INPUTS:
        st.sidebar.number_input(label, min_value=lo, max_value=hi, step=step, key=f"cc_{key}")
    clicked = st.sidebar.button("Assign segment", type="primary", use_container_width=True)

    if not clicked:
        st.info("Load a segment example in the sidebar, or edit the fields, then assign.")
        return

    features = bundle["features"]
    row = {name: float(st.session_state[f"cc_{name}"]) for name in features}
    X = pd.DataFrame([row], columns=features)
    for col in bundle["skewed_cols"]:
        X[col] = np.log1p(X[col])
    Xs = bundle["scaler"].transform(X)
    label = int(bundle["model"].predict(Xs)[0])
    name = bundle["cluster_names"][label]

    st.metric("Assigned segment", f"{label} — {name}")
    st.write(ACTIONS[label])
    st.caption(
        "Cluster sizes in the training data: Low-activity 17.5%, Cash-advance users "
        "30.8%, Revolvers 28.8%, Transactors 22.8%. Silhouette at k = 4 is 0.217."
    )


def render_build():
    st.write(
        "K-Means on the CC General file (8,950 customers). The four business names "
        "come from the mean profile of each cluster, not from the algorithm itself."
    )
    notebook_links("cc_general_segmentation.ipynb", KAGGLE, "Kaggle")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("k", "4")
    m2.metric("Silhouette", "0.217")
    m3.metric("PCA variance (2D)", "55.8%")
    m4.metric("Customers", "8,950")

    concept(
        "K-Means",
        "Pick k centres and assign each customer to the nearest one, then move "
        "the centres and repeat. It finds compact groups in the scaled feature "
        "space; we have to choose k and then name the groups ourselves.",
    )
    concept(
        "Silhouette score",
        "How close a point is to its own cluster versus the next-nearest cluster "
        "(1 is perfect, 0 is on the border). Here k = 2 scores 0.251 but only "
        "splits the book in half; k = 4 scores 0.217 and gives four actionable "
        "profiles, so we kept 4. On card data, around 0.2 is common.",
    )
    concept(
        "PCA (for the picture only)",
        "PCA compresses the 17 scaled features into two axes so we can draw the "
        "clusters. The first two components explain 55.8% of the variance "
        "(33.9% + 22.0%). K-Means was fitted on all 17 features, not on these two.",
    )

    st.subheader("The 12 training steps")
    with st.expander("Steps 1–2 — Data and missing values"):
        st.markdown(
            """
- 8,950 customers, one row each, no duplicate IDs. 314 missing values were
  filled with the median: CREDIT_LIMIT (1 row, 3,000) and MINIMUM_PAYMENTS
  (313 rows, 312.34).
            """
        )
    with st.expander("Steps 3–5 — Skew, log1p and scaling"):
        st.markdown(
            """
- The most skewed columns are MINIMUM_PAYMENTS (13.85), ONEOFF_PURCHASES (10.05)
  and PURCHASES (8.14). PURCHASES and ONEOFF_PURCHASES are highly correlated (r = 0.917).
- Ten amount and count columns were log-transformed; frequency rates were left
  as they are. Then all 17 features were standardised.
            """
        )
    with st.expander("Step 6 — Choose k (elbow and silhouette)"):
        st.markdown(
            """
- The elbow flattens at k = 4. Silhouette is a little higher at k = 2 (0.251),
  but that only splits the book in half. Silhouette at k = 4 is **0.217**,
  inertia 86,547. We chose 4 for the business profiles.
            """
        )
    with st.expander("Steps 7–8 — Fit K-Means and name the clusters"):
        st.markdown(
            """
- Cluster sizes: 1,570, 2,760, 2,580 and 2,040.
- **0 Low-activity (17.5%):** low balance, low spend, low cash advance.
- **1 Cash-advance users (30.8%):** cash advance about 2,220, utilisation 61.9%, almost never pay in full.
- **2 Revolvers (28.8%):** high spend (2,604) and high balance (2,244) on the highest limits (6,393).
- **3 Transactors (22.8%):** frequent purchases, highest full-payment rate (29.2%), little cash advance.
            """
        )
    with st.expander("Step 9 — PCA 2D plot"):
        st.markdown(
            """
- Used only to draw the clusters. Two components explain **55.8%** of the
  variance. The plot is a sketch, not the space K-Means was trained in.
            """
        )
    with st.expander("Steps 10–12 — Actions, MLflow and the saved model"):
        st.markdown(
            """
- Each name has a suggested action (offers, limit policy or collections).
- MLflow logged k = 4 and silhouette 0.217. The scaler and model are in
  `cc_general_kmeans.joblib` (38.9 KB).
            """
        )


st.set_page_config(page_title="Credit-card customer segmentation", layout="wide")
bundle = load_bundle()
# cluster_names may have been saved with int keys; joblib keeps them as ints
if bundle["cluster_names"] and 0 not in bundle["cluster_names"]:
    bundle["cluster_names"] = {int(k): v for k, v in bundle["cluster_names"].items()}

st.title("Credit-card customer segmentation")
demo, build = st.tabs(["Assign a segment", "How the model was built"])
with demo:
    render_demo(bundle)
with build:
    render_build()
