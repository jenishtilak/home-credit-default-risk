"""CFPB complaints RAG demo — TF-IDF retrieval plus how it was built."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hub_lib import concept, notebook_links

KAGGLE = "https://www.kaggle.com/code/jenishtilakt/consumer-complaint-rag-langchain-faiss"

EXAMPLES = {
    "Mortgage payment reported late": (
        "I sent my mortgage payment on time but the servicer says they never "
        "received it. Now they are reporting me as late to the credit bureaus "
        "and charging late fees. I have the bank confirmation of the payment."
    ),
    "Credit card fee I was not told about": (
        "My credit card company added an annual fee and an over-limit charge "
        "that were never explained when I opened the account. I asked them to "
        "remove the fees and they refused."
    ),
    "Credit report after identity theft": (
        "Someone opened accounts in my name. I filed an identity theft report "
        "and asked the credit bureau to correct my credit report, but the "
        "fraudulent accounts are still listed."
    ),
}


@st.cache_resource
def load_index():
    df = pd.read_csv(ROOT / "data" / "complaints_sample.csv")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(df["narrative"].astype(str))
    return df, vectorizer, matrix


def set_example(text):
    st.session_state["rag_query"] = text


def templated_answer(query, hits):
    top = hits.iloc[0]
    n_same = int((hits["product"] == top["product"]).sum())
    excerpt = str(top["narrative"]).replace("\n", " ").strip()
    if len(excerpt) > 320:
        excerpt = excerpt[:320].rsplit(" ", 1)[0] + "…"
    return (
        f"This looks most like a **{top['product']}** complaint about "
        f"**{top['issue']}**. Of the five closest past complaints, "
        f"**{n_same}** are also {top['product']}.\n\n"
        f"A typical extract from the closest match:\n\n> {excerpt}"
    )


def render_demo(df, vectorizer, matrix):
    st.write(
        "Paste a consumer complaint, or load an example. The demo finds the "
        "closest past complaints with **TF-IDF** (a word-overlap search) and "
        "fills a short template from those hits."
    )
    st.warning(
        "This is retrieval only — no large language model is called. "
        "In the training notebook, answers were generated with Flan-T5 after "
        "MiniLM + FAISS retrieval. In production, the same retrieval step would "
        "be followed by an LLM such as **Amazon Bedrock**, and a person would "
        "review the draft before it is sent."
    )

    if "rag_query" not in st.session_state:
        st.session_state["rag_query"] = EXAMPLES["Mortgage payment reported late"]

    st.caption("Try an example")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=set_example, args=(text,), use_container_width=True)

    query = st.text_area("Complaint or question", height=140, key="rag_query")
    clicked = st.button("Find similar complaints", type="primary")

    if not clicked:
        st.info("Enter a complaint, or pick an example, then search.")
        return
    if not query.strip():
        st.info("Please enter some text first.")
        return

    q = vectorizer.transform([query.strip()])
    scores = cosine_similarity(q, matrix).ravel()
    idx = np.argsort(scores)[::-1][:5]
    hits = df.iloc[idx].copy()
    hits["similarity"] = scores[idx]

    st.subheader("Templated extractive answer")
    st.markdown(templated_answer(query, hits))

    st.subheader("Top matching complaints")
    show = hits[["similarity", "product", "issue", "company", "narrative"]].reset_index(drop=True)
    show["similarity"] = show["similarity"].map(lambda x: f"{x:.3f}")
    show["narrative"] = show["narrative"].astype(str).str.slice(0, 280)
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption(
        "The search index is a 2,000-complaint sample from the public CFPB file "
        "(narratives only). Names and account numbers in the source are already "
        "redacted as XXXX."
    )


def render_build():
    st.write(
        "The training notebook is a full RAG pipeline: clean and chunk narratives, "
        "embed them with MiniLM, retrieve with FAISS, then generate a draft with "
        "Flan-T5. This app uses a lighter, free stand-in (TF-IDF retrieval + a "
        "template) so it can run on Streamlit Cloud without a paid LLM."
    )
    notebook_links("cfpb_complaints_rag.ipynb", KAGGLE, "Kaggle")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision@5", "69.5%")
    m2.metric("Hit rate", "93.2%")
    m3.metric("Index size", "7,600 narratives")
    m4.metric("Chunks", "22,274")

    concept(
        "RAG (retrieval-augmented generation)",
        "First find the most relevant documents, then ask a language model to "
        "answer using only those documents. Retrieval grounds the answer; the "
        "LLM writes it in plain language.",
    )
    concept(
        "Embeddings",
        "A model turns a paragraph into a list of numbers so that similar "
        "meanings sit close together. The notebook uses all-MiniLM-L6-v2 (384 "
        "numbers per chunk). This demo uses TF-IDF instead, which only looks at "
        "shared words.",
    )
    concept(
        "Retrieval",
        "Given a new complaint, score every stored chunk and keep the top k. "
        "The notebook uses FAISS over MiniLM vectors (precision@5 69.5%, vs a "
        "17.7% random baseline). This demo uses cosine similarity on TF-IDF.",
    )

    st.subheader("The 8 training steps")
    with st.expander("Step 1 — Problem, narratives and the sample"):
        st.markdown(
            """
- Goal: help an analyst draft a reply from similar past complaints, without
  changing any credit-risk score.
- The public file has 555,957 rows; **66,806** have a narrative (12.0%). We
  sampled **8,000**, with 7,600 in the index and 400 held out.
            """
        )
    with st.expander("Step 2 — EDA on product, issue and length"):
        st.markdown(
            """
- Most narratives are about a few products (mortgage, debt collection, credit
  reporting and credit cards). That mix matters, because a retriever can look
  strong just by always returning the majority product.
            """
        )
    with st.expander("Step 3 — Clean the text and chunk it"):
        st.markdown(
            """
- Chunked with LangChain's RecursiveCharacterTextSplitter into **22,274** chunks,
  so a long complaint can match on the relevant part.
            """
        )
    with st.expander("Step 4 — MiniLM embeddings and a FAISS index"):
        st.markdown(
            """
- all-MiniLM-L6-v2, 384 dimensions, 22,274 vectors in FAISS. This is the
  production-style retriever. The Streamlit demo does not load MiniLM or FAISS.
            """
        )
    with st.expander("Step 5 — Retrieval demo"):
        st.markdown(
            """
- Typical questions (a late mortgage payment, a surprise card fee, an identity-
  theft credit-report issue) were run through the index to show the top neighbours.
            """
        )
    with st.expander("Step 6 — Retrieval evaluation"):
        st.markdown(
            """
- Mean precision@5 **69.5%**, hit rate **93.2%**. Random baseline 17.7%;
  majority baseline 22.5% (Mortgage in the chunk mix). Product is only a proxy
  label for relevance.
            """
        )
    with st.expander("Step 7 — Generation with Flan-T5, and Bedrock in production"):
        st.markdown(
            """
- Flan-T5-base drafted short answers. They stayed on the right product but were
  only loosely grounded (one named an unconfirmed "$77.00 rewards balance").
- In production, **Amazon Bedrock** (or another approved model) would replace
  Flan-T5, with complaint-id logging and human review. This app does not call
  any generator.
            """
        )
    with st.expander("Step 8 — Optional product classifier"):
        st.markdown(
            """
- Embeddings plus logistic regression: accuracy **77.2%**, weighted F1 **78.5%**,
  versus a 26.2% majority baseline.
            """
        )


st.set_page_config(page_title="CFPB complaints RAG", layout="wide")
df, vectorizer, matrix = load_index()
st.title("CFPB complaints RAG")
demo, build = st.tabs(["Search similar complaints", "How it was built"])
with demo:
    render_demo(df, vectorizer, matrix)
with build:
    render_build()
