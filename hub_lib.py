"""Shared helpers for the Streamlit project pages."""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
GITHUB_BLOB = "https://github.com/jenishtilak/home-credit-default-risk/blob/main/notebooks"


def concept(title, text):
    st.info(f"**Concept in one line — {title}.** {text}")


def notebook_links(filename, kaggle_url, kaggle_label=None):
    github = f"{GITHUB_BLOB}/{filename}"
    label = kaggle_label or "Kaggle"
    st.markdown(
        f"- Training notebook on [GitHub]({github}) (with outputs)  \n"
        f"- Same notebook on [{label}]({kaggle_url})"
    )
