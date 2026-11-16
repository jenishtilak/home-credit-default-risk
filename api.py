"""FastAPI service for the Home Credit default-risk model."""

from pathlib import Path
from typing import Literal

import joblib
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

from scoring import EDUCATION_COLUMNS, build_features, decide, top_shap_reasons

APP_DIR = Path(__file__).parent

# Load the model once at startup.
bundle = joblib.load(APP_DIR / "home_credit_lgbm.joblib")
model = bundle["model"]
features = bundle["features"]
medians = bundle["medians"]
explainer = shap.TreeExplainer(model)

app = FastAPI(
    title="Home Credit Default Risk API",
    description="Score a loan applicant and return a default probability, a decision band, and the top SHAP reasons.",
)

Education = Literal[
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]


class Applicant(BaseModel):
    ext1: float = Field(0.51, ge=0, le=1, description="External score 1 (0-1, higher is safer)")
    ext2: float = Field(0.57, ge=0, le=1, description="External score 2")
    ext3: float = Field(0.54, ge=0, le=1, description="External score 3")
    age: int = Field(43, ge=21, le=70, description="Age in years")
    income: float = Field(147150, gt=0, description="Annual income")
    credit: float = Field(513531, gt=0, description="Credit amount")
    annuity: float = Field(24903, gt=0, description="Loan annuity")
    years_employed: float = Field(4.5, ge=0, description="Years employed")
    gender: Literal["Female", "Male"] = Field("Female", description="Female or Male")
    education: Education = Field(
        "Secondary / secondary special",
        description="Education level",
    )


class Prediction(BaseModel):
    default_probability: float
    decision: str
    top_reasons: list[str]


@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "Home Credit default-risk API is running. POST /predict to score an applicant.",
    }


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    X = build_features(
        applicant.ext1,
        applicant.ext2,
        applicant.ext3,
        applicant.age,
        applicant.income,
        applicant.credit,
        applicant.annuity,
        applicant.years_employed,
        applicant.gender,
        applicant.education,
        features,
        medians,
    )
    # Feature order must match the saved model.
    assert list(X.columns) == list(features)

    prob = float(model.predict_proba(X)[0, 1])
    decision, _, _ = decide(prob)
    return {
        "default_probability": round(prob, 4),
        "decision": decision,
        "top_reasons": top_shap_reasons(explainer, X, features, n=3),
    }
