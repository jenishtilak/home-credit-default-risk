"""Shared scoring helpers used by the Streamlit app and the FastAPI service."""

import numpy as np
import pandas as pd

# Decision bands from the notebook, measured on the validation scores:
#   Approve  = bottom 60%  (score < q60)
#   Review   = next 25%    (q60 <= score < q85)
#   Reject   = top 15%     (score >= q85)
# The 0.5912 cut-off is the 80th percentile: it flags the riskiest 20%.
APPROVE_MAX = 0.4132
REVIEW_MAX = 0.6460
FLAG_CUTOFF = 0.5912

EDUCATION_COLUMNS = {
    "Secondary / secondary special": "NAME_EDUCATION_TYPE_Secondary___secondary_special",
    "Higher education": "NAME_EDUCATION_TYPE_Higher_education",
    "Incomplete higher": "NAME_EDUCATION_TYPE_Incomplete_higher",
    "Lower secondary": "NAME_EDUCATION_TYPE_Lower_secondary",
    "Academic degree": "NAME_EDUCATION_TYPE_Academic_degree",
}

# Most common category in the training data, after one-hot encoding.
DEFAULT_DUMMIES = {
    "NAME_CONTRACT_TYPE_Cash_loans": 1,
    "CODE_GENDER_F": 1,
    "FLAG_OWN_CAR_N": 1,
    "FLAG_OWN_REALTY_Y": 1,
    "NAME_TYPE_SUITE_Unaccompanied": 1,
    "NAME_INCOME_TYPE_Working": 1,
    "NAME_EDUCATION_TYPE_Secondary___secondary_special": 1,
    "NAME_FAMILY_STATUS_Married": 1,
    "NAME_HOUSING_TYPE_House___apartment": 1,
    "OCCUPATION_TYPE_Missing": 1,
    "WEEKDAY_APPR_PROCESS_START_TUESDAY": 1,
    "ORGANIZATION_TYPE_Business_Entity_Type_3": 1,
    "HOUSETYPE_MODE_Missing": 1,
    "WALLSMATERIAL_MODE_Missing": 1,
    "EMERGENCYSTATE_MODE_No": 1,
}

FRIENDLY_NAMES = {
    "EXT_SOURCE_MEAN": "Average external score",
    "EXT_SOURCE_1": "External score 1",
    "EXT_SOURCE_2": "External score 2",
    "EXT_SOURCE_3": "External score 3",
    "CREDIT_TERM": "Loan term (credit / annuity)",
    "INST_LATE_RATE": "Late payment rate",
    "AMT_INCOME_TOTAL": "Income",
    "AMT_CREDIT": "Credit amount",
    "AMT_ANNUITY": "Annuity",
    "AMT_GOODS_PRICE": "Goods price",
    "AGE_YEARS": "Age",
    "YEARS_EMPLOYED": "Years employed",
    "DAYS_EMPLOYED": "Days employed",
    "DAYS_BIRTH": "Age (in days)",
    "CREDIT_INCOME_RATIO": "Credit / income",
    "ANNUITY_INCOME_RATIO": "Annuity / income",
    "DEBT_TO_CREDIT": "Bureau debt / credit",
    "BUREAU_TOTAL_CREDIT": "Bureau total credit",
    "BUREAU_TOTAL_DEBT": "Bureau total debt",
    "BUREAU_ACTIVE_COUNT": "Active bureau loans",
    "BUREAU_LOAN_COUNT": "Bureau loan count",
    "PREV_REFUSED_COUNT": "Previous refusals",
    "PREV_APPROVED_COUNT": "Previous approvals",
    "PREV_APP_COUNT": "Previous applications",
    "PREV_MEAN_AMT_APPLICATION": "Avg. previous application amount",
    "DAYS_LAST_PHONE_CHANGE": "Days since phone change",
    "CODE_GENDER_M": "Male",
    "CODE_GENDER_F": "Female",
    "NAME_EDUCATION_TYPE_Higher_education": "Higher education",
    "NAME_EDUCATION_TYPE_Secondary___secondary_special": "Secondary education",
    "NAME_EDUCATION_TYPE_Lower_secondary": "Lower secondary education",
}


def build_features(ext1, ext2, ext3, age, income, credit, annuity,
                   years_employed, gender, education, features, medians):
    """One model row: user inputs + training medians + the notebook's derived features."""
    row = {name: 0.0 for name in features}

    for col, val in medians.items():
        if col in row:
            row[col] = float(val)

    for col, val in DEFAULT_DUMMIES.items():
        row[col] = val

    row["EXT_SOURCE_1"] = ext1
    row["EXT_SOURCE_2"] = ext2
    row["EXT_SOURCE_3"] = ext3
    row["DAYS_BIRTH"] = -age * 365.25
    row["AGE_YEARS"] = age
    row["DAYS_EMPLOYED"] = -years_employed * 365.25
    row["YEARS_EMPLOYED"] = years_employed
    row["AMT_INCOME_TOTAL"] = income
    row["AMT_CREDIT"] = credit
    row["AMT_ANNUITY"] = annuity
    # Keep goods price in line with the loan amount just entered.
    row["AMT_GOODS_PRICE"] = credit

    row["CODE_GENDER_F"] = 1 if gender == "Female" else 0
    row["CODE_GENDER_M"] = 1 if gender == "Male" else 0
    row["CODE_GENDER_XNA"] = 0

    for col in EDUCATION_COLUMNS.values():
        row[col] = 0
    row[EDUCATION_COLUMNS[education]] = 1

    # Same derived features as the training notebook.
    row["CREDIT_INCOME_RATIO"] = credit / income if income else 0
    row["ANNUITY_INCOME_RATIO"] = annuity / income if income else 0
    row["CREDIT_TERM"] = credit / annuity if annuity else 0
    row["EXT_SOURCE_MEAN"] = (ext1 + ext2 + ext3) / 3
    bureau_credit = row["BUREAU_TOTAL_CREDIT"]
    row["DEBT_TO_CREDIT"] = row["BUREAU_TOTAL_DEBT"] / bureau_credit if bureau_credit > 0 else 0

    return pd.DataFrame([row], columns=features)


def decide(prob):
    """Return (band, short message, colour) from the validation-score bands."""
    if prob < APPROVE_MAX:
        return "Approve", "Low risk — recommend approve", "#1e8449"
    if prob < REVIEW_MAX:
        return "Review", "Medium risk — send to manual review", "#d68910"
    return "Reject", "High risk — recommend decline", "#c0392b"


def top_shap_reasons(explainer, X, features, n=3):
    """Top n SHAP reasons for this applicant, in plain words."""
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    values = np.array(values).reshape(-1)

    reasons = pd.Series(values, index=features)
    top = reasons.reindex(reasons.abs().sort_values(ascending=False).head(n).index)

    lines = []
    for name, value in top.items():
        label = FRIENDLY_NAMES.get(name, name)
        if value > 0:
            lines.append(f"{label} pushes the default risk up")
        else:
            lines.append(f"{label} pulls the default risk down")
    return lines
