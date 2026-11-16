# Home Credit Default Risk

A loan-default model for a Muthoot FinCorp Data Scientist interview. A credit officer enters a few applicant details; the service scores the risk of default, recommends approve / review / reject, and explains the top reasons with SHAP.

This repo serves the same model in two ways:

- A **FastAPI** prediction API, deployed on Render (free web service)
- A **Streamlit** app, deployed on Streamlit Community Cloud

Live URLs (fill in after deploy):

- API (Render): `https://home-credit-default-risk-api-q0qa.onrender.com`
- Streamlit app: `https://jenishtilak-home-credit-default-risk-app-jboaro.streamlit.app/`

## Project summary

**Business goal.** Help a credit officer decide whether to approve a Home Credit loan by estimating the probability that the applicant will default, then placing them in an approve, review, or reject band.

**Data.** 307,511 applicants from the Kaggle Home Credit Default Risk competition, with an 8.07% default rate (24,825 defaults and 282,686 repaid). The training table was joined with bureau, previous-application and instalment summaries.

**Model.** Tuned LightGBM, with `scale_pos_weight` set to 11.39 so each default counts about 11 times as much in training. Validation AUC is **0.7804**, versus **0.7600** for a logistic regression baseline. SHAP explains each score. The operating cut-off is **0.5912**, which flags the riskiest 20% of applicants (precision 22.5%, recall 55.7%). Decision bands, taken from the validation score distribution:

- **Approve** if score < 0.4132 (bottom 60%)
- **Review** if 0.4132 ≤ score < 0.6460 (next 25%)
- **Reject** if score ≥ 0.6460 (top 15%)

The officer edits 10 fields (three external scores, age, income, credit, annuity, years employed, gender, education). Every other feature is filled with the training-set median, and the same derived features as the notebook are computed (average external score, loan term, income ratios). The saved model file is `home_credit_lgbm.joblib` (1.8 MB).

The Streamlit app is a **project hub**. The home page links to each practice project and to **How this app is deployed**. Home Credit is live first: it still has the **Score an applicant** tab (demo with SHAP) and **How the model was built** (the 12 training steps, with the key numbers). The other five project demos are being added in the same app.

## Training notebook

The executed 12-step training notebook (with outputs) is in this repo:

- GitHub: https://github.com/jenishtilak/home-credit-default-risk/blob/main/notebooks/home_credit_default_risk.ipynb
- Kaggle: https://www.kaggle.com/code/jenishtilakt/loan-default-credit-risk-scorecard-lightgbm


## App structure (Streamlit hub)

- `app.py` — Streamlit entry point (`st.navigation`)
- `pages/home.py` — hub home, one card per project
- `pages/home_credit.py` — Home Credit scoring demo and 12-step walkthrough
- `pages/deployment.py` — end-to-end flow, request path, live links, and JD vs free alternatives
- `api.py`, `scoring.py`, `home_credit_lgbm.joblib`, `render.yaml` — the FastAPI service on Render (unchanged)

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api:app --reload
```

Open http://localhost:8000/docs and use **Try it out** on `POST /predict`. The request body already has typical-applicant defaults.

Start the Streamlit app:

```bash
streamlit run app.py
```

## Example request

Score a typical applicant (the API defaults):

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

Score a high-risk example:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ext1": 0.16,
    "ext2": 0.18,
    "ext3": 0.15,
    "age": 28,
    "income": 110000,
    "credit": 650000,
    "annuity": 38000,
    "years_employed": 1.0,
    "gender": "Male",
    "education": "Secondary / secondary special"
  }'
```

Example response shape:

```json
{
  "default_probability": 0.7932,
  "decision": "Reject",
  "top_reasons": [
    "Average external score pushes the default risk up",
    "Annuity pushes the default risk up",
    "External score 3 pushes the default risk up"
  ]
}
```

`GET /` returns a short health message.

## Deploy

**Render (FastAPI).** This folder includes a `render.yaml` blueprint: a free Python web service, Python 3.11, `pip install -r requirements.txt`, and `uvicorn api:app --host 0.0.0.0 --port $PORT`. Create a GitHub repo from this folder, then in Render choose **New > Blueprint** and point it at the repo. After it goes live, replace `https://home-credit-default-risk-api-q0qa.onrender.com` above with the service URL.

**Streamlit Community Cloud.** In share.streamlit.io, deploy the same repo, set the main file to `app.py`, and use Python 3.11. After it goes live, replace `https://jenishtilak-home-credit-default-risk-app-jboaro.streamlit.app/` above with the app URL.

## Production on AWS

The Render and Streamlit deploys are interview demos. A production path on AWS would look like this:

1. Package the API in a Docker image and push it to **Amazon ECR**.
2. Store `home_credit_lgbm.joblib` in **S3** (versioned), and load it at startup rather than baking a single file into the image.
3. Serve the model on **SageMaker** (managed endpoint) or run the FastAPI container on **ECS Fargate**, with **API Gateway** in front for auth, throttling and a stable URL.
4. Send logs and latency metrics to **CloudWatch**. Watch for feature drift with **PSI** (Population Stability Index) on the incoming applicant distribution versus the training data, and alert when it moves.
5. Optionally add a credit-policy assistant with **Bedrock** RAG: retrieve policy documents and explain a decline in plain language, without changing the model score itself.
