# AI Revenue Recovery — Payment Retry Optimizer

**Razorpay AI Builder Internship 2026 — Track 3: AI Revenue Recovery**

## Problem

When a payment fails, most gateways either retry blindly (wasting bank/network
calls, annoying customers) or don't retry at all (losing recoverable revenue).
Different failure reasons have very different retry economics — a
`bank_server_down` failure is often transient and worth retrying soon, while
`expired_card` almost never succeeds on retry and should be escalated to
manual recovery (e.g. prompting the customer for a new card) instead.

## What this does

Given a failed transaction's attributes (amount, payment method, failure
reason, bank, time, customer history), the model:

1. Predicts the **probability that an immediate retry succeeds**
2. Scans all 24 hours to recommend the **best time to retry**
3. Outputs a clear **recommended action**: `retry_now`, `retry_later`, or
   `do_not_retry_escalate_to_manual_recovery`

This turns a blind retry policy into a targeted one — retrying only when it's
likely to work, at the time it's most likely to work, and routing dead-end
cases to manual recovery instead of wasting attempts.

## Approach

- **Data**: Synthetic dataset (`data/generate_data.py`) simulating 15,000
  failed transactions with realistic failure-reason distributions and a
  retry-outcome generator built from domain assumptions (e.g. transient
  failures like bank downtime/timeouts are more retry-friendly than
  fundamental failures like insufficient funds or risk flags). Designed as a
  drop-in replacement for real gateway data — same schema.
- **Model**: Gradient Boosting Classifier (scikit-learn) with one-hot encoded
  categorical features (payment method, failure reason, bank). ROC-AUC ~0.74
  on held-out test data.
- **Serving**: FastAPI app exposing a `/predict` endpoint.

## Project structure

```
ai-revenue-recovery/
├── data/
│   ├── generate_data.py       # synthetic dataset generator
│   └── failed_transactions.csv
├── models/
│   ├── train_model.py         # training script
│   └── retry_model.joblib     # trained model
├── analysis/
│   ├── explain_model.py       # feature importance + confusion matrix
│   ├── cost_benefit.py        # blind vs targeted retry ROI comparison
│   ├── feature_importance.png
│   └── confusion_matrix.png
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
├── app/
│   └── main.py                # FastAPI service
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python data/generate_data.py      # regenerate dataset (optional, already included)
python models/train_model.py      # retrain model (optional, already included)
python analysis/explain_model.py  # generates feature importance + confusion matrix charts
python analysis/cost_benefit.py   # prints ROI comparison: blind vs AI-targeted retry
uvicorn app.main:app --reload --port 8000
```

### Interactive dashboard

```bash
streamlit run dashboard/app.py
```

Opens a browser UI with three tabs:
- **Live Prediction** — enter transaction details, get an instant retry recommendation
- **Model Insights** — feature importance chart, confusion matrix, failure-reason breakdown
- **Business Impact** — adjustable retry-cost slider showing blind vs AI-targeted retry ROI

### API

Then open `http://localhost:8000/docs` for interactive API docs, or:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1499,
    "payment_method": "UPI",
    "failure_reason": "bank_server_down",
    "bank": "HDFC",
    "hour_of_day": 2,
    "day_of_week": 2,
    "prev_retry_attempts": 0,
    "customer_success_rate_history": 0.8
  }'
```

Example response:

```json
{
  "current_success_probability": 0.63,
  "best_retry_hour": 22,
  "best_retry_success_probability": 0.79,
  "recommended_action": "retry_now"
}
```

## Model explainability

`analysis/explain_model.py` extracts feature importances directly from the
trained Gradient Boosting model (after undoing one-hot encoding) and plots
the top 15 drivers of retry success, plus a confusion matrix showing where
the model gets predictions right/wrong. On this dataset, `failure_reason`
(especially transient reasons like `bank_server_down` / `network_timeout`)
and `prev_retry_attempts` are the strongest predictors — consistent with the
domain assumptions used to generate the data.

## Business impact analysis

`analysis/cost_benefit.py` compares two retry policies:
- **Blind retry**: retry every failed transaction
- **AI-targeted retry**: retry only transactions the model scores ≥ 0.5

It reports both **net ₹ gain** and **ROI (₹ recovered per ₹ spent)**, plus a
break-even sweep across different assumed retry costs. Key finding: the
targeted policy achieves a meaningfully higher ROI per attempt while skipping
the majority of low-probability failures — valuable when retries carry
hidden costs beyond a flat gateway fee (bank risk-flagging, customer
friction, support load). The script also searches for the retry-cost
threshold that maximizes net gain, so the policy can be tuned once real
unit economics are known.

## Future work

- Swap synthetic data for real (anonymized) transaction logs
- Add SHAP-based explainability per individual prediction for support/ops teams
- A/B test recommended retry timing against blind-retry baseline
- Deploy dashboard + API together (e.g. Render/Streamlit Cloud) for a live demo link
