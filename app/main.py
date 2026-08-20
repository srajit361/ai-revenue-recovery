"""
AI Revenue Recovery — Payment Retry Optimizer
Razorpay AI Builder Internship 2026 — Track 3

Given a failed transaction, predicts:
1. Probability of retry success (as-is)
2. The best hour of day to retry (to maximize success probability)
3. A recommended action (retry now / retry later / do not retry — escalate)
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "retry_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="AI Revenue Recovery — Payment Retry Optimizer",
    description="Predicts optimal retry strategy for failed payment transactions",
    version="1.0.0",
)


class Transaction(BaseModel):
    amount: float = Field(..., example=1499.00)
    payment_method: str = Field(..., example="UPI")
    failure_reason: str = Field(..., example="bank_server_down")
    bank: str = Field(..., example="HDFC")
    hour_of_day: int = Field(..., ge=0, le=23, example=14)
    day_of_week: int = Field(..., ge=0, le=6, example=2)
    prev_retry_attempts: int = Field(..., ge=0, example=0)
    customer_success_rate_history: float = Field(..., ge=0, le=1, example=0.8)


def build_row(txn: Transaction, hour_override: int = None) -> pd.DataFrame:
    hour = hour_override if hour_override is not None else txn.hour_of_day
    return pd.DataFrame([{
        "amount": txn.amount,
        "payment_method": txn.payment_method,
        "failure_reason": txn.failure_reason,
        "bank": txn.bank,
        "hour_of_day": hour,
        "day_of_week": txn.day_of_week,
        "is_weekend": 1 if txn.day_of_week >= 5 else 0,
        "prev_retry_attempts": txn.prev_retry_attempts,
        "customer_success_rate_history": txn.customer_success_rate_history,
    }])


@app.get("/")
def root():
    return {"status": "ok", "service": "AI Revenue Recovery - Retry Optimizer"}


@app.post("/predict")
def predict(txn: Transaction):
    row = build_row(txn)
    prob = float(model.predict_proba(row)[0, 1])

    # scan all 24 hours to find the best retry window
    best_hour, best_prob = txn.hour_of_day, prob
    for h in range(24):
        p = float(model.predict_proba(build_row(txn, hour_override=h))[0, 1])
        if p > best_prob:
            best_hour, best_prob = h, p

    if prob >= 0.6:
        action = "retry_now"
    elif best_prob >= 0.5:
        action = "retry_later"
    else:
        action = "do_not_retry_escalate_to_manual_recovery"

    return {
        "current_success_probability": round(prob, 3),
        "best_retry_hour": best_hour,
        "best_retry_success_probability": round(best_prob, 3),
        "recommended_action": action,
    }


# Run with: uvicorn app.main:app --reload --port 8000
