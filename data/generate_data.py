"""
Generates a synthetic dataset simulating failed payment transactions
for a payment gateway (Razorpay-style), used to train a retry-success
prediction model.

Why synthetic: real transaction data is sensitive/private. This mimics
realistic distributions of failure reasons, payment methods, amounts,
and retry outcomes so the model logic is genuinely useful on real data
later (drop-in replacement).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 15000

payment_methods = ["UPI", "Credit Card", "Debit Card", "Netbanking", "Wallet"]
method_probs = [0.45, 0.20, 0.20, 0.10, 0.05]

failure_reasons = [
    "insufficient_funds",
    "bank_server_down",
    "wrong_otp",
    "card_declined",
    "network_timeout",
    "risk_flagged",
    "expired_card",
]
failure_probs = [0.28, 0.18, 0.12, 0.15, 0.15, 0.07, 0.05]

banks = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "IDFC"]

rows = []
for i in range(N):
    method = np.random.choice(payment_methods, p=method_probs)
    reason = np.random.choice(failure_reasons, p=failure_probs)
    bank = np.random.choice(banks)
    amount = round(np.random.lognormal(mean=6.5, sigma=1.0), 2)  # skewed, realistic txn amounts
    hour_of_day = np.random.randint(0, 24)
    day_of_week = np.random.randint(0, 7)
    prev_attempts = np.random.poisson(0.7)
    customer_success_rate_history = np.clip(np.random.beta(5, 2), 0, 1)  # past reliability
    is_weekend = 1 if day_of_week >= 5 else 0

    # Retry outcome logic (ground truth generator, not visible to model)
    base = 0.35
    if reason == "bank_server_down":
        base += 0.25  # transient, retry works well
    if reason == "network_timeout":
        base += 0.20
    if reason == "insufficient_funds":
        base -= 0.15  # unlikely to succeed soon
    if reason == "risk_flagged":
        base -= 0.30
    if reason == "expired_card":
        base -= 0.35
    if method == "UPI":
        base += 0.08
    if prev_attempts >= 2:
        base -= 0.10 * prev_attempts
    base += 0.15 * customer_success_rate_history
    if 1 <= hour_of_day <= 5:  # bank maintenance windows
        base -= 0.10

    prob_success = np.clip(base + np.random.normal(0, 0.08), 0.02, 0.95)
    retry_success = np.random.binomial(1, prob_success)

    rows.append({
        "transaction_id": f"txn_{i:06d}",
        "amount": amount,
        "payment_method": method,
        "failure_reason": reason,
        "bank": bank,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "prev_retry_attempts": prev_attempts,
        "customer_success_rate_history": round(customer_success_rate_history, 3),
        "retry_success": retry_success,
    })

df = pd.DataFrame(rows)
df.to_csv("data/failed_transactions.csv", index=False)
print(f"Generated {len(df)} rows. Retry success rate: {df['retry_success'].mean():.2%}")
print(df.head())
