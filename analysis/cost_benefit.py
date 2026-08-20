"""
Cost-benefit analysis: compares "retry everything" (blind policy) vs
"retry only what the model recommends" (targeted policy).

This is the number that actually matters to a business like Razorpay —
not accuracy or AUC, but rupees saved/recovered.

Assumptions (documented so they're easy to challenge/adjust):
- Every retry attempt costs a fixed amount (bank gateway fee / infra cost),
  regardless of outcome.
- A successful retry recovers the full transaction amount as revenue.
- These are illustrative constants — replace with real Razorpay unit
  economics if available.
"""

import pandas as pd
import joblib

RETRY_COST = 2.0          # ₹ cost per retry attempt (gateway/infra fee)
MODEL_PATH = "models/retry_model.joblib"
DATA_PATH = "data/failed_transactions.csv"

pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "amount", "payment_method", "failure_reason", "bank",
    "hour_of_day", "day_of_week", "is_weekend",
    "prev_retry_attempts", "customer_success_rate_history",
]

X = df[FEATURES]
df["predicted_prob"] = pipeline.predict_proba(X)[:, 1]

# ---------- Policy 1: Blind retry (retry every single failed transaction) ----------
blind_cost = len(df) * RETRY_COST
blind_recovered = df.loc[df["retry_success"] == 1, "amount"].sum()
blind_net = blind_recovered - blind_cost

# ---------- Policy 2: Targeted retry (only retry if model prob >= 0.5) ----------
THRESHOLD = 0.5
targeted = df[df["predicted_prob"] >= THRESHOLD]
targeted_cost = len(targeted) * RETRY_COST
targeted_recovered = targeted.loc[targeted["retry_success"] == 1, "amount"].sum()
targeted_net = targeted_recovered - targeted_cost

print("=" * 60)
print("COST-BENEFIT: Blind Retry vs AI-Targeted Retry")
print("=" * 60)
print(f"\nTotal failed transactions: {len(df):,}")
print(f"Retry cost per attempt: ₹{RETRY_COST}")

print(f"\n--- Policy 1: Blind Retry (retry everything) ---")
print(f"Transactions retried: {len(df):,}")
print(f"Total retry cost:     ₹{blind_cost:,.2f}")
print(f"Revenue recovered:    ₹{blind_recovered:,.2f}")
print(f"Net gain:             ₹{blind_net:,.2f}")

print(f"\n--- Policy 2: AI-Targeted Retry (retry only if prob >= {THRESHOLD}) ---")
print(f"Transactions retried: {len(targeted):,} ({len(targeted)/len(df):.1%} of failures)")
print(f"Total retry cost:     ₹{targeted_cost:,.2f}")
print(f"Revenue recovered:    ₹{targeted_recovered:,.2f}")
print(f"Net gain:             ₹{targeted_net:,.2f}")

blind_roi = blind_recovered / blind_cost
targeted_roi = targeted_recovered / targeted_cost

print(f"\n--- Efficiency (₹ recovered per ₹ spent on retries) ---")
print(f"Blind retry ROI:    {blind_roi:,.1f}x")
print(f"AI-targeted ROI:    {targeted_roi:,.1f}x  ({(targeted_roi/blind_roi - 1)*100:+.1f}% more efficient)")
print(f"Wasted retry attempts avoided: {len(df) - len(targeted):,} "
      f"({(len(df)-len(targeted))/len(df):.1%} of all failures)")

print(f"\n--- Why this matters ---")
print("At a flat ₹2/retry, blind retry wins on raw net gain simply because")
print("attempts are cheap. But real retry cost isn't just the gateway fee —")
print("excessive retries risk bank fraud-flagging, hurt customer trust, and")
print("consume support bandwidth. The AI-targeted policy achieves higher")
print(f"ROI per attempt ({targeted_roi:.0f}x vs {blind_roi:.0f}x) while skipping")
print(f"{len(df)-len(targeted):,} attempts that were very unlikely to succeed —")
print("that headroom can be spent on other recovery channels (SMS/manual)")
print("for the skipped, low-probability cases instead.")

print(f"\n--- Break-even: at what retry cost does a 0.5 threshold beat blind retry? ---")
for cost in [2, 10, 25, 50, 100]:
    b_net = blind_recovered - len(df) * cost
    t_net = targeted_recovered - len(targeted) * cost
    winner = "Targeted" if t_net > b_net else "Blind"
    print(f"  ₹{cost:>4}/retry -> Blind net: ₹{b_net:>14,.0f} | Targeted net: ₹{t_net:>14,.0f} | Better: {winner}")

# ---------- Threshold optimization: what threshold actually maximizes net gain? ----------
print(f"\n--- Optimal threshold search (at ₹{RETRY_COST}/retry) ---")
best_thresh, best_net = None, -float("inf")
results = []
for t in [round(x, 2) for x in [i / 20 for i in range(0, 21)]]:
    subset = df[df["predicted_prob"] >= t]
    if len(subset) == 0:
        continue
    cost = len(subset) * RETRY_COST
    recovered = subset.loc[subset["retry_success"] == 1, "amount"].sum()
    net = recovered - cost
    results.append((t, len(subset), net))
    if net > best_net:
        best_thresh, best_net = t, net

print(f"Net gain is maximized at threshold = {best_thresh} -> ₹{best_net:,.0f} "
      f"(retrying {[r[1] for r in results if r[0]==best_thresh][0]:,} of {len(df):,} failures)")
print("This confirms: at this dataset's success rate, the model adds most value")
print("by catching the small slice of near-zero-probability failures to skip —")
print("not by aggressively filtering. Adjust RETRY_COST above to see how the")
print("optimal threshold shifts as retries get more expensive/risky.")
