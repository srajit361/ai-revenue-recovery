"""
Streamlit dashboard — AI Revenue Recovery
Interactive demo: enter a failed transaction's details, see the model's
retry recommendation live, plus overall model insights (feature importance,
cost-benefit summary).

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="AI Revenue Recovery", page_icon="💳", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "retry_model.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "failed_transactions.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

st.title("💳 AI Revenue Recovery — Payment Retry Optimizer")

tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📊 Model Insights", "💰 Business Impact"])

# ---------------- TAB 1: Live Prediction ----------------
with tab1:
    st.subheader("Try a failed transaction")
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Amount (₹)", min_value=1.0, value=1499.0, step=10.0)
        payment_method = st.selectbox("Payment Method", ["UPI", "Credit Card", "Debit Card", "Netbanking", "Wallet"])
        failure_reason = st.selectbox("Failure Reason", [
            "bank_server_down", "network_timeout", "wrong_otp",
            "card_declined", "insufficient_funds", "risk_flagged", "expired_card",
        ])
        bank = st.selectbox("Bank", ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes Bank", "IDFC"])

    with col2:
        hour_of_day = st.slider("Hour of Day (failure occurred)", 0, 23, 14)
        day_of_week = st.selectbox("Day of Week", list(range(7)),
                                    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        prev_retry_attempts = st.number_input("Previous Retry Attempts", min_value=0, value=0, step=1)
        customer_success_rate_history = st.slider("Customer's Historical Success Rate", 0.0, 1.0, 0.8)

    if st.button("Predict Retry Outcome", type="primary"):
        row = pd.DataFrame([{
            "amount": amount, "payment_method": payment_method, "failure_reason": failure_reason,
            "bank": bank, "hour_of_day": hour_of_day, "day_of_week": day_of_week,
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "prev_retry_attempts": prev_retry_attempts,
            "customer_success_rate_history": customer_success_rate_history,
        }])
        prob = model.predict_proba(row)[0, 1]

        best_hour, best_prob = hour_of_day, prob
        for h in range(24):
            r = row.copy()
            r["hour_of_day"] = h
            p = model.predict_proba(r)[0, 1]
            if p > best_prob:
                best_hour, best_prob = h, p

        if prob >= 0.6:
            action, color = "RETRY NOW", "green"
        elif best_prob >= 0.5:
            action, color = "RETRY LATER", "orange"
        else:
            action, color = "ESCALATE TO MANUAL RECOVERY", "red"

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Success Probability", f"{prob:.1%}")
        m2.metric("Best Retry Hour", f"{best_hour}:00", f"{best_prob:.1%} success chance")
        m3.markdown(f"### Action: :{color}[{action}]")

# ---------------- TAB 2: Model Insights ----------------
with tab2:
    st.subheader("What drives retry success?")
    c1, c2 = st.columns(2)
    with c1:
        fi_path = os.path.join(os.path.dirname(__file__), "..", "analysis", "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Feature Importance")
        else:
            st.info("Run `python analysis/explain_model.py` first to generate this chart.")
    with c2:
        cm_path = os.path.join(os.path.dirname(__file__), "..", "analysis", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")
        else:
            st.info("Run `python analysis/explain_model.py` first to generate this chart.")

    st.subheader("Failure reason breakdown")
    reason_stats = df.groupby("failure_reason")["retry_success"].agg(["mean", "count"]).reset_index()
    reason_stats.columns = ["Failure Reason", "Success Rate", "Count"]
    reason_stats["Success Rate"] = (reason_stats["Success Rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(reason_stats.sort_values("Count", ascending=False), use_container_width=True)

# ---------------- TAB 3: Business Impact ----------------
with tab3:
    st.subheader("Blind Retry vs AI-Targeted Retry")

    RETRY_COST = st.slider("Assumed cost per retry attempt (₹)", 1, 100, 2)

    X = df[["amount", "payment_method", "failure_reason", "bank", "hour_of_day",
            "day_of_week", "is_weekend", "prev_retry_attempts", "customer_success_rate_history"]]
    df["predicted_prob"] = model.predict_proba(X)[:, 1]

    blind_cost = len(df) * RETRY_COST
    blind_recovered = df.loc[df["retry_success"] == 1, "amount"].sum()
    blind_net = blind_recovered - blind_cost
    blind_roi = blind_recovered / blind_cost

    targeted = df[df["predicted_prob"] >= 0.5]
    targeted_cost = len(targeted) * RETRY_COST
    targeted_recovered = targeted.loc[targeted["retry_success"] == 1, "amount"].sum()
    targeted_net = targeted_recovered - targeted_cost
    targeted_roi = targeted_recovered / targeted_cost if targeted_cost > 0 else 0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔵 Blind Retry (retry everything)")
        st.metric("Transactions Retried", f"{len(df):,}")
        st.metric("Net Gain", f"₹{blind_net:,.0f}")
        st.metric("ROI", f"{blind_roi:.1f}x")
    with c2:
        st.markdown("### 🟢 AI-Targeted Retry")
        st.metric("Transactions Retried", f"{len(targeted):,}", f"{len(targeted)-len(df):,}")
        st.metric("Net Gain", f"₹{targeted_net:,.0f}")
        st.metric("ROI", f"{targeted_roi:.1f}x", f"{(targeted_roi/blind_roi - 1)*100:+.1f}%")

    st.info(
        "💡 At low retry costs, blind retry can win on raw net gain simply because "
        "attempts are cheap — but it wastes effort on near-zero-probability cases. "
        "The AI-targeted policy consistently recovers more ₹ per ₹ spent (higher ROI) "
        "and frees up capacity for alternative recovery channels (SMS/manual) on the "
        "cases it skips. Try raising the slider above to simulate higher-risk/cost retries."
    )
