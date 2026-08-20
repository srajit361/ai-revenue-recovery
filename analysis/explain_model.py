"""
Model explainability: feature importance + confusion matrix.
Run after training the model. Saves two PNG charts to analysis/.

Why this matters: a plain accuracy number doesn't tell you WHY the model
makes a decision, or WHERE it goes wrong. Recruiters/interviewers care about
this because "AI" that can't explain itself is hard to trust in production.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

MODEL_PATH = "models/retry_model.joblib"
DATA_PATH = "data/failed_transactions.csv"

pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "amount", "payment_method", "failure_reason", "bank",
    "hour_of_day", "day_of_week", "is_weekend",
    "prev_retry_attempts", "customer_success_rate_history",
]
TARGET = "retry_success"

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 1. Feature Importance ----------
# Pull feature names out after one-hot encoding so the chart is readable
prep = pipeline.named_steps["prep"]
clf = pipeline.named_steps["clf"]

cat_features = prep.transformers_[0][1].get_feature_names_out(["payment_method", "failure_reason", "bank"])
numeric_features = ["amount", "hour_of_day", "day_of_week", "is_weekend",
                     "prev_retry_attempts", "customer_success_rate_history"]
all_feature_names = list(cat_features) + numeric_features

importances = clf.feature_importances_
imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(9, 6))
sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
plt.title("Top 15 Feature Importances — Retry Success Model")
plt.xlabel("Importance")
plt.ylabel("")
plt.tight_layout()
plt.savefig("analysis/feature_importance.png", dpi=150)
plt.close()
print("Saved analysis/feature_importance.png")
print("\nTop 5 drivers of retry success:")
print(imp_df.head(5).to_string(index=False))

# ---------- 2. Confusion Matrix ----------
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Fail", "Predicted Success"],
            yticklabels=["Actual Fail", "Actual Success"])
plt.title("Confusion Matrix — Retry Prediction")
plt.tight_layout()
plt.savefig("analysis/confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved analysis/confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (correctly predicted fail):  {tn}")
print(f"False Positives (predicted success, actually failed): {fp}")
print(f"False Negatives (predicted fail, actually succeeded): {fn}")
print(f"True Positives (correctly predicted success): {tp}")
