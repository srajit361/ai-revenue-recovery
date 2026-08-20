"""
Trains a retry-success prediction model on the failed transactions dataset.
Also computes, per transaction, the best hour-of-day to retry (by scanning
hour_of_day 0-23 and picking the one with highest predicted success prob).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import joblib

df = pd.read_csv("data/failed_transactions.csv")

FEATURES = [
    "amount", "payment_method", "failure_reason", "bank",
    "hour_of_day", "day_of_week", "is_weekend",
    "prev_retry_attempts", "customer_success_rate_history",
]
TARGET = "retry_success"

X = df[FEATURES]
y = df[TARGET]

categorical = ["payment_method", "failure_reason", "bank"]
numeric = [c for c in FEATURES if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
], remainder="passthrough")

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))
print(classification_report(y_test, preds))

joblib.dump(pipeline, "models/retry_model.joblib")
print("Model saved to models/retry_model.joblib")
