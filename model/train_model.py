import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import json

# Feature extraction function with safety check
def extract_features(url):
    if not isinstance(url, str):
        url = ""
    return {
        "url_length": len(url),
        "dot_count": url.count("."),
        "has_ip": 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
        "has_at": 1 if "@" in url else 0,
        "has_hyphen": 1 if "-" in url else 0
    }

# Load legitimate URLs dataset without header, assign columns
legit = pd.read_csv("../data/top-1m.csv", header=None)
# Assign columns - adjust if your CSV has different columns/order
legit.columns = ["rank", "url"]
legit["label"] = 0

# Load phishing URLs dataset - assuming 'url' column exists
phish = pd.read_csv("../data/verified_online.csv")
phish["label"] = 1

# Inspect columns - Uncomment to debug
# print("Legit columns:", legit.columns)
# print("Phish columns:", phish.columns)

# Keep only 'url' and 'label' columns for both
legit = legit[["url", "label"]]
phish = phish[["url", "label"]]

# Combine datasets
df = pd.concat([legit, phish], ignore_index=True)

# Check for any missing urls and drop them
df = df.dropna(subset=["url"])

# Extract features into a DataFrame
features = df["url"].apply(extract_features).apply(pd.Series)
labels = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Check label distribution
print("Label distribution in whole data:")
print(df["label"].value_counts())
print("Label distribution in training set:")
print(y_train.value_counts())
print("Label distribution in test set:")
print(y_test.value_counts())

# Train XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Export simple rule-based classifier rules (mock thresholdpositive_ratio = y_train.mean()
positive_ratio = y_train.mean()
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    base_score=positive_ratio  # automatic class probability
)
model.fit(X_train, y_train)
def rule_based_classifier(f):
    score = 0
    score += 1.0 if f["url_length"] > 75 else 0
    score += 1.0 if f["dot_count"] > 3 else 0
    score += 1.5 if f["has_ip"] else 0
    score += 1.0 if f["has_at"] else 0
    score += 1.0 if f["has_hyphen"] else 0
    return 1 if score > 2.5 else 0

model_rules = {
    "threshold": 2.5,
    "rules": {
        "url_length": "> 75",
        "dot_count": "> 3",
        "has_ip": True,
        "has_at": True,
        "has_hyphen": True
    }
}

with open("model_rules.json", "w") as f:
    json.dump(model_rules, f, indent=4)

print("Model training and rule export complete.")

import joblib
joblib.dump(model, "xgb_model.pkl")

