"""
train_xgboost.py
----------------
Train XGBoost classifier and save out-of-sample probabilities.

Aligned with Section 3.2
"""

import numpy as np
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
MODEL_DIR = Path("models/boosting")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------
p_test = model.predict_proba(X_test)[:, 1]
y_pred = (p_test >= 0.5).astype(int)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
metrics = {
    "F1": float(f1_score(y_test, y_pred, zero_division=0)),
    "AUC": float(roc_auc_score(y_test, p_test))
}

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
model.save_model(MODEL_DIR / "xgboost_model.json")

np.save(RESULTS_DIR / "p_xgboost_test.npy", p_test)

with open(RESULTS_DIR / "xgboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] XGBoost training completed.")
print(metrics)
