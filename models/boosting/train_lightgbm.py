"""
train_lightgbm.py
-----------------
Train LightGBM classifier.

Aligned with Section 3.2
"""

import numpy as np
import json
from pathlib import Path
import lightgbm as lgb
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
# DATASETS
# ------------------------------------------------------------
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# ------------------------------------------------------------
# PARAMS
# ------------------------------------------------------------
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False)
]

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
model = lgb.train(
    params,
    train_data,
    num_boost_round=600,
    valid_sets=[val_data],
    callbacks=callbacks
)

# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------
p_test = model.predict(X_test)
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
model.save_model(MODEL_DIR / "lightgbm_model.txt")

np.save(RESULTS_DIR / "p_lightgbm_test.npy", p_test)

with open(RESULTS_DIR / "lightgbm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] LightGBM training completed.")
print(metrics)
