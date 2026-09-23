"""
train_catboost.py
-----------------
Train CatBoost classifier on preprocessed tabular data.

Aligned with:
- Section 3.2 (Nonlinear modelling through gradient boosting)

Inputs:
- data/processed/X_train.npy, y_train.npy
- data/processed/X_val.npy, y_val.npy
- data/processed/X_test.npy, y_test.npy
- data/processed/feature_metadata.json

Outputs:
- models/boosting/catboost_model.cbm
- results/tables/catboost_metrics.json
"""
"""
train_catboost.py
-----------------
Train CatBoost classifier and save out-of-sample probabilities.
"""

import numpy as np
import json
from pathlib import Path
from catboost import CatBoostClassifier
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
model = CatBoostClassifier(
    iterations=600,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=100
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# ------------------------------------------------------------
# PREDICT (OUT-OF-SAMPLE)
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
model.save_model(MODEL_DIR / "catboost_model.cbm")

np.save(RESULTS_DIR / "p_catboost_test.npy", p_test)

with open(RESULTS_DIR / "catboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] CatBoost training completed.")
print(metrics)
