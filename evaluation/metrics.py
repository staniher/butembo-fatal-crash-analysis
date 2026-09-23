"""
metrics.py
----------
Final evaluation of the hybrid ensemble.

Aligned with Section 3.11
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
y_test = np.load(DATA_DIR / "y_test.npy")

# IMPORTANT: load FINAL ensemble probabilities (no recomputation)
p_final = np.load(RESULTS_DIR / "p_final_test.npy")

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
y_pred = (p_final >= 0.5).astype(int)

metrics = {
    "F1": float(f1_score(y_test, y_pred, zero_division=0)),
    "AUC": float(roc_auc_score(y_test, p_final)),
    "Accuracy": float(accuracy_score(y_test, y_pred))
}

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
with open(RESULTS_DIR / "final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] Final evaluation metrics saved.")
print(metrics)
