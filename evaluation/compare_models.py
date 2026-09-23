"""
compare_models.py
-----------------
Aggregate evaluation metrics across all models for reporting
and comparative analysis.

Important principles:
- This script DOES NOT recompute any metric.
- It ONLY reads metrics already computed and saved by:
    * train_catboost.py
    * train_xgboost.py
    * train_lightgbm.py
    * train_lstm.py
    * train_gnn.py
    * metrics.py (final ensemble)

Aligned with:
- Section 3.6 (Model evaluation)
- Section 3.11 (Comparative analysis)
"""

import json
from pathlib import Path

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# METRICS FILES (EXPLICIT, NO MAGIC)
# ------------------------------------------------------------
MODEL_FILES = {
    "catboost": "catboost_metrics.json",
    "xgboost": "xgboost_metrics.json",
    "lightgbm": "lightgbm_metrics.json",
    "lstm": "lstm_metrics.json",
    "gnn": "gnn_metrics.json",
    "final": "final_metrics.json"
}

summary = {}

# ------------------------------------------------------------
# LOAD METRICS (NO RECOMPUTATION)
# ------------------------------------------------------------
for model_name, filename in MODEL_FILES.items():
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path, "r") as f:
            summary[model_name] = json.load(f)
    else:
        summary[model_name] = {
            "warning": "metrics file not found"
        }

# ------------------------------------------------------------
# SAVE COMPARISON TABLE
# ------------------------------------------------------------
with open(RESULTS_DIR / "comparison_table.json", "w") as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------------------
# LOG
# ------------------------------------------------------------
print("[INFO] Model comparison table saved.")
for model, metrics in summary.items():
    print(f" - {model}: {metrics}")
