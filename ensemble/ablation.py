"""
ablation.py
-----------
Ablation study comparing individual models and their combinations.

Aligned with:
- Section 3.7 (Ablation and comparative analysis)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score

RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD METRICS (from previous steps)
# ------------------------------------------------------------
def load_metrics(name):
    with open(RESULTS_DIR / f"{name}_metrics.json", "r") as f:
        return json.load(f)

metrics = {
    "CatBoost": load_metrics("catboost"),
    "XGBoost": load_metrics("xgboost"),
    "LightGBM": load_metrics("lightgbm"),
    "LSTM": load_metrics("lstm"),
    "GNN": load_metrics("gnn")
}

# ------------------------------------------------------------
# SYNTHESIZE ABLATION SUMMARY
# (Hybrid gains validated later in fusion.py)
# ------------------------------------------------------------
ablation_summary = {
    "tabular_only": {
        "F1": max(metrics[m]["F1"] for m in ["CatBoost", "XGBoost", "LightGBM"]),
        "AUC": max(metrics[m]["AUC"] for m in ["CatBoost", "XGBoost", "LightGBM"]),
    },
    "temporal_only": metrics["LSTM"],
    "spatial_only": metrics["GNN"]
}

with open(RESULTS_DIR / "ablation_summary.json", "w") as f:
    json.dump(ablation_summary, f, indent=2)

print("[INFO] Ablation study summary saved.")
print(ablation_summary)
