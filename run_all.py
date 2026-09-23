"""
run_all.py
----------
Global orchestrator for the Butembo Digital Twin pipeline.

This script executes the full methodology described in Sections 3.1–3.12
of the paper, in the correct and reproducible order.

Usage:
    python run_all.py
"""

import subprocess
import sys

# ------------------------------------------------------------
# Utility function
# ------------------------------------------------------------
def run_step(description, command):
    print("\n" + "=" * 70)
    print(f"[RUN] {description}")
    print("=" * 70)
    result = subprocess.run(
        [sys.executable] + command,
        capture_output=False
    )
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {description}")
        sys.exit(1)

# ------------------------------------------------------------
# PIPELINE EXECUTION
# ------------------------------------------------------------

# STEP 1 — Preprocessing (3.1)
run_step(
    "STEP 1: Data preprocessing",
    ["preprocessing/preprocessing.py"]
)

run_step(
    "STEP 1b: Build hourly temporal profile",
    ["models/lstm/build_hourly_series.py"]
)

# STEP 2 — Graph construction (3.4)
run_step(
    "STEP 2: Build spatial graph",
    ["graph/build_graph.py"]
)

run_step(
    "STEP 2b: Build node features",
    ["graph/node_features.py"]
)

# STEP 3 — Boosting models (3.2)
run_step(
    "STEP 3: Train CatBoost",
    ["models/boosting/train_catboost.py"]
)

run_step(
    "STEP 3b: Train XGBoost",
    ["models/boosting/train_xgboost.py"]
)

run_step(
    "STEP 3c: Train LightGBM",
    ["models/boosting/train_lightgbm.py"]
)

# STEP 4 — LSTM (3.3)
run_step(
    "STEP 4: Train LSTM",
    ["models/lstm/train_lstm.py"]
)

# STEP 5 — GNN (3.4)
run_step(
    "STEP 5: Train GNN",
    ["models/gnn/train_gnn.py"]
)

# STEP 6 — Ablation & Ensemble (3.7–3.8)
run_step(
    "STEP 6: Ablation study",
    ["ensemble/ablation.py"]
)

run_step(
    "STEP 6b: Ensemble fusion",
    ["ensemble/fusion.py"]
)

# STEP 7 — Evaluation & Calibration (3.6, 3.11)
run_step(
    "STEP 7: Final evaluation metrics",
    ["evaluation/metrics.py"]
)

run_step(
    "STEP 7b: Calibration analysis",
    ["evaluation/calibration.py"]
)

run_step(
    "STEP 7c: Model comparison table",
    ["evaluation/compare_models.py"]
)

# STEP 8 — Digital Twin simulation (3.9–3.10)
run_step(
    "STEP 8: Digital Twin simulation",
    ["simulation/digital_twin.py"]
)

print("\n" + "=" * 70)
print("✅ FULL DIGITAL TWIN PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
