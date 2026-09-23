"""
digital_twin.py
---------------
Spatio-temporal Digital Twin simulation for road accident risk in Butembo.

Aligned with:
- Section 3.9 (Integration into the digital twin simulation environment)
- Section 3.10 (Digital Twin implementation)

Inputs:
- results/tables/ensemble_weights.json
- graph/edge_index.pt
- graph/node_features.pt
- data/temporal/hourly_profile.npy

Outputs:
- results/tables/simulation_baseline.json
- results/tables/simulation_scenarios.json
"""

import json
import numpy as np
import torch
from pathlib import Path

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
GRAPH_DIR = Path("graph")
TEMP_DIR = Path("data/temporal")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD INPUTS
# ------------------------------------------------------------
with open(RESULTS_DIR / "ensemble_weights.json", "r") as f:
    cfg = json.load(f)

alpha, beta, gamma = cfg["alpha"], cfg["beta"], cfg["gamma"]

edge_index = torch.load(GRAPH_DIR / "edge_index.pt")
node_features = torch.load(GRAPH_DIR / "node_features.pt").numpy()

hourly_profile = np.load(TEMP_DIR / "hourly_profile.npy")  # shape (24,)

num_nodes = node_features.shape[0]

# ------------------------------------------------------------
# BASELINE RISK (per node)
# ------------------------------------------------------------
# Use fatal_rate (feature index 1) as baseline spatial risk
spatial_risk = node_features[:, 1]  # [N]

# Normalize
if spatial_risk.max() > 0:
    spatial_risk = spatial_risk / spatial_risk.max()

# Temporal risk modifier (24h)
temporal_modifier = hourly_profile  # already normalized

# ------------------------------------------------------------
# SIMULATION CORE
# ------------------------------------------------------------
def simulate_day(intervention=None):
    """
    Simulate risk for each road segment over 24 hours.

    intervention:
        dict with optional keys:
        - 'speed_reduction': float in (0,1]
        - 'closure_nodes': list of node indices
    """
    intervention = intervention or {}
    speed_coeff = intervention.get("speed_reduction", 1.0)
    closed = set(intervention.get("closure_nodes", []))

    risk_map = {}

    for h in range(24):
        hour_risk = []
        for i in range(num_nodes):
            if i in closed:
                hour_risk.append(0.0)
                continue

            # Hybrid risk composition (simplified, consistent with ensemble)
            p = spatial_risk[i]
            p = p * temporal_modifier[h]
            p = p * speed_coeff

            hour_risk.append(float(p))

        risk_map[h] = hour_risk

    return risk_map

# ------------------------------------------------------------
# BASELINE SIMULATION
# ------------------------------------------------------------
baseline = simulate_day()

with open(RESULTS_DIR / "simulation_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

# ------------------------------------------------------------
# INTERVENTION SCENARIOS
# ------------------------------------------------------------
scenarios = {
    "speed_limit_20pct": simulate_day(
        intervention={"speed_reduction": 0.8}
    ),
    "targeted_closure": simulate_day(
        intervention={"closure_nodes": list(range(min(5, num_nodes)))}
    )
}

with open(RESULTS_DIR / "simulation_scenarios.json", "w") as f:
    json.dump(scenarios, f, indent=2)

print("[INFO] Digital Twin simulation completed.")
print("[INFO] Outputs:")
print(" - simulation_baseline.json")
print(" - simulation_scenarios.json")
