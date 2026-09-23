"""
analysis_additional_figures.py
------------------------------
Generate additional analytical figures from Digital Twin simulations.

Inputs (REAL project files):
- results/tables/simulation_baseline.json
- results/tables/simulation_scenarios.json

Figures generated (300 DPI, Elsevier-ready):
A. Scatter: Baseline vs Speed reduction
B. Histogram: Risk distributions
C. Bar chart: Top-10 highest-risk segments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASELINE_FILE = Path("results/tables/simulation_baseline.json")
SCENARIOS_FILE = Path("results/tables/simulation_scenarios.json")
OUTPUT_DIR = Path("results/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD & AGGREGATE DATA
# ------------------------------------------------------------
def average_risk_from_simulation(sim_data):
    """
    sim_data: dict {hour -> list of risks}
    return: np.array of average risk per segment
    """
    hourly_values = []
    for hour in sim_data:
        hourly_values.append(sim_data[hour])

    hourly_values = np.array(hourly_values)  # (24, n_segments)
    return hourly_values.mean(axis=0)


# Load baseline
with open(BASELINE_FILE, "r") as f:
    baseline_json = json.load(f)

baseline_avg = average_risk_from_simulation(baseline_json)

# Load scenarios
with open(SCENARIOS_FILE, "r") as f:
    scenarios_json = json.load(f)

speed_avg = average_risk_from_simulation(
    scenarios_json["speed_limit_20pct"]
)

# ------------------------------------------------------------
# FIGURE A — Scatter plot
# ------------------------------------------------------------
plt.figure(figsize=(6, 6), dpi=300)
plt.scatter(baseline_avg, speed_avg, alpha=0.7)
plt.plot(
    [baseline_avg.min(), baseline_avg.max()],
    [baseline_avg.min(), baseline_avg.max()],
    linestyle="--",
    color="gray"
)

plt.xlabel("Baseline average risk")
plt.ylabel("Speed reduction average risk")
plt.title("Baseline vs speed reduction risk per segment")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_baseline_vs_speed.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# FIGURE B — Histogram comparison
# ------------------------------------------------------------
plt.figure(figsize=(7, 4), dpi=300)
plt.hist(baseline_avg, bins=15, alpha=0.6, label="Baseline")
plt.hist(speed_avg, bins=15, alpha=0.6, label="Speed reduction")

plt.xlabel("Predicted risk")
plt.ylabel("Number of segments")
plt.legend()
plt.title("Risk distribution under baseline and speed reduction")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_risk_comparison.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# FIGURE C — Top-10 highest-risk segments
# ------------------------------------------------------------
segments = np.arange(len(baseline_avg))
top_idx = np.argsort(baseline_avg)[-10:][::-1]

plt.figure(figsize=(7, 4), dpi=300)
plt.bar(segments[top_idx], baseline_avg[top_idx], label="Baseline")
plt.bar(
    segments[top_idx],
    speed_avg[top_idx],
    label="Speed reduction",
    alpha=0.7
)

plt.xlabel("Segment ID")
plt.ylabel("Average predicted risk")
plt.title("Top-10 highest-risk road segments")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_segments_risk.png", dpi=300)
plt.close()

print("[INFO] Additional figures A–C generated successfully.")
