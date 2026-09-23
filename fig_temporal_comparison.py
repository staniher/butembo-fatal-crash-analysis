import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("results/tables")

with open(BASE / "simulation_baseline.json") as f:
    baseline = json.load(f)

with open(BASE / "simulation_scenarios.json") as f:
    scenarios = json.load(f)

def hourly_mean(data):
    return [np.mean(data[str(h)]) for h in range(24)]

plt.figure(figsize=(8,4))

plt.plot(range(24), hourly_mean(baseline),
         marker='o', label="Baseline")

plt.plot(range(24), hourly_mean(scenarios["speed_limit_20pct"]),
         marker='o', label="Speed −20%")

plt.plot(range(24), hourly_mean(scenarios["targeted_closure"]),
         marker='o', label="Targeted closure")

plt.xlabel("Hour of day")
plt.ylabel("Average predicted risk")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/temporal_comparison_scenarios.png", dpi=300)
plt.show()
