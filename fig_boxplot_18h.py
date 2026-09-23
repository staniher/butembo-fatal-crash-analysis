import json
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("results/tables")

with open(BASE / "simulation_baseline.json") as f:
    baseline = json.load(f)

with open(BASE / "simulation_scenarios.json") as f:
    scenarios = json.load(f)

data = [
    baseline["18"],
    scenarios["speed_limit_20pct"]["18"],
    scenarios["targeted_closure"]["18"]
]

labels = ["Baseline", "Speed −20%", "Targeted closure"]

plt.figure(figsize=(7,4))
plt.boxplot(data, labels=labels, showfliers=False)
plt.ylabel("Predicted risk")
plt.title("Spatial risk distribution at 18:00")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/boxplot_spatial_18h.png", dpi=300)
plt.show()
