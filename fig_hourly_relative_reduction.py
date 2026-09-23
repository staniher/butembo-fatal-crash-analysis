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
    return np.array([np.mean(data[str(h)]) for h in range(24)])

base = hourly_mean(baseline)

plt.figure(figsize=(8,4))

for label, key in {
    "Speed −20%": "speed_limit_20pct",
    "Targeted closure": "targeted_closure"
}.items():
    scen = hourly_mean(scenarios[key])

    # Mask division where baseline is zero or very small
    reduction = np.full_like(base, np.nan)
    mask = base > 1e-6   # seuil de sécurité numérique
    reduction[mask] = (base[mask] - scen[mask]) / base[mask] * 100

    plt.plot(range(24), reduction, marker='o', label=label)

plt.xlabel("Hour of day")
plt.ylabel("Risk reduction (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/figures/hourly_relative_reduction.png", dpi=300)
plt.show()

pip install streamlit pandas mysql-connector-python matplotlib











