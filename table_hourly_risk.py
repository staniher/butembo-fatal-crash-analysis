import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/tables")

with open(BASE / "simulation_baseline.json") as f:
    baseline = json.load(f)

with open(BASE / "simulation_scenarios.json") as f:
    scenarios = json.load(f)

def hourly_mean(data):
    return [np.mean(data[str(h)]) for h in range(24)]

df = pd.DataFrame({
    "Hour": range(24),
    "Baseline": hourly_mean(baseline),
    "Speed_limit_20pct": hourly_mean(scenarios["speed_limit_20pct"]),
    "Targeted_closure": hourly_mean(scenarios["targeted_closure"])
})

df.to_csv("results/tables/table_hourly_risk.csv", index=False)
print(df.head())
