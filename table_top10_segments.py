import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/tables")

with open(BASE / "simulation_baseline.json") as f:
    baseline = json.load(f)

with open(BASE / "simulation_scenarios.json") as f:
    scenarios = json.load(f)

hour = "18"

baseline_arr = np.array(baseline[hour])
top_idx = np.argsort(baseline_arr)[-10:][::-1]

df = pd.DataFrame({
    "Segment ID": top_idx,
    "Baseline": baseline_arr[top_idx],
    "Speed −20%": np.array(scenarios["speed_limit_20pct"][hour])[top_idx],
    "Targeted closure": np.array(scenarios["targeted_closure"][hour])[top_idx]
})

df["Reduction speed (%)"] = (
    (df["Baseline"] - df["Speed −20%"]) / df["Baseline"] * 100
)

df["Reduction closure (%)"] = (
    (df["Baseline"] - df["Targeted closure"]) / df["Baseline"] * 100
)

df.to_csv("results/tables/table_top10_segments_18h.csv", index=False)
print(df)
