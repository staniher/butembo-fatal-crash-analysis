import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results/tables")

with open(BASE / "simulation_baseline.json") as f:
    baseline = json.load(f)

with open(BASE / "simulation_scenarios.json") as f:
    scenarios = json.load(f)

def stats_at_hour(data, hour="18"):
    arr = np.array(data[hour])
    return arr.mean(), arr.std(), arr.max()

rows = {
    "Baseline": stats_at_hour(baseline),
    "Speed −20%": stats_at_hour(scenarios["speed_limit_20pct"]),
    "Targeted closure": stats_at_hour(scenarios["targeted_closure"])
}

df = pd.DataFrame.from_dict(
    rows, orient="index",
    columns=["Mean risk", "Std dev", "Max risk"]
)

df.to_csv("results/tables/table_spatial_stats_18h.csv")
print(df)
