"""
node_features.py
----------------
Computes node feature matrix for each road segment.

Aligned with:
- Section 3.4 (node descriptors x_v)

Input:
- data/raw/2014_2025_butembo_road_accidents_data.csv
- graph/location_index.json

Output:
- graph/node_features.pt
- graph/node_feature_names.json
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

RAW_DATA_PATH = "data/raw/2014_2025_butembo_road_accidents_data.csv"
GRAPH_DIR = Path("graph")

with open(GRAPH_DIR / "location_index.json", "r", encoding="utf-8") as f:
    loc2idx = json.load(f)

num_nodes = len(loc2idx)

print("[INFO] Loading dataset for node features...")
df = pd.read_csv(RAW_DATA_PATH)

# Target mapping
def map_target(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().lower()
    return 1 if v == "fatal" else 0 if v == "non-fatal" else np.nan

df["accident_TYPE"] = df["accident_TYPE"].apply(map_target)
df = df.dropna(subset=["ACCIDENT_LOCATION", "accident_TYPE"])

df["ACCIDENT_LOCATION"] = df["ACCIDENT_LOCATION"].astype(str).str.lower().str.strip()

# Initialize aggregates
counts = np.zeros(num_nodes)
fatal_rate = np.zeros(num_nodes)
avg_hour = np.zeros(num_nodes)

# Hour normalization
def normalize_hour(x):
    try:
        x = int(str(x).split("h")[0])
        return x if 0 <= x <= 23 else np.nan
    except:
        return np.nan

df["Hour"] = df["Hour"].apply(normalize_hour)

for loc, g in df.groupby("ACCIDENT_LOCATION"):
    if loc not in loc2idx:
        continue
    i = loc2idx[loc]
    counts[i] = len(g)
    fatal_rate[i] = g["accident_TYPE"].mean()
    avg_hour[i] = g["Hour"].dropna().mean() if g["Hour"].notna().any() else 0.0

# Stack features
X = np.vstack([
    counts,
    fatal_rate,
    avg_hour / 23.0  # normalize hour
]).T

X = torch.tensor(X, dtype=torch.float)

torch.save(X, GRAPH_DIR / "node_features.pt")

feature_names = ["accident_count", "fatal_rate", "avg_hour_norm"]
with open(GRAPH_DIR / "node_feature_names.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=2)

print("[INFO] Node features saved:", X.shape)
