"""
build_graph.py
--------------
Builds the road-network graph (edge_index) from harmonized textual locations.

Aligned with:
- Section 3.4 (Spatial propagation modelling using GNNs)

Input:
- data/raw/2014_2025_butembo_road_accidents_data.csv

Output:
- graph/edge_index.pt
- graph/location_index.json
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import itertools
import warnings

warnings.filterwarnings("ignore")

RAW_DATA_PATH = "data/raw/2014_2025_butembo_road_accidents_data.csv"
OUT_DIR = Path("graph")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_location(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    # minimal normalization (keep consistent with preprocessing)
    x = " ".join(x.split())
    return x

print("[INFO] Loading dataset for graph construction...")
df = pd.read_csv(RAW_DATA_PATH)

# Normalize locations
df["ACCIDENT_LOCATION"] = df["ACCIDENT_LOCATION"].apply(normalize_location)
df = df.dropna(subset=["ACCIDENT_LOCATION"])

# Index locations -> nodes
locations = sorted(df["ACCIDENT_LOCATION"].unique().tolist())
loc2idx = {loc: i for i, loc in enumerate(locations)}

with open(OUT_DIR / "location_index.json", "w", encoding="utf-8") as f:
    json.dump(loc2idx, f, indent=2)

print(f"[INFO] Nodes (road segments): {len(locations)}")

# --- Edge construction rules ---
edges = set()

# Rule 1: Co-occurrence within same date (likely adjacency / corridor)
if "date_accident" in df.columns:
    df["date_accident"] = pd.to_datetime(df["date_accident"], errors="coerce")
    for _, g in df.dropna(subset=["date_accident"]).groupby("date_accident"):
        locs = g["ACCIDENT_LOCATION"].unique().tolist()
        for a, b in itertools.combinations(locs, 2):
            edges.add((loc2idx[a], loc2idx[b]))
            edges.add((loc2idx[b], loc2idx[a]))

# Rule 2: Textual intersection hints (shared tokens)
def tokens(loc):
    return set(loc.replace("-", " ").split())

for i, a in enumerate(locations):
    ta = tokens(a)
    for j in range(i + 1, len(locations)):
        b = locations[j]
        tb = tokens(b)
        if len(ta.intersection(tb)) > 0:
            edges.add((loc2idx[a], loc2idx[b]))
            edges.add((loc2idx[b], loc2idx[a]))

# Fallback: ensure connectivity (chain nearest by index if isolated)
deg = {i: 0 for i in range(len(locations))}
for u, v in edges:
    deg[u] += 1

for i in range(len(locations) - 1):
    if deg[i] == 0:
        edges.add((i, i + 1))
        edges.add((i + 1, i))

# Build edge_index
edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
torch.save(edge_index, OUT_DIR / "edge_index.pt")

print("[INFO] edge_index saved:", edge_index.shape)
