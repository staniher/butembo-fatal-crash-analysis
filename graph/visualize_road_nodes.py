"""
visualize_nodes_simple.py
-------------------------
Minimal visualization of Digital Twin nodes (road segments)
WITHOUT coordinates and WITHOUT pre-built graph file.

Purpose:
- show nodes 0–N as used by the Digital Twin
- provide an illustrative network structure
- generate a 300 DPI figure suitable for an Elsevier article
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASELINE_FILE = Path("results/tables/simulation_baseline.json")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# DETERMINE NUMBER OF NODES
# ------------------------------------------------------------
with open(BASELINE_FILE, "r") as f:
    baseline = json.load(f)

# node IDs come directly from the Digital Twin output
node_ids = sorted(int(k) for k in baseline.keys())
N = len(node_ids)

print(f"[INFO] Found {N} road segments (nodes).")

# ------------------------------------------------------------
# BUILD A SIMPLE ABSTRACT GRAPH
# ------------------------------------------------------------
G = nx.Graph()

# Add nodes
for n in node_ids:
    G.add_node(n)

# Add simple chain connections (purely illustrative)
for i in range(N - 1):
    G.add_edge(node_ids[i], node_ids[i + 1])

# ------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------
pos = nx.spring_layout(G, seed=42)

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------
plt.figure(figsize=(7, 7), dpi=300)

nx.draw(
    G,
    pos,
    node_size=120,
    node_color="steelblue",
    with_labels=True,
    font_size=6,
    edge_color="lightgray",
    alpha=0.9
)

plt.title("Abstract representation of road segments (Digital Twin nodes)")
plt.tight_layout()

out = OUTPUT_DIR / "road_segments_nodes_only.png"
plt.savefig(out, dpi=300)
plt.close()

print(f"[INFO] Node visualization saved to {out}")
