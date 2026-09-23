"""
visualize_nodes_by_risk.py
--------------------------
Color nodes (road segments) by average risk for:
- Baseline
- speed_limit_20pct
- targeted_closure

No GPS, no pre-built graph required.
Stable Matplotlib handling (no colorbar error).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------
BASELINE_FILE  = PROJECT_ROOT / "results" / "tables" / "simulation_baseline.json"
SCENARIOS_FILE = PROJECT_ROOT / "results" / "tables" / "simulation_scenarios.json"

if not BASELINE_FILE.exists():
    raise FileNotFoundError(BASELINE_FILE)

if not SCENARIOS_FILE.exists():
    raise FileNotFoundError(SCENARIOS_FILE)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
with open(BASELINE_FILE, "r") as f:
    baseline = json.load(f)

with open(SCENARIOS_FILE, "r") as f:
    scenarios = json.load(f)

node_ids = sorted(int(k) for k in baseline.keys())
N = len(node_ids)
print(f"[INFO] Found {N} road segments (nodes).")

# ------------------------------------------------------------
# BUILD ABSTRACT GRAPH
# ------------------------------------------------------------
G = nx.Graph()
for n in node_ids:
    G.add_node(n)

for i in range(N - 1):
    G.add_edge(node_ids[i], node_ids[i + 1])

pos = nx.spring_layout(G, seed=42)

# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------
def average_risk(data_dict):
    return {int(k): float(np.mean(v)) for k, v in data_dict.items()}

risk_baseline = average_risk(baseline)
risk_speed = average_risk(scenarios.get("speed_limit_20pct", {}))
risk_closure = average_risk(scenarios.get("targeted_closure", {}))

# ------------------------------------------------------------
# PLOT FUNCTION (FIXED)
# ------------------------------------------------------------
def plot_risk(risk_map, title, filename):
    values = [risk_map[n] for n in node_ids]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=values,
        cmap="Reds",
        node_size=140,
        ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="lightgray",
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=6,
        ax=ax
    )

    cbar = fig.colorbar(nodes, ax=ax)
    cbar.set_label("Average predicted risk")

    ax.set_title(title)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved {filename}")

# ------------------------------------------------------------
# GENERATE FIGURES
# ------------------------------------------------------------
plot_risk(
    risk_baseline,
    "Baseline risk – abstract road network",
    "nodes_risk_baseline.png"
)

if risk_speed:
    plot_risk(
        risk_speed,
        "Scenario: speed limit −20% – risk distribution",
        "nodes_risk_speed_limit.png"
    )

if risk_closure:
    plot_risk(
        risk_closure,
        "Scenario: targeted closure – risk distribution",
        "nodes_risk_targeted_closure.png"
    )
