"""
train_gnn.py
------------
Train a Graph Neural Network (GCN) on the reconstructed road network.

Aligned with:
- Section 3.4 (Spatial propagation modelling using GNNs)

Inputs:
- graph/edge_index.pt
- graph/node_features.pt

Outputs:
- models/gnn/gnn_model.pt
- results/tables/gnn_metrics.json
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
GRAPH_DIR = Path("graph")
OUT_MODEL = Path("models/gnn/gnn_model.pt")
OUT_METRICS = Path("results/tables/gnn_metrics.json")
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD GRAPH DATA
# ------------------------------------------------------------
edge_index = torch.load(GRAPH_DIR / "edge_index.pt")      # [2, E]
X = torch.load(GRAPH_DIR / "node_features.pt")            # [N, F]

num_nodes = X.size(0)

# ------------------------------------------------------------
# BUILD NODE LABELS (proxy)
# ------------------------------------------------------------
# We use fatal_rate (feature index 1) as supervision proxy
# Consistent with data-limited context described in the paper
y = (X[:, 1] >= X[:, 1].mean()).long()

# Train / test split on nodes
torch.manual_seed(42)
perm = torch.randperm(num_nodes)
split = int(0.8 * num_nodes)
train_idx = perm[:split]
test_idx  = perm[split:]

# ------------------------------------------------------------
# MODEL DEFINITION
# ------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x).squeeze()

model = GCN(in_channels=X.size(1), hidden_channels=16)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.BCELoss()

# ------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------
model.train()
for epoch in range(300):
    optimizer.zero_grad()
    out = model(X, edge_index)
    loss = loss_fn(out[train_idx], y[train_idx].float())
    loss.backward()
    optimizer.step()

# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    proba = model(X, edge_index).detach().cpu().numpy()

pred = (proba >= 0.5).astype(int)

y_true = y.cpu().numpy()

metrics = {
    "F1": float(f1_score(y_true[test_idx], pred[test_idx])),
    "AUC": float(roc_auc_score(y_true[test_idx], proba[test_idx]))
}

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
torch.save(model.state_dict(), OUT_MODEL)
with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] GNN training completed.")
print(metrics)
