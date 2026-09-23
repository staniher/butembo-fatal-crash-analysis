import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths
# =========================
RESULTS_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load baseline simulation
# =========================
with open(RESULTS_DIR / "simulation_baseline.json", "r") as f:
    baseline = json.load(f)

# =========================
# Aggregate temporal risk
# =========================
hours = list(range(24))
avg_risk = []
std_risk = []

for h in hours:
    risks = np.array(baseline[str(h)])
    avg_risk.append(risks.mean())
    std_risk.append(risks.std())

avg_risk = np.array(avg_risk)
std_risk = np.array(std_risk)

# =========================
# Plot
# =========================
plt.figure(figsize=(7,4))

# Mean risk curve
plt.plot(hours, avg_risk, marker='o', linewidth=2, label="Average risk")

# Variability band (± std)
plt.fill_between(
    hours,
    avg_risk - std_risk,
    avg_risk + std_risk,
    alpha=0.2,
    label="±1 std across segments"
)

# Reference hour (18:00)
plt.axvline(
    x=18,
    linestyle='--',
    linewidth=1.5,
    alpha=0.7,
    label="Selected snapshot (18:00)"
)

# Axes & layout
plt.xticks(hours)
plt.xlabel("Hour of day")
plt.ylabel("Average predicted risk")
plt.title("Temporal risk dynamics over 24 hours (baseline)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()

# =========================
# Save & show
# =========================
plt.savefig(FIGURES_DIR / "temporal_risk_24h.png", dpi=300)
plt.show()
