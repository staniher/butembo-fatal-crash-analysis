"""
app.py
------
Streamlit dashboard for visualizing the Butembo Digital Twin outputs.

Aligned with:
- Section 3.10.4 (Visualization and deployment)
"""

import streamlit as st
import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results/tables")

st.set_page_config(page_title="Butembo Digital Twin", layout="wide")

st.title("🚦 Butembo Digital Twin – Road Accident Risk")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
baseline_path = RESULTS_DIR / "simulation_baseline.json"
scenarios_path = RESULTS_DIR / "simulation_scenarios.json"

if not baseline_path.exists():
    st.error("Baseline simulation not found.")
    st.stop()

with open(baseline_path, "r") as f:
    baseline = json.load(f)

scenarios = {}
if scenarios_path.exists():
    with open(scenarios_path, "r") as f:
        scenarios = json.load(f)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.header("Options")
hour = st.sidebar.slider("Hour of day", 0, 23, 8)

scenario_name = st.sidebar.selectbox(
    "Scenario",
    ["Baseline"] + list(scenarios.keys())
)

# ------------------------------------------------------------
# SELECT DATA
# ------------------------------------------------------------
if scenario_name == "Baseline":
    data = baseline[str(hour)]
else:
    data = scenarios[scenario_name][str(hour)]

data = np.array(data)

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------
st.subheader(f"Risk distribution – Hour {hour}:00 ({scenario_name})")

st.write(f"Number of road segments: {len(data)}")

st.bar_chart(data)

st.markdown("""
### Interpretation
- Higher bars indicate higher predicted accident risk.
- Compare scenarios to assess the impact of interventions
  such as speed reduction or targeted closures.
""")
