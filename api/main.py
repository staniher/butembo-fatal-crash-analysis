"""
main.py
-------
FastAPI interface for the Butembo Digital Twin.

Aligned with:
- Section 3.10.4 (System architecture and deployment)
"""

from fastapi import FastAPI
import json
from pathlib import Path

app = FastAPI(
    title="Butembo Digital Twin API",
    description="Hybrid ML-based road accident risk prediction and simulation",
    version="1.0"
)

RESULTS_DIR = Path("results/tables")

@app.get("/")
def root():
    return {
        "message": "Butembo Digital Twin API is running",
        "endpoints": ["/baseline", "/scenarios"]
    }

@app.get("/baseline")
def get_baseline_simulation():
    """
    Return baseline 24h risk simulation
    """
    path = RESULTS_DIR / "simulation_baseline.json"
    if not path.exists():
        return {"error": "Baseline simulation not found"}
    with open(path, "r") as f:
        return json.load(f)

@app.get("/scenarios")
def get_scenarios():
    """
    Return intervention scenarios simulation
    """
    path = RESULTS_DIR / "simulation_scenarios.json"
    if not path.exists():
        return {"error": "Scenarios simulation not found"}
    with open(path, "r") as f:
        return json.load(f)
