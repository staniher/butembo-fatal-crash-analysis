"""
build_hourly_series.py
----------------------
Builds the 24-hour accident frequency profile used by the LSTM model.

Aligned with:
- Section 3.1 (feature engineering)
- Section 3.3 (temporal sequence modelling)

Input:
- data/raw/2014_2025_butembo_road_accidents_data.csv

Output:
- data/temporal/hourly_profile.npy
- data/temporal/hourly_metadata.json
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RAW_DATA_PATH = "data/raw/2014_2025_butembo_road_accidents_data.csv"
OUTPUT_DIR = Path("data/temporal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def normalize_hour(value):
    """
    Normalize mixed hour formats:
    '18h20', '5h', '17h00', '08', 8 -> int in [0, 23]
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        v = int(value)
        return v if 0 <= v <= 23 else np.nan

    value = str(value).lower().strip()
    match = re.search(r"(\d{1,2})", value)
    if match:
        hour = int(match.group(1))
        return hour if 0 <= hour <= 23 else np.nan

    return np.nan


def map_target(value):
    """
    Fatal -> 1 ; Non-fatal -> 0
    """
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    if v == "fatal":
        return 1
    if v == "non-fatal":
        return 0
    return np.nan


# ============================================================
# LOAD & CLEAN
# ============================================================

print("[INFO] Loading raw dataset for temporal aggregation...")
df = pd.read_csv(RAW_DATA_PATH)

# Target consistency (optional filter to keep valid labels)
df["accident_TYPE"] = df["accident_TYPE"].apply(map_target)
df = df.dropna(subset=["accident_TYPE"])

# Hour normalization
df["Hour"] = df["Hour"].apply(normalize_hour)
df = df.dropna(subset=["Hour"])
df["Hour"] = df["Hour"].astype(int)

print(f"[INFO] Records retained for temporal profile: {len(df)}")

# ============================================================
# AGGREGATION (24-hour profile)
# ============================================================

# Count total accidents per hour across all dates
hourly_counts = (
    df.groupby("Hour")
      .size()
      .reindex(range(24), fill_value=0)
      .values
)

# Optional: normalize to [0,1] for stable LSTM training
max_count = hourly_counts.max()
hourly_profile = hourly_counts / max_count if max_count > 0 else hourly_counts

# ============================================================
# SAVE OUTPUTS
# ============================================================

np.save(OUTPUT_DIR / "hourly_profile.npy", hourly_profile)

metadata = {
    "description": "Normalized 24-hour accident frequency profile for LSTM",
    "hours": list(range(24)),
    "raw_counts": hourly_counts.tolist(),
    "normalization": "counts / max(counts)" if max_count > 0 else "none",
    "records_used": int(len(df))
}

with open(OUTPUT_DIR / "hourly_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("[INFO] Hourly temporal profile built successfully.")
print("[INFO] Saved:")
print(" - data/temporal/hourly_profile.npy")
print(" - data/temporal/hourly_metadata.json")
