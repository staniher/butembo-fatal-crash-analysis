"""
preprocessing.py
-----------------
Preprocessing pipeline for the Butembo Digital Twin project.

Aligned strictly with:
Section 3.1 – Dataset description and preprocessing

Author: (to be filled)
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
RAW_DATA_PATH = "data/raw/2014_2025_butembo_road_accidents_data.csv"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "accident_TYPE"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_hour(value):
    """
    Normalize mixed hour formats such as:
    '18h20', '5h', '17h00', '08', 8 → integer [0, 23]
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        if 0 <= int(value) <= 23:
            return int(value)
        return np.nan

    value = str(value).lower().strip()
    match = re.search(r"(\d{1,2})", value)
    if match:
        hour = int(match.group(1))
        if 0 <= hour <= 23:
            return hour

    return np.nan


def map_target(value):
    """
    Binary target mapping:
    Fatal     → 1
    Non-fatal → 0
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()
    if value == "fatal":
        return 1
    elif value == "non-fatal":
        return 0
    else:
        return np.nan


def harmonize_damage(value):
    """
    Harmonize vehicle damage condition into:
    {No, Minor, Major}
    """
    if pd.isna(value):
        return "Unknown"

    value = str(value).lower()

    if any(k in value for k in ["aucun", "no", "none", "sans"]):
        return "No"
    elif any(k in value for k in ["leger", "minor", "faible"]):
        return "Minor"
    elif any(k in value for k in ["grave", "major", "important", "total"]):
        return "Major"
    else:
        return "Unknown"


def normalize_category(value):
    """
    Generic categorical normalization.
    """
    if pd.isna(value):
        return "Unknown"
    value = str(value).strip().title()
    return value if value else "Unknown"


# ============================================================
# LOAD DATA
# ============================================================

print("[INFO] Loading raw dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print(f"[INFO] Initial shape: {df.shape}")

# ============================================================
# DROP IRRELEVANT / PII COLUMNS
# ============================================================

PII_COLUMNS = [
    "DRIVER_NAME",
    "CAR_OWNER_NAME"
]

df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns], errors="ignore")

# ============================================================
# TARGET VARIABLE
# ============================================================

df[TARGET_COL] = df[TARGET_COL].apply(map_target)
df = df.dropna(subset=[TARGET_COL])

# ============================================================
# DATE & TIME PROCESSING
# ============================================================

df["Hour"] = df["Hour"].apply(normalize_hour)
df["Hour"] = df["Hour"].fillna(df["Hour"].median()).astype(int)

df["date_accident"] = pd.to_datetime(
    df["date_accident"],
    errors="coerce"
)

# ============================================================
# NUMERIC FEATURES
# ============================================================

if "Number_of_years_in_use" in df.columns:
    df["Number_of_years_in_use"] = pd.to_numeric(
        df["Number_of_years_in_use"],
        errors="coerce"
    )

# ============================================================
# CATEGORICAL HARMONIZATION
# ============================================================

CATEGORICAL_COLS = [
    "TYPE_VEHICULE",
    "VEHICULE_NAME",
    "DRIVERS_STATE",
    "ACCIDENT_LOCATION",
    "ROAD_CONDITION"
]

for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = df[col].apply(normalize_category)

if "VEHICULE_DAMAGE_CONDITION" in df.columns:
    df["VEHICULE_DAMAGE_CONDITION"] = df["VEHICULE_DAMAGE_CONDITION"].apply(
        harmonize_damage
    )

# ============================================================
# FEATURE SELECTION
# ============================================================

FEATURE_COLS = [
    "Hour",
    "TYPE_VEHICULE",
    "VEHICULE_NAME",
    "Number_of_years_in_use",
    "DRIVERS_STATE",
    "VEHICULE_DAMAGE_CONDITION",
    "ACCIDENT_LOCATION",
    "ROAD_CONDITION"
]

FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

numeric_features = ["Hour", "Number_of_years_in_use"]
numeric_features = [c for c in numeric_features if c in X.columns]

categorical_features = list(set(FEATURE_COLS) - set(numeric_features))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ============================================================
# SPLITTING STRATEGY
# ============================================================

print("[INFO] Performing stratified split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

# ============================================================
# FIT TRANSFORM
# ============================================================

print("[INFO] Fitting preprocessing pipeline...")
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

# ============================================================
# SAVE OUTPUTS
# ============================================================

np.save(OUTPUT_DIR / "X_train.npy", X_train_proc)
np.save(OUTPUT_DIR / "X_val.npy", X_val_proc)
np.save(OUTPUT_DIR / "X_test.npy", X_test_proc)

np.save(OUTPUT_DIR / "y_train.npy", y_train.values)
np.save(OUTPUT_DIR / "y_val.npy", y_val.values)
np.save(OUTPUT_DIR / "y_test.npy", y_test.values)

with open(OUTPUT_DIR / "feature_metadata.json", "w") as f:
    json.dump({
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }, f, indent=4)

print("[INFO] Preprocessing completed successfully.")
print(f"[INFO] Train size: {X_train_proc.shape}")
print(f"[INFO] Validation size: {X_val_proc.shape}")
print(f"[INFO] Test size: {X_test_proc.shape}")
