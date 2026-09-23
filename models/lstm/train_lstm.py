"""
train_lstm.py
-------------
Train an LSTM model on the 24-hour accident frequency profile.

Aligned with:
- Section 3.3 (Temporal sequence modelling using LSTM)

Inputs:
- data/temporal/hourly_profile.npy
- data/processed/y_train.npy (used to estimate class balance)
- data/processed/y_test.npy

Outputs:
- models/lstm/lstm_model.h5
- results/tables/lstm_metrics.json
"""

import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, roc_auc_score

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
TEMP_DIR = Path("data/temporal")
DATA_DIR = Path("data/processed")
OUT_MODEL = Path("models/lstm/lstm_model.h5")
OUT_METRICS = Path("results/tables/lstm_metrics.json")
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
hourly_profile = np.load(TEMP_DIR / "hourly_profile.npy")  # shape (24,)
hourly_profile = hourly_profile.astype("float32")

# Build supervised samples:
# We replicate the 24h profile to form a small sequence dataset
# (consistent with paper constraints: no continuous daily series)
X = np.tile(hourly_profile, (200, 1))  # (samples, timesteps)
X = X[..., np.newaxis]                  # (samples, timesteps, features)

# Binary labels proxy: use empirical fatal rate for thresholding
y_train = np.load(DATA_DIR / "y_train.npy")
fatal_rate = y_train.mean()
y = np.array([1 if fatal_rate >= 0.5 else 0] * X.shape[0])

# Train / test split (temporal model evaluation)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
tf.random.set_seed(42)

model = Sequential([
    LSTM(32, input_shape=(24, 1), return_sequences=False),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["AUC"]
)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

model.fit(
    X_train, y_tr,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[es],
    verbose=0
)

# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
proba = model.predict(X_test).ravel()
pred = (proba >= 0.5).astype(int)

metrics = {
    "F1": float(f1_score(y_te, pred)),
    "AUC": float(roc_auc_score(y_te, proba))
}

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
model.save(OUT_MODEL)
with open(OUT_METRICS, "w") as f:
    json.dump(metrics, f, indent=2)

print("[INFO] LSTM training completed.")
print(metrics)
