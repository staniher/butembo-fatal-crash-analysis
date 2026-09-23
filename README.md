# A Hybrid Digital Twin Framework for Road Accident Prediction in Butembo (DR Congo) 
# Using Machine Learning, Graph Neural Networks, and Spatio-Temporal Simulation
---

## 1. Project Overview

This project implements a **Hybrid Digital Twin** for road accident risk prediction in **Butembo (DR Congo)**.  
It combines:

- Tabular machine learning (Gradient Boosting),
- Temporal modelling (LSTM),
- Spatial modelling (Graph Neural Networks),
- Ensemble learning,
- A spatio-temporal simulation engine (Digital Twin),
- An API and a visualization dashboard.

The implementation follows **Sections 3.1 to 3.12** of the associated scientific paper.

---

## 2. Project Architecture

ButemboDigitalTwin/
│
├── data/raw/2014_2025_butembo_road_accidents_data.csv
├── preprocessing/
├───── preprocessing.py
├── graph/
├───── build_graph.py
├───── node_features.py
├── models/
├───── boosting/
├───────── train_catboost.py
├───────── train_lightgbm.py
├───────── train_xgboost.py
├───── gnn/
├───────── train_gnn.py
├───── lstm/
├───────── build_hourly_series.py
├───────── train_lstm.py
├── ensemble/
├───── ablation.py
├───── fusion.py
├── evaluation/
├───── calibration.py
├───── compare_models.py
├───── metrics.py
├── simulation/
├───── digital_twin.py
├── api/
├───── main.py
├── dashboard/
├───── app.py
├── results/
├── run_all.py   ##Script executing the full methodology described in Sections 3.1–3.12, in the  correct order.
├── requirements.txt
└── README.md

## 3. Installation

### 3.1 Create environment

conda create -n butembo_dt python=3.10
conda activate butembo_dt

### 3.2 Install dependencies

pip install -r requirements.txt

## 4. Execution Pipeline (MANDATORY ORDER)

### STEP 1 — Data Preprocessing (Section 3.1)

python preprocessing/preprocessing.py
python models/lstm/build_hourly_series.py

Outputs (N.B:The file hourly_profile.npy encodes the collective temporal memory of accident occurrences and allows the Digital Twin to exhibit explicit time-dependent dynamics):

	data/processed/*.npy
	data/temporal/hourly_profile.npy

### STEP 2 — Graph Construction (Section 3.4)

python graph/build_graph.py
python graph/node_features.py

Outputs:

	graph/edge_index.pt
	graph/node_features.pt

### STEP 3 — Tabular Models (Section 3.2)

python models/boosting/train_catboost.py
python models/boosting/train_xgboost.py
python models/boosting/train_lightgbm.py

Outputs:

	Trained models
	Metrics in results/tables/

### STEP 4 — Temporal Model (LSTM) (Section 3.3)

python models/lstm/train_lstm.py

### STEP 5 — Spatial Model (GNN) (Section 3.4)

python models/gnn/train_gnn.py

### STEP 6 — Ablation & Ensemble (Sections 3.7–3.8)

python ensemble/ablation.py
python ensemble/fusion.py

### STEP 7 — Evaluation & Calibration (Sections 3.6, 3.11)

python evaluation/metrics.py
python evaluation/calibration.py
python evaluation/compare_models.py


### STEP 8 — Digital Twin Simulation (Sections 3.9–3.10)

python simulation/digital_twin.py

## 5. API Execution

uvicorn api.main:app --reload

Endpoints:

	/baseline

	/scenarios

## 6. Dashboard Execution

streamlit run dashboard/app.py

## 7. Results

N.B: To see graphes, we added two other codes in the graph folder. Thus graph becomes:

├── graph/
├───── build_graph.py
├───── node_features.py
├───── visualize_road_nodes.py 
├───── visualize_nodes_by_risk.py

All numerical results and simulation outputs are stored in:

	results/tables/

## 8. Reproducibility

	1. Fixed random seeds
	2. Deterministic preprocessing
	3. Clear execution order
	4. No hidden dependencies

## 9. License & Academic Use

This project is intended for academic and research purposes.