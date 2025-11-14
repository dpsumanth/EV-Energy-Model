# ⚡ EV Energy Consumption Prediction Model

## 🧭 Overview
A production-style machine learning pipeline to predict **electric vehicle (EV) energy consumption (kWh)** from **driving, vehicle, and environmental features**.  
The system uses **gradient boosting ensembles** (XGBoost, LightGBM, CatBoost), **feature engineering**, and **hyperparameter optimization** to achieve high accuracy, and exposes real-time inference through a **Streamlit** app.

---

## 🧠 Problem Statement
Accurate EV energy prediction supports **range estimation**, **battery management**, and **energy-aware route planning**.  
This project builds an end-to-end pipeline: data prep → modeling → evaluation → deployment.

---

## 🧰 Tech Stack
| Layer | Tools |
|---|---|
| Language | Python |
| Data & ML | Pandas, NumPy, scikit-learn |
| Models | XGBoost, LightGBM, CatBoost |
| Optimization | Optuna |
| Visualization / App | Matplotlib (optional), **Streamlit** |
| Packaging | joblib / pickle |
| Dev | Git, VS Code / Jupyter |

---

## 🧩 Project Structure
```bash
EV-Energy-Model/
│
├── data/
│   ├── dataset.csv            # Training/validation data (example placeholder)
│   └── sample_inputs.csv      # Example rows for quick inference testing
│
├── src/
│   ├── preprocessing.py       # Cleaning, encoding, scaling, feature engineering
│   ├── training.py            # Model training + CV + Optuna tuning
│   ├── evaluation.py          # R², RMSE, MAE; diagnostics/plots
│   └── ensemble_model.py      # Weighted/blended ensemble of GBMs
│
├── models/
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   └── catboost_model.pkl
│
├── app.py                     # Streamlit UI for real-time predictions
├── requirements.txt           # Dependencies
└── README.md                  # This file
