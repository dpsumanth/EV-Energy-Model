# ⚡ EV Energy Consumption Prediction Model

## 🧭 Overview
A production-grade **Machine Learning (ML)** pipeline to predict **Electric Vehicle (EV) energy consumption (kWh)** using driving behavior, environmental, and vehicle parameters.  
The model leverages **gradient boosting ensemble algorithms** (XGBoost, LightGBM, CatBoost), **feature engineering**, and **hyperparameter tuning** via Optuna, and is deployed with **Streamlit** for real-time predictions.

---

## 🧠 Problem Statement
Accurate EV energy prediction plays a crucial role in **battery management**, **range estimation**, and **energy-efficient route planning**.  
This project aims to build an optimized ML system that generalizes well across various driving and environmental conditions.

---

## 🧰 Tech Stack
| Layer | Technologies Used |
|--------|-------------------|
| **Language** | Python |
| **Libraries** | Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| **Deployment** | Streamlit |
| **Tools** | VS Code, Git, Jupyter Notebook |
| **Version Control** | Git & GitHub |

---

## 🧩 Project Structure
The following structure outlines all the key directories and components in this project:

```bash
EV-Energy-Model/
│
├── data/
│   ├── dataset.csv             # Training/validation dataset
│   └── sample_inputs.csv       # Example input data for testing model predictions
│
├── src/
│   ├── preprocessing.py        # Data cleaning, encoding, scaling, and feature engineering
│   ├── training.py             # Model training, cross-validation, and Optuna hyperparameter tuning
│   ├── evaluation.py           # Performance metrics (R², RMSE, MAE) and visualization scripts
│   └── ensemble_model.py       # Weighted ensemble combining XGBoost, LightGBM, and CatBoost
│
├── models/
│   ├── xgboost_model.pkl       # Saved XGBoost model
│   ├── lightgbm_model.pkl      # Saved LightGBM model
│   └── catboost_model.pkl      # Saved CatBoost model
│
├── app.py                      # Streamlit web app for real-time prediction
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
