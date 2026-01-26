# 🏠 Housing Price Prediction — Production-Grade MLOps Pipeline
<p align="center">
<img src="image/image.png" alt="Real Estate Data Pipeline Cover Image" />
</p>

An end-to-end **production-oriented machine learning pipeline** for predicting housing prices using the Ames Housing dataset.  
This project demonstrates **correct MLOps principles**, including data leakage prevention, train/validation/test separation, experiment tracking, reproducibility, and artifact management.

---

## 📌 Overview

This repository implements a **fully modular ML training and inference system**, designed to mirror how models are built, evaluated, and promoted in real-world production environments.

Key goals of this project:
- Predict house prices using structured tabular data
- Apply **proper train / validation / test workflows**
- Track experiments and configurations
- Persist models and preprocessing artifacts for deployment
- Provide a clean separation between training and inference

---

## 🧠 Core MLOps Concepts Demonstrated

- ✅ Train / Validation / Test split (no leakage)
- ✅ Preprocessing fitted **only on training data**
- ✅ Validation-based model selection
- ✅ Test set used **once** for final evaluation
- ✅ Configuration-driven pipelines (YAML)
- ✅ Experiment tracking with MLflow
- ✅ Reproducible artifacts (model + preprocessor + metadata)
- ✅ Separate training and inference pipelines

---

## 📁 Project Structure

```
housing_price_predictor/
├── README.md
├── config/                         # Modular config set-up
│   ├── config.yaml                 # Entry point for user to customize the config
│   ├── default_config.yaml         # Default config to set the baseline
│   └── config_manager.py           # Modular script that power config-driven setup
│
├── data/
│   ├── AmesHousing.csv             # Raw data
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── data_split/                 # Modular script handles data split
│   │   └── data_splitter.py
│   │
│   ├── features_engineer/          # Modular script handle feature engineer: scaling, imputer, encoder,..
│   │   └── preprocessor.py
│   │
│   └── pipelines/                  # Script that compile others sub-sripts to built full pipelines
│       ├── training_pipeline.py    # Modular script handle training pipeline: load, cleaning, split, features engineer, & train
│       ├── inference_pipeline.py   # Modular script handle inference pipeline
│       ├── fine_tune.py            # Modular script handle fine-tune/ hyperparameter seach
│      
│
├── models/                         # Saved model / artifacts / metadata
│   └── production/                             
│       ├── model.pkl
│       ├── preprocessor.pkl
│       ├── config.yaml
│       └── metadata.json
│
├── notebook/                       # EDA for understanding data
│   ├── EDA.ipynb
│   └── Model_Exploration.ipynb     # EDA for understanding model baseline
│
├── docs/                           # Recommed docs for detailed set-up and functionality
│   ├── PRODUCTION_STRUCTURE.md
│   ├── WORKFLOW_DIAGRAM.md
│   └── QUICK_REFERENCE.txt
│
├── predict.py
├── requirements.txt                # Python dependencies            
```

---

## 🔄 End-to-End Workflow

```
Raw Data
   ↓
Cleaning (light learning)
   ↓
Train / Val / Test Split   
   ↓
Preprocessing 
   ↓
Model Training
   ↓
Validation Evaluation & Selection
   ↓
Final Training (train + val)
   ↓
Test Evaluation (once)
   ↓
Artifact Saving (model, preprocessor, metadata)
```

---

## ✨ Features

### 🧹 Data Preprocessing

**Data Quality Checks**
- Missing value detection and handling  
- Duplicate/Outlier records removal  
- Data type validation and correction  

**Feature Handling**
- Numerical feature scaling using **StandardScaler**
- Categorical feature encoding handled inbalanced data issue
- Feature schema and ordering persisted for consistent inference

### 🧠 Feature Set Used

**Numerical Features**
- Lot Area  
- Total Bsmt SF  
- 1st Flr SF  
- 2nd Flr SF  
- Gr Liv Area  
- Garage Area  
- Overall Qual  
- Overall Cond  
- Year Built  
- Year Remod/Add  
- Bedroom AbvGr  
- Full Bath  
- Half Bath  
- TotRms AbvGrd  
- Fireplaces  
- Garage Cars  

**Categorical Features**
- Neighborhood  
- MS Zoning  
- Bldg Type  
- House Style  
- Foundation  
- Central Air  
- Garage Type  

### ✂️ Data Splitting Strategy

- **Training:** 70%  
- **Validation:** 10%  
- **Test:** 20%  
- Split performed **before preprocessing** to prevent data leakage  
- Fixed random seed for reproducibility  

### 🤖 Models Evaluated ( happen in model_exploratory notebook)
- **Linear Regression** — baseline model  
- **Ridge / Lasso Regression** — regularized linear models  
- **Random Forest Regressor** — ensemble of decision trees  
- **Gradient Boosting Regressor** — sequential boosting  ==> Best candidate & got chosen for production model!
- **Support Vector Regressor (RBF)** — non-linear regression  

### 📊 Evaluation Metrics

- R²  
- RMSE  
- MAE  
- MSE  

Validation metrics are used for **model selection and tuning**.  
The test set is used **once** for final unbiased evaluation.

### ✅ Best Practices Implemented

- ✅ Train / Validation / Test split to prevent data leakage  
- ✅ Preprocessing fitted **only on training data** to data leakage
- ✅ Validation-based model selection  
- ✅ Test set isolated for final reporting  
- ✅ Configuration-driven pipelines (YAML) ==> enable fully centralized control for users
- ✅ Experiment tracking with MLflow   ==> better view/understanding history runs
- ✅ Model and preprocessor persistence  
- ✅ Reproducible runs via fixed random seeds  
- ✅ Modular code structure  
- ✅ Separate inference pipeline for deployment  

---


## 🧪 Experiment Tracking

This project uses **MLflow** for experiment tracking.

Tracked per run:
- Hyperparameters
- Data split configuration
- Validation and test metrics
- Model artifacts
- Preprocessing configuration

Launch MLflow UI:

```bash
mlflow ui
```

Open: http://localhost:5000

---

## 🐳 Docker Containerization

**Why containerize?**
- **Reproducibility:** lock runtime dependencies and OS-level behavior so training/inference behaves the same everywhere.  
- **Portability:** run the same stack locally, on a server, or in CI without environment drift.  
- **Service isolation:** keep MLflow, FastAPI, and Streamlit separated with clear ports and volumes.  
- **Faster onboarding:** one command starts the full stack without manual setup.  

**Set up**

---

## 📦 Saved Artifacts

- `model.pkl` — trained estimator
- `preprocessor.pkl` — fitted preprocessing pipeline
- `config.yaml` — training configuration snapshot
- `metadata.json` — metrics and feature information

---


## 🚀 Why This Project Matters

This project focuses on **engineering discipline**, not just accuracy:
- Prevents data leakage
- Ensures reproducibility
- Mirrors real production ML workflows

---

## 🔮 Future Enhancements

- Cross-validation
- Model interpretability (SHAP)
- CI/CD integration
- Drift detection
- API-based inference

---


## 👤 Author

**Ha Do**  
GitHub: https://github.com/HaDo1802
