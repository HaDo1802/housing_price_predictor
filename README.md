# Prices Predictor System - MLOps Project

A machine learning project for predicting house prices using a modular, MLOps-best-practices structure.

## 📁 Project Structure

```
prices-predictor-system/
├── src/
│   ├── components/
│   │   ├── data_preprocessing.py    # Data loading, cleaning, and feature engineering
│   │   ├── model_training.py        # Multi-model training and evaluation
│   │   └── __init__.py
│   ├── pipelines/
│   │   ├── ml_pipeline.py          # Main ML pipeline orchestrator
│   │   └── __init__.py
│   ├── steps/                       # (Legacy - can be removed)
│   ├── analysis/                    # Data analysis notebooks/scripts
│   ├── scripts/                     # Utility scripts
│   ├── utils/                       # Helper utilities
│   └── __init__.py
├── data/                            # Raw data files
├── extracted_data/                  # Extracted data from archives
├── models/                          # Trained models and preprocessors
├── train.py                         # Main training script
├── predict.py                       # Prediction script
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Getting Started

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training

Run the training pipeline to train multiple models and automatically select the best one:

```bash
python train.py
```

This will:

- Load and preprocess the data
- Train 6 different regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regression (SVR)
- Evaluate all models on the test set
- Select the best model based on R² score
- Save the best model and preprocessor to `models/` directory

### 3. Making Predictions

Use the trained model to make predictions on new data:

```bash
python predict.py
```

Or integrate it in your Python code:

```python
from predict import PredictionService
import pandas as pd

# Initialize the service
service = PredictionService()

# Make predictions
new_data = pd.DataFrame({...})  # Your features
predictions = service.predict(new_data)
```

## 📚 Module Documentation

### DataPreprocessor (`src/components/data_preprocessing.py`)

Handles all data preprocessing tasks:

```python
from src.components.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(verbose=True)

# Load data
df = preprocessor.load_data("data/archive.zip")

# Handle missing values
df = preprocessor.handle_missing_values(df, method="mean")

# Detect and handle outliers
df = preprocessor.detect_and_handle_outliers(df, method="iqr", remove=True)

# Complete preprocessing pipeline
X_train, X_test, y_train, y_test, X, y = preprocessor.preprocess(
    df=df,
    target_col="SalePrice",
    test_size=0.2,
    handle_missing_method="mean",
    handle_outliers=True,
    scale_numeric=True,
    encode_categorical=True
)

# Save/Load preprocessor
preprocessor.save_preprocessor("models/preprocessor.pkl")
preprocessor.load_preprocessor("models/preprocessor.pkl")
```

**Available Methods:**

- `load_data()`: Load from ZIP or CSV files
- `handle_missing_values()`: Handle missing data (mean, median, mode, drop)
- `detect_and_handle_outliers()`: Remove or cap outliers (IQR, Z-score)
- `apply_log_transformation()`: Apply log transformation to features
- `identify_feature_types()`: Automatically identify numeric and categorical features
- `preprocess()`: Complete preprocessing pipeline

### ModelTrainer (`src/components/model_training.py`)

Multi-model training and evaluation framework:

```python
from src.components.model_training import ModelTrainer

trainer = ModelTrainer(verbose=True, random_state=42)

# Train all models
trainer.train_all_models(X_train, y_train)

# Evaluate all models
results = trainer.evaluate_all_models(X_test, y_test)

# Get comparison table
comparison = trainer.get_model_comparison(metric="r2")
print(comparison)

# Select best model
best_name, best_model, metrics = trainer.select_best_model(metric="r2")

# Make predictions
predictions = trainer.predict(X_test, use_best=True)

# Save/Load models
trainer.save_best_model("models/best_model.pkl")
trainer.save_all_models("models/")
```

**Available Models:**

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. Support Vector Regression (SVR)

**Evaluation Metrics:**

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

### MLPipeline (`src/pipelines/ml_pipeline.py`)

Orchestrates the complete workflow:

```python
from src.pipelines.ml_pipeline import MLPipeline

pipeline = MLPipeline(
    data_path="data/archive.zip",
    target_column="SalePrice",
    verbose=True
)

# Run complete pipeline
results = pipeline.run()

# Save artifacts
pipeline.save_artifacts(
    model_dir="models/",
    preprocessor_path="models/preprocessor.pkl"
)

# Access results
print(results['best_model_name'])
print(results['metrics'])
print(results['all_results'])
```

## 📊 Example Workflow

```python
# 1. Create pipeline
from src.pipelines.ml_pipeline import MLPipeline

pipeline = MLPipeline("data/archive.zip", "SalePrice")

# 2. Run pipeline
results = pipeline.run()

# 3. Save artifacts
pipeline.save_artifacts()

# 4. Make predictions
from predict import PredictionService
service = PredictionService()
prediction = service.predict(new_data_df)

print(f"Predicted Price: ${prediction[0]:,.2f}")
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
enable_cache: False

settings:
  docker:
    required_integrations:
      - mlflow

model:
  name: prices_predictor
  license: Apache 2.0
  description: Predictor of housing prices.
  tags: ["regression", "housing", "price_prediction"]
```

## 📋 Requirements

See `requirements.txt` for all dependencies. Key packages:

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning models
- matplotlib: Visualization
- seaborn: Statistical visualization

## 📈 Model Performance

The pipeline evaluates models using multiple metrics:

- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)

Models are ranked by R² score by default.

## 🎯 Best Practices

This project follows MLOps best practices:

1. **Modular Design**: Separate concerns (preprocessing, training)
2. **No ZenML Dependency**: Pure scikit-learn for simplicity
3. **Artifact Management**: Save models and preprocessors for deployment
4. **Logging**: Comprehensive logging throughout the pipeline
5. **Reproducibility**: Fixed random seeds for reproducible results
6. **Scalability**: Easy to add new models or preprocessing steps

## 🚀 Deployment

To deploy the model:

1. Train the model: `python train.py`
2. Load in production:
   ```python
   from predict import PredictionService
   service = PredictionService()
   ```
3. Make predictions on new data

## 📝 License

Apache 2.0

## 👤 Author

Your Name

---

**Last Updated**: December 2025
