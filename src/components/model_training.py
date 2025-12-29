"""
Model Training Module

This module handles model training with multiple algorithms and automatic
selection of the best model based on validation metrics.
"""

import logging
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Multi-model training and evaluation framework that tests various
    regression models and selects the best one based on validation metrics.
    """

    def __init__(self, verbose: bool = True, random_state: int = 42):
        """
        Initialize the ModelTrainer.

        Parameters:
        verbose (bool): Whether to log training steps
        random_state (int): Random seed for reproducibility
        """
        self.verbose = verbose
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

        # Initialize available models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize a dictionary of models to test."""
        self.models = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0, random_state=self.random_state),
            "lasso": Lasso(alpha=0.1, random_state=self.random_state),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=self.random_state
            ),
            "svr": SVR(kernel="rbf", C=100, epsilon=0.1),
        }

        if self.verbose:
            logger.info(f"Initialized {len(self.models)} models for training")

    def add_model(self, name: str, model: Any) -> None:
        """
        Add a custom model to the training pipeline.

        Parameters:
        name (str): Model name
        model: Scikit-learn compatible model
        """
        self.models[name] = model
        if self.verbose:
            logger.info(f"Added model: {name}")

    def train_model(
        self, model_name: str, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """
        Train a single model.

        Parameters:
        model_name (str): Name of the model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available: {list(self.models.keys())}"
            )

        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model

        if self.verbose:
            logger.info(f"Trained model: {model_name}")

    def evaluate_model(
        self, model_name: str, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.

        Parameters:
        model_name (str): Name of the model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target

        Returns:
        Dict[str, float]: Dictionary of evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")

        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)

        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        self.results[model_name] = metrics

        if self.verbose:
            logger.info(
                f"Evaluated {model_name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}"
            )

        return metrics

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train all available models.

        Parameters:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        """
        if self.verbose:
            logger.info(f"Starting training of {len(self.models)} models...")

        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")

    def evaluate_all_models(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Evaluate all trained models.

        Parameters:
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target

        Returns:
        Dict[str, Dict]: Results for all models
        """
        if self.verbose:
            logger.info(f"Evaluating {len(self.trained_models)} models...")

        for model_name in self.trained_models.keys():
            try:
                self.evaluate_model(model_name, X_test, y_test)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")

        return self.results

    def select_best_model(self, metric: str = "r2") -> Tuple[str, Any, Dict]:
        """
        Select the best model based on a given metric.

        Parameters:
        metric (str): Metric to optimize ('r2', 'rmse', 'mae', 'mse')

        Returns:
        Tuple[str, model, dict]: (model_name, best_model, metrics)
        """
        if not self.results:
            raise ValueError("No results available. Train and evaluate models first.")

        if metric == "r2":
            # Higher R² is better
            best_name = max(
                self.results, key=lambda x: self.results[x].get("r2", -float("inf"))
            )
        elif metric in ["rmse", "mae", "mse"]:
            # Lower error is better
            best_name = min(
                self.results, key=lambda x: self.results[x].get(metric, float("inf"))
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        self.best_model_name = best_name
        self.best_model = self.trained_models[best_name]

        if self.verbose:
            logger.info(f"Selected best model: {best_name} (metric: {metric})")
            logger.info(f"Metrics: {self.results[best_name]}")

        return best_name, self.best_model, self.results[best_name]

    def get_model_comparison(self, metric: str = "r2") -> pd.DataFrame:
        """
        Get a comparison table of all models.

        Parameters:
        metric (str): Metric to sort by

        Returns:
        pd.DataFrame: Comparison table
        """
        if not self.results:
            raise ValueError("No results available. Train and evaluate models first.")

        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values(by=metric, ascending=(metric != "r2"))

        if self.verbose:
            logger.info("\nModel Comparison:")
            logger.info(comparison_df)

        return comparison_df

    def predict(
        self, X: np.ndarray, use_best: bool = True, model_name: str = None
    ) -> np.ndarray:
        """
        Make predictions using a model.

        Parameters:
        X (np.ndarray): Features to predict on
        use_best (bool): Whether to use the best model
        model_name (str): Specific model to use (if use_best=False)

        Returns:
        np.ndarray: Predictions
        """
        if use_best:
            if self.best_model is None:
                raise ValueError(
                    "No best model selected. Call select_best_model first."
                )
            return self.best_model.predict(X)
        else:
            if model_name is None:
                raise ValueError("model_name required when use_best=False")
            if model_name not in self.trained_models:
                raise ValueError(f"Model '{model_name}' not trained")
            return self.trained_models[model_name].predict(X)

    def save_best_model(self, filepath: str) -> None:
        """
        Save the best trained model to disk.

        Parameters:
        filepath (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model to save. Call select_best_model first.")

        with open(filepath, "wb") as f:
            pickle.dump(self.best_model, f)

        if self.verbose:
            logger.info(f"Best model ({self.best_model_name}) saved to {filepath}")

    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.

        Parameters:
        filepath (str): Path to load the model

        Returns:
        Trained model
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)

        if self.verbose:
            logger.info(f"Model loaded from {filepath}")

        return model

    def save_all_models(self, directory: str) -> None:
        """
        Save all trained models to a directory.

        Parameters:
        directory (str): Directory to save models
        """
        import os

        os.makedirs(directory, exist_ok=True)

        for model_name, model in self.trained_models.items():
            filepath = os.path.join(directory, f"{model_name}.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(model, f)

            if self.verbose:
                logger.info(f"Saved model: {model_name}")

    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """
        Get feature importance from tree-based models.

        Parameters:
        model_name (str): Model to get importance from (defaults to best model)

        Returns:
        Dict[str, float]: Feature importance scores
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model specified or best model selected")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model '{model_name}' not trained")
            model = self.trained_models[model_name]

        if not hasattr(model, "feature_importances_"):
            logger.warning(f"Model {model_name} does not support feature importance")
            return {}

        return dict(enumerate(model.feature_importances_))
