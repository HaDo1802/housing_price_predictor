"""
Main ML Pipeline

This script orchestrates the complete machine learning workflow:
1. Data loading and preprocessing
2. Multi-model training
3. Model evaluation and selection
4. Model saving and deployment
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.components.data_preprocessing import DataPreprocessor
from src.components.model_training import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Complete machine learning pipeline from data preprocessing to model selection.
    """

    def __init__(self, data_path: str, target_column: str, verbose: bool = True):
        """
        Initialize the ML Pipeline.

        Parameters:
        data_path (str): Path to the data file
        target_column (str): Name of the target column
        verbose (bool): Whether to log pipeline steps
        """
        self.data_path = data_path
        self.target_column = target_column
        self.verbose = verbose

        self.preprocessor = DataPreprocessor(verbose=verbose)
        self.trainer = ModelTrainer(verbose=verbose)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None

    def run(self) -> dict:
        """
        Execute the complete ML pipeline.

        Returns:
        dict: Pipeline results including best model and metrics
        """
        try:
            # Step 1: Load data
            self._load_data()

            # Step 2: Preprocess data
            self._preprocess_data()

            # Step 3: Train models
            self._train_models()

            # Step 4: Evaluate models
            self._evaluate_models()

            # Step 5: Select best model
            results = self._select_best_model()

            if self.verbose:
                logger.info("Pipeline completed successfully!")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _load_data(self) -> None:
        """Load data from file."""
        if self.verbose:
            logger.info(f"Loading data from {self.data_path}")

        self.df = self.preprocessor.load_data(self.data_path)

        if self.verbose:
            logger.info(f"Data loaded: {self.df.shape}")

    def _preprocess_data(self) -> None:
        """Preprocess the data."""
        if self.verbose:
            logger.info("Starting data preprocessing...")

        self.X_train, self.X_test, self.y_train, self.y_test, _, _ = (
            self.preprocessor.preprocess(
                df=self.df,
                target_col=self.target_column,
                test_size=0.2,
                random_state=42,
                handle_missing_method="mean",
                handle_outliers=True,
                scale_numeric=True,
                encode_categorical=True,
            )
        )

        if self.verbose:
            logger.info(
                f"Preprocessing complete - Train: {self.X_train.shape}, Test: {self.X_test.shape}"
            )

    def _train_models(self) -> None:
        """Train all models."""
        if self.verbose:
            logger.info("Starting model training...")

        self.trainer.train_all_models(self.X_train, self.y_train)

        if self.verbose:
            logger.info("Model training completed!")

    def _evaluate_models(self) -> None:
        """Evaluate all trained models."""
        if self.verbose:
            logger.info("Starting model evaluation...")

        self.trainer.evaluate_all_models(self.X_test, self.y_test)

        # Display comparison table
        comparison_df = self.trainer.get_model_comparison(metric="r2")
        if self.verbose:
            logger.info("\nModel Comparison Table:")
            logger.info("\n" + comparison_df.to_string())

    def _select_best_model(self) -> dict:
        """Select the best model and return results."""
        best_name, best_model, metrics = self.trainer.select_best_model(metric="r2")

        results = {
            "best_model_name": best_name,
            "best_model": best_model,
            "metrics": metrics,
            "all_results": self.trainer.results,
        }

        if self.verbose:
            logger.info(f"\nBest Model: {best_name}")
            logger.info(f"Metrics: {metrics}")

        return results

    def save_artifacts(
        self, model_dir: str = "models", preprocessor_path: str = None
    ) -> None:
        """
        Save the best model and preprocessor.

        Parameters:
        model_dir (str): Directory to save models
        preprocessor_path (str): Path to save preprocessor
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save best model
        best_model_path = os.path.join(model_dir, "best_model.pkl")
        self.trainer.save_best_model(best_model_path)

        # Save all models
        self.trainer.save_all_models(model_dir)

        # Save preprocessor
        if preprocessor_path is None:
            preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")

        self.preprocessor.save_preprocessor(preprocessor_path)

        if self.verbose:
            logger.info(f"Artifacts saved to {model_dir}")


def main():
    """Main entry point for the ML pipeline."""
    # Configuration
    DATA_PATH = "data/archive.zip"
    TARGET_COLUMN = "SalePrice"
    MODEL_DIR = "models"

    # Create pipeline
    pipeline = MLPipeline(
        data_path=DATA_PATH, target_column=TARGET_COLUMN, verbose=True
    )

    # Run pipeline
    results = pipeline.run()

    # Save artifacts
    pipeline.save_artifacts(model_dir=MODEL_DIR)

    return results


if __name__ == "__main__":
    results = main()
    print("\nPipeline completed successfully!")
    print(f"Best Model: {results['best_model_name']}")
    print(f"Metrics: {results['metrics']}")
