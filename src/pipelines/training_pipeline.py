"""
Simplified Production Training Pipeline

Single model (Gradient Boosting) workflow:
Raw Data → Select Features → Clean & Validate → Split → Preprocess → Train → Evaluate → Save
"""

import json
import logging
import pickle
# Import custom modules
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config_manager import ConfigManager
from src.data_split.data_splitter import DataSplitter
from src.features_engineer.preprocessor import ProductionPreprocessor

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Simplified training pipeline for Gradient Boosting Regressor.

    Usage:
        pipeline = TrainingPipeline('config/config.yaml')
        pipeline.run()
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline with configuration."""
        # Load config
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        self.splitter = DataSplitter(
            test_size=self.config.data.test_size,
            val_size=self.config.data.val_size,
            random_state=self.config.data.random_state,
            verbose=True,
        )

        self.preprocessor = ProductionPreprocessor(
            scaling_method=self.config.preprocessing.scaling_method,
            encoding_method=self.config.preprocessing.encoding_method,
            verbose=True,
        )

        # Model
        self.model = GradientBoostingRegressor(
            **self.config.model.hyperparameters,
            random_state=self.config.model.random_state,
        )

        # Data storage
        self.df_raw = None
        self.df_selected = None

        logger.info("Pipeline initialized")

    def load_data(self) -> pd.DataFrame:
        """Step 1: Load raw data."""
        logger.info(f"Loading data from {self.config.data.raw_data_path}")

        data_path = Path(self.config.data.raw_data_path)

        if data_path.suffix == ".csv":
            self.df_raw = pd.read_csv(data_path)
        elif data_path.suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(data_path, "r") as zip_ref:
                zip_ref.extractall("data/extracted")
            csv_files = list(Path("data/extracted").glob("*.csv"))
            self.df_raw = pd.read_csv(csv_files[0])
        else:
            raise ValueError(f"Unsupported format: {data_path.suffix}")

        logger.info(f"Loaded data: {self.df_raw.shape}")
        return self.df_raw

    def select_features(self) -> pd.DataFrame:
        """Step 2: Select features from config."""
        logger.info("Selecting features from config...")

        features_config = self.config.__dict__.get("features", {})

        # Get feature lists
        numeric = features_config.numeric
        categorical = features_config.categorical

        # Combine features + target
        selected = numeric + categorical
        target_col = self.config.data.target_column
        if target_col not in selected:
            selected.append(target_col)

        # Check availability
        available = [f for f in selected if f in self.df_raw.columns]
        missing = set(selected) - set(available)

        if missing:
            logger.warning(f"Features in config but missing in data: {missing}")

        self.df_selected = self.df_raw[available].copy()

        logger.info(
            f"Selected {len(available)} features "
            f"({len(numeric)} numeric, {len(categorical)} categorical)"
        )

        return self.df_selected

    def split_data(self) -> None:
        """Step 4: Split data."""
        logger.info("Splitting data...")

        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = (
            self.splitter.split_dataframe(
                self.df_selected, target_col=self.config.data.target_column
            )
        )

        logger.info(
            f"Train: {len(self.X_train)}, "
            f"Val: {len(self.X_val)}, "
            f"Test: {len(self.X_test)}"
        )

    def preprocess_data(self) -> None:
        """Step 5: Preprocess (fit on train, transform all)."""
        logger.info("Preprocessing...")

        # Fit on train only
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)

        # Transform val and test
        self.X_val_transformed = self.preprocessor.transform(self.X_val)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)

        logger.info("Preprocessing complete")

    def train_model(self) -> None:
        """Step 6: Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting Regressor...")

        self.model.fit(self.X_train_transformed, self.y_train)

        logger.info("✓ Model trained")

    def evaluate_model(self) -> dict:
        """Step 7: Evaluate on test and validation sets."""
        logger.info("Evaluating model...")

        # Test set
        y_test_pred = self.model.predict(self.X_test_transformed)
        test_metrics = {
            "r2": r2_score(self.y_test, y_test_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            "mae": mean_absolute_error(self.y_test, y_test_pred),
            "mse": mean_squared_error(self.y_test, y_test_pred),
        }

        # Validation set
        y_val_pred = self.model.predict(self.X_val_transformed)
        val_metrics = {
            "r2": r2_score(self.y_val, y_val_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            "mae": mean_absolute_error(self.y_val, y_val_pred),
            "mse": mean_squared_error(self.y_val, y_val_pred),
        }

        logger.info(
            f"Test  - R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}"
        )
        logger.info(
            f"Val   - R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}"
        )

        return {"test": test_metrics, "validation": val_metrics}

    def save_artifacts(self, output_dir: str = "models/production") -> None:
        """Step 8: Save model and artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving artifacts to {output_dir}...")

        # Save model
        with open(output_path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Save preprocessor
        self.preprocessor.save(str(output_path / "preprocessor.pkl"))

        # Save config
        self.config_manager.save_config(str(output_path / "config.yaml"))

        # Save metadata
        metadata = {
            "model_type": "GradientBoostingRegressor",
            "hyperparameters": self.config.model.hyperparameters,
            "test_metrics": self.metrics["test"],
            "val_metrics": self.metrics["validation"],
            "feature_names": self.preprocessor.get_feature_names(),
            "train_size": len(self.X_train),
            "val_size": len(self.X_val),
            "test_size": len(self.X_test),
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("✓ All artifacts saved")

    def run(self) -> dict:
        """Run training with MLflow tracking"""
        # Start MLflow run
        with mlflow.start_run(run_name=self.config.training.run_name):

            # Log config as parameters
            mlflow.log_params(
                {
                    "test_size": self.config.data.test_size,
                    "val_size": self.config.data.val_size,
                    "random_state": self.config.data.random_state,
                    "scaling_method": self.config.preprocessing.scaling_method,
                    **self.config.model.hyperparameters,  # All hyperparameters
                }
            )
            # Set tags
            mlflow.set_tag("model_type", "GradientBoostingRegressor")
            mlflow.set_tag("project", "house_price_prediction")
            # Execute pipeline steps
            self.load_data()
            self.select_features()
            self.split_data()
            self.preprocess_data()
            self.train_model()
            self.metrics = self.evaluate_model()

            # Log metrics
            mlflow.log_metrics(
                {
                    "test_r2": self.metrics["test"]["r2"],
                    "test_rmse": self.metrics["test"]["rmse"],
                    "test_mae": self.metrics["test"]["mae"],
                    "val_r2": self.metrics["validation"]["r2"],
                    "val_rmse": self.metrics["validation"]["rmse"],
                }
            )

            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                signature=mlflow.models.infer_signature(
                    self.X_train_transformed, self.y_train
                ),
            )

            # Log artifacts
            mlflow.log_artifact(self.config_path, "config.yaml")

            # Log feature names
            with open("feature_names.json", "w") as f:
                json.dump(self.preprocessor.get_feature_names(), f)
            mlflow.log_artifact("feature_names.json")

            self.save_artifacts()

            return self.metrics


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run pipeline
    pipeline = TrainingPipeline("config/config.yaml")
    metrics = pipeline.run()

    print(f"\n✓ Training complete!")
    print(f"  Test R²: {metrics['test']['r2']:.4f}")
    print(f"  Test RMSE: {metrics['test']['rmse']:.4f}")
