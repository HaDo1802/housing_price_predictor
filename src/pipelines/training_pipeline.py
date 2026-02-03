"""
Simplified Production Training Pipeline

Single model (Gradient Boosting) workflow:
Raw Data -> Select Features -> Split -> Preprocess -> Train -> Evaluate -> Register
"""

import json
import logging
import pickle
import subprocess
import tempfile
import time

# Import custom modules
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
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

        self.model = GradientBoostingRegressor(
            **self.config.model.hyperparameters,
            random_state=self.config.model.random_state,
        )

        self.registry_model_name = getattr(
            self.config.training, "registry_model_name", "housing_price_predictor"
        )

        self.df_raw = None
        self.df_selected = None
        self.metrics = None

        logger.info("Pipeline initialized")

    def load_data(self) -> pd.DataFrame:
        """Step 1: Load raw data."""
        logger.info("Loading data from %s", self.config.data.raw_data_path)
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

        logger.info("Loaded data: %s", self.df_raw.shape)
        return self.df_raw

    def select_features(self) -> pd.DataFrame:
        """Step 2: Select features from config."""
        logger.info("Selecting features from config...")

        features_config = self.config.__dict__.get("features", {})
        numeric = features_config.numeric
        categorical = features_config.categorical

        selected = numeric + categorical
        target_col = self.config.data.target_column
        if target_col not in selected:
            selected.append(target_col)

        available = [f for f in selected if f in self.df_raw.columns]
        missing = set(selected) - set(available)
        if missing:
            logger.warning("Features in config but missing in data: %s", missing)

        self.df_selected = self.df_raw[available].copy()
        logger.info(
            "Selected %s features (%s numeric, %s categorical)",
            len(available),
            len(numeric),
            len(categorical),
        )
        return self.df_selected

    def split_data(self) -> None:
        """Step 3: Split data."""
        logger.info("Splitting data...")
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = (
            self.splitter.split_dataframe(
                self.df_selected, target_col=self.config.data.target_column
            )
        )
        logger.info(
            "Train: %s, Val: %s, Test: %s",
            len(self.X_train),
            len(self.X_val),
            len(self.X_test),
        )

    def preprocess_data(self) -> None:
        """Step 4: Preprocess (fit on train, transform all)."""
        logger.info("Preprocessing...")
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_val_transformed = self.preprocessor.transform(self.X_val)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)
        logger.info("Preprocessing complete")

    def train_model(self) -> None:
        """Step 5: Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting Regressor...")
        self.model.fit(self.X_train_transformed, self.y_train)
        logger.info("Model trained")

    def evaluate_model(self) -> dict:
        """Step 6: Evaluate on test and validation sets."""
        logger.info("Evaluating model...")

        y_test_pred = self.model.predict(self.X_test_transformed)
        test_metrics = {
            "r2": r2_score(self.y_test, y_test_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            "mae": mean_absolute_error(self.y_test, y_test_pred),
            "mse": mean_squared_error(self.y_test, y_test_pred),
        }

        y_val_pred = self.model.predict(self.X_val_transformed)
        val_metrics = {
            "r2": r2_score(self.y_val, y_val_pred),
            "rmse": np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            "mae": mean_absolute_error(self.y_val, y_val_pred),
            "mse": mean_squared_error(self.y_val, y_val_pred),
        }

        logger.info(
            "Test - R2: %.4f, RMSE: %.4f", test_metrics["r2"], test_metrics["rmse"]
        )
        logger.info(
            "Val  - R2: %.4f, RMSE: %.4f", val_metrics["r2"], val_metrics["rmse"]
        )
        return {"test": test_metrics, "validation": val_metrics}

    def _build_metadata(self) -> dict:
        """Build metadata payload used in local backup and MLflow artifacts."""
        return {
            "model_type": "GradientBoostingRegressor",
            "hyperparameters": self.config.model.hyperparameters,
            "test_metrics": self.metrics["test"],
            "val_metrics": self.metrics["validation"],
            "feature_names": self.preprocessor.get_feature_names(),
            "train_size": len(self.X_train),
            "val_size": len(self.X_val),
            "test_size": len(self.X_test),
        }

    def _get_git_commit(self) -> str:
        """Get current git commit hash if available."""
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return "unknown"

    def save_artifacts(self, output_dir: str = None) -> Path:
        """Save model artifacts to timestamped experiment backup directory."""
        if output_dir is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = Path("models/experiments") / timestamp
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving local backup artifacts to %s", output_path)

        with open(output_path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        self.preprocessor.save(str(output_path / "preprocessor.pkl"))
        self.config_manager.save_config(str(output_path / "config.yaml"))

        with open(output_path / "metadata.json", "w") as f:
            json.dump(self._build_metadata(), f, indent=2)

        return output_path

    def register_model(self, run_id: str) -> str:
        """Register model in MLflow Model Registry and apply required tags."""
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=self.registry_model_name)
        version = str(result.version)

        client = MlflowClient()
        status = client.get_model_version(self.registry_model_name, version).status
        attempts = 0
        while status == "PENDING_REGISTRATION" and attempts < 30:
            time.sleep(1)
            attempts += 1
            status = client.get_model_version(self.registry_model_name, version).status

        version_tags = {
            "training_date": datetime.now(timezone.utc).isoformat(),
            "git_commit": self._get_git_commit(),
            "model_type": "GradientBoostingRegressor",
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                name=self.registry_model_name,
                version=version,
                key=key,
                value=str(value),
            )

        logger.info(
            "Model registered successfully: %s v%s (status=%s)",
            self.registry_model_name,
            version,
            status,
        )
        print(f"Model registration successful: {self.registry_model_name} version={version}")
        return version

    def auto_promote_if_better(
        self,
        new_model_version: str,
        new_test_r2: float,
        improvement_threshold: float = 0.02,
    ) -> None:
        """
        Promote model if it is > 0.02 better test R2 than current Production model.
        If no Production exists, promote automatically.
        """
        client = MlflowClient()
        production_versions = client.get_latest_versions(
            self.registry_model_name, stages=["Production"]
        )

        if not production_versions:
            logger.info(
                "No Production model exists. Auto-promoting version %s.", new_model_version
            )
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Staging",
            )
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Production",
                archive_existing_versions=True,
            )
            return

        current_prod = production_versions[0]
        prod_run = client.get_run(current_prod.run_id)
        prod_r2 = prod_run.data.metrics.get("test_r2")

        if prod_r2 is None:
            logger.warning(
                "Current Production version %s has no test_r2. Keeping new version in Staging.",
                current_prod.version,
            )
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Staging",
            )
            return

        improvement = new_test_r2 - float(prod_r2)
        if improvement > improvement_threshold:
            logger.info(
                "Auto-promote YES: new version %s improves test_r2 by %.4f (new=%.4f, prod=%.4f).",
                new_model_version,
                improvement,
                new_test_r2,
                float(prod_r2),
            )
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Staging",
            )
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Production",
                archive_existing_versions=True,
            )
        else:
            logger.info(
                "Auto-promote NO: improvement %.4f <= threshold %.4f. New version %s stays in Staging.",
                improvement,
                improvement_threshold,
                new_model_version,
            )
            client.transition_model_version_stage(
                name=self.registry_model_name,
                version=new_model_version,
                stage="Staging",
            )

    def run(self) -> dict:
        """Run training, log to MLflow, register model, and manage lifecycle."""
        mlflow.set_experiment(self.config.training.experiment_name)

        with mlflow.start_run(run_name=self.config.training.run_name):
            git_commit = self._get_git_commit()

            mlflow.log_params(
                {
                    "test_size": self.config.data.test_size,
                    "val_size": self.config.data.val_size,
                    "random_state": self.config.data.random_state,
                    "scaling_method": self.config.preprocessing.scaling_method,
                    **self.config.model.hyperparameters,
                }
            )
            mlflow.set_tag("model_type", "GradientBoostingRegressor")
            mlflow.set_tag("project", "house_price_prediction")
            mlflow.set_tag("training_date", datetime.now(timezone.utc).isoformat())
            mlflow.set_tag("git_commit", git_commit)

            self.load_data()
            self.select_features()
            self.split_data()
            self.preprocess_data()
            self.train_model()
            self.metrics = self.evaluate_model()

            mlflow.log_metrics(
                {
                    "test_r2": self.metrics["test"]["r2"],
                    "test_rmse": self.metrics["test"]["rmse"],
                    "test_mae": self.metrics["test"]["mae"],
                    "val_r2": self.metrics["validation"]["r2"],
                    "val_rmse": self.metrics["validation"]["rmse"],
                }
            )

            mlflow.sklearn.log_model(
                self.model,
                artifact_path="model",
                signature=mlflow.models.infer_signature(
                    self.X_train_transformed, self.y_train
                ),
            )

            mlflow.log_artifact(self.config_path, artifact_path="config")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                self.preprocessor.save(str(tmp_path / "preprocessor.pkl"))
                with open(tmp_path / "metadata.json", "w") as f:
                    json.dump(self._build_metadata(), f, indent=2)
                with open(tmp_path / "feature_names.json", "w") as f:
                    json.dump(self.preprocessor.get_feature_names(), f)
                mlflow.log_artifacts(str(tmp_path))

            run_id = mlflow.active_run().info.run_id
            model_version = self.register_model(run_id)
            self.auto_promote_if_better(
                new_model_version=model_version,
                new_test_r2=self.metrics["test"]["r2"],
            )

            backup_path = self.save_artifacts()
            logger.info("Local experiment backup path: %s", backup_path)

            return self.metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    pipeline = TrainingPipeline("config/config.yaml")
    metrics = pipeline.run()

    print("\nTraining complete")
    print(f"Test R2: {metrics['test']['r2']:.4f}")
    print(f"Test RMSE: {metrics['test']['rmse']:.4f}")
