"""
Production Inference Pipeline (MLflow Registry based)

Loads model artifacts from MLflow Model Registry instead of local files.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Production inference pipeline.

    Usage:
        pipeline = InferencePipeline(model_name="housing_price_predictor", stage="Production")
        predictions = pipeline.predict(new_data_df)
    """

    def __init__(
        self,
        model_name: str = "housing_price_predictor",
        stage: str = "Production",
        version: Optional[Union[int, str]] = None,
    ):
        """
        Initialize inference pipeline from MLflow Model Registry.

        Args:
            model_name: Registered model name in MLflow
            stage: Model stage to load when version is not provided
            version: Explicit model version to load (overrides stage)
        """
        self.model_name = model_name
        self.stage = stage
        self.version = str(version) if version is not None else None
        self.client = MlflowClient()

        self.model_version_info = self._resolve_model_version()
        self.model_uri = self._build_model_uri()

        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()
        self.metadata = self._load_metadata()

        logger.info(
            "Inference pipeline initialized from registry: name=%s version=%s stage=%s",
            self.model_name,
            self.model_version_info.version,
            self.model_version_info.current_stage,
        )

    def _build_model_uri(self) -> str:
        if self.version is not None:
            return f"models:/{self.model_name}/{self.version}"
        return f"models:/{self.model_name}/{self.stage}"

    def _resolve_model_version(self):
        """Resolve target model version from registry."""
        if self.version is not None:
            return self.client.get_model_version(self.model_name, self.version)

        latest_versions = self.client.get_latest_versions(
            self.model_name, stages=[self.stage]
        )
        if not latest_versions:
            raise ValueError(
                f"No model found for name='{self.model_name}' at stage='{self.stage}'."
            )

        # get_latest_versions returns one per stage; keep highest version when duplicated
        latest_versions = sorted(
            latest_versions,
            key=lambda mv: int(mv.version),
            reverse=True,
        )
        return latest_versions[0]

    def _load_model(self):
        """Load sklearn model from MLflow registry URI."""
        model = mlflow.sklearn.load_model(self.model_uri)
        logger.info("Model loaded from %s", self.model_uri)
        return model

    def _download_run_artifact(self, artifact_name: str) -> Path:
        """Download an artifact from the model version run."""
        run_id = self.model_version_info.run_id
        artifact_uri = f"runs:/{run_id}/{artifact_name}"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        return Path(local_path)

    def _load_preprocessor(self):
        """Load fitted preprocessor from run artifacts."""
        preprocessor_path = self._download_run_artifact("preprocessor.pkl")
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info("Preprocessor loaded from run artifact: %s", preprocessor_path)
        return preprocessor

    def _load_metadata(self) -> Dict:
        """Load metadata from run artifacts."""
        metadata_path = self._download_run_artifact("metadata.json")
        with open(metadata_path, "r") as f:
            return json.load(f)

    def get_model_info(self) -> Dict:
        """Return loaded model version info and tracked metrics."""
        run = self.client.get_run(self.model_version_info.run_id)
        return {
            "model_name": self.model_name,
            "version": self.model_version_info.version,
            "stage": self.model_version_info.current_stage,
            "run_id": self.model_version_info.run_id,
            "source": self.model_version_info.source,
            "metrics": run.data.metrics,
            "tags": self.model_version_info.tags,
        }

    def list_available_models(self) -> List[Dict]:
        """List all registered versions and key metadata for this model."""
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        results = []
        for mv in sorted(versions, key=lambda item: int(item.version)):
            run = self.client.get_run(mv.run_id)
            results.append(
                {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "run_id": mv.run_id,
                    "status": mv.status,
                    "test_r2": run.data.metrics.get("test_r2"),
                    "val_r2": run.data.metrics.get("val_r2"),
                }
            )
        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        self._validate_input(X)
        X_transformed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_transformed)
        logger.info("Made predictions for %s samples", len(predictions))
        return predictions

    def predict_single(self, features: Dict[str, Union[float, str]]) -> float:
        """Make prediction for a single sample."""
        df = pd.DataFrame([features])
        prediction = self.predict(df)
        return prediction[0]

    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple:
        """Predict and derive quantile interval from ensemble estimators when available."""
        preds = self.predict(X)
        X_t = self.preprocessor.transform(X)

        if hasattr(self.model, "estimators_"):
            est = self.model.estimators_
            if isinstance(est, list):
                trees = est
            else:
                trees = np.array(est).ravel().tolist()

            tree_preds = np.array([t.predict(X_t) for t in trees])
            lower = np.percentile(tree_preds, 2.5, axis=0)
            upper = np.percentile(tree_preds, 97.5, axis=0)
            return preds, lower, upper

        logger.warning("Model does not support uncertainty estimation")
        return preds, preds, preds

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data has required features."""
        if hasattr(self.preprocessor, "numeric_features") and hasattr(
            self.preprocessor, "categorical_features"
        ):
            expected_features = set(
                self.preprocessor.numeric_features + self.preprocessor.categorical_features
            )
        else:
            expected_features = set(self.metadata.get("feature_names", []))

        actual_features = set(X.columns)
        missing_features = expected_features - actual_features

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Expected features: {expected_features}"
            )

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance from the model."""
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not support feature importance")

        feature_names = self.metadata["feature_names"]
        importances = self.model.feature_importances_

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        importance_df = importance_df.sort_values("importance", ascending=False)
        return importance_df.head(top_n)

    def explain_prediction(
        self, features: Dict[str, Union[float, str]], top_n: int = 5
    ) -> Dict:
        """Explain prediction with top features."""
        prediction = self.predict_single(features)

        if hasattr(self.model, "feature_importances_"):
            importance_df = self.get_feature_importance(top_n)
            return {
                "prediction": float(prediction),
                "top_features": importance_df.to_dict("records"),
            }

        return {
            "prediction": float(prediction),
            "message": "Model does not support feature importance",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = InferencePipeline(model_name="housing_price_predictor", stage="Production")

    print("Loaded model info:")
    print(pipeline.get_model_info())
    print("\nAvailable versions:")
    print(pipeline.list_available_models())
