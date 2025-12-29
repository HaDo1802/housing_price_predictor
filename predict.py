"""
Prediction script using the trained model

This script loads the best trained model and preprocessor to make predictions
on new data.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making predictions using a trained model."""

    def __init__(
        self,
        model_path: str = "models/best_model.pkl",
        preprocessor_path: str = "models/preprocessor.pkl",
    ):
        """
        Initialize prediction service.

        Parameters:
        model_path (str): Path to the trained model
        preprocessor_path (str): Path to the fitted preprocessor
        """
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        logger.info("Prediction service initialized")

    @staticmethod
    def _load_model(path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model

    @staticmethod
    def _load_preprocessor(path: str):
        """Load preprocessor from disk."""
        with open(path, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters:
        data (pd.DataFrame): Features to predict on

        Returns:
        np.ndarray: Predictions
        """
        # Transform data using preprocessor
        X_transformed = self.preprocessor.transform(data)

        # Make predictions
        predictions = self.model.predict(X_transformed)

        logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions

    def predict_single(self, features: dict) -> float:
        """
        Make a prediction for a single sample.

        Parameters:
        features (dict): Feature dictionary

        Returns:
        float: Prediction
        """
        df = pd.DataFrame([features])
        prediction = self.predict(df)
        return prediction[0]


def main():
    """Main entry point for predictions."""
    # Example usage
    service = PredictionService()

    # Example input (modify based on your data)
    example_data = {
        "Order": [1],
        "PID": [526301100],
        "MS SubClass": [60],
        "Lot Frontage": [65.0],
        "Lot Area": [8450],
        "Overall Qual": [7],
        "Overall Cond": [5],
        "Year Built": [2003],
        "Year Remod/Add": [2003],
        "Mas Vnr Area": [196.0],
        "BsmtFin SF 1": [706],
        "BsmtFin SF 2": [0],
        "Bsmt Unf SF": [150],
        "Total Bsmt SF": [856],
        "1st Flr SF": [856],
        "2nd Flr SF": [854],
        "Low Qual Fin SF": [0],
        "Gr Liv Area": [1710],
        "Bsmt Full Bath": [1],
        "Bsmt Half Bath": [0],
        "Full Bath": [1],
        "Half Bath": [0],
        "Bedroom AbvGr": [3],
        "Kitchen AbvGr": [1],
        "TotRms AbvGrd": [7],
        "Fireplaces": [2],
        "Garage Yr Blt": [2003],
        "Garage Cars": [2],
        "Garage Area": [500.0],
        "Wood Deck SF": [210.0],
        "Open Porch SF": [0],
        "Enclosed Porch": [0],
        "3Ssn Porch": [0],
        "Screen Porch": [0],
        "Pool Area": [0],
        "Misc Val": [0],
        "Mo Sold": [5],
        "Yr Sold": [2010],
    }

    df = pd.DataFrame(example_data)
    prediction = service.predict(df)

    logger.info(f"Prediction: ${prediction[0]:.2f}")
    print(f"Predicted Sale Price: ${prediction[0]:,.2f}")


if __name__ == "__main__":
    main()
