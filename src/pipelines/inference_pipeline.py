"""
Production Inference Pipeline

This shows how to use trained models in production.

Key points:
1. Load SAVED artifacts (model + preprocessor)
2. Apply SAME preprocessing to new data
3. Make predictions
4. Handle edge cases
"""

import logging
import pickle
from pathlib import Path
from typing import Union, List, Dict
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Production inference pipeline.
    
    Usage:
        # Initialize
        pipeline = InferencePipeline('models/production')
        
        # Make predictions on new data
        predictions = pipeline.predict(new_data_df)
        
        # Make single prediction
        prediction = pipeline.predict_single(feature_dict)
    """
    
    def __init__(self, model_dir: str = "models/production"):
        """
        Initialize inference pipeline.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = Path(model_dir)
        self._validate_model_dir()
        
        # Load artifacts
        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()
        self.metadata = self._load_metadata()
        
        logger.info(f"Inference pipeline initialized with model: {self.metadata['best_model']}")
    
    def _validate_model_dir(self) -> None:
        """Validate that model directory contains required files"""
        required_files = ['model.pkl', 'preprocessor.pkl', 'metadata.json']
        
        for filename in required_files:
            filepath = self.model_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required file not found: {filepath}\n"
                    f"Model directory must contain: {required_files}"
                )
    
    def _load_model(self):
        """Load trained model"""
        model_path = self.model_dir / "model.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def _load_preprocessor(self):
        """Load fitted preprocessor"""
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        return preprocessor
    
    def _load_metadata(self) -> Dict:
        """Load model metadata"""
        metadata_path = self.model_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features dataframe (same format as training data)
        
        Returns:
            Array of predictions
        """
        # Validate input
        self._validate_input(X)
        
        # Preprocess using SAME preprocessor from training
        X_transformed = self.preprocessor.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_transformed)
        
        logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions
    
    def predict_single(self, features: Dict[str, Union[float, str]]) -> float:
        """
        Make prediction for a single sample.
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Single prediction
        """
        # Convert to dataframe
        df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.predict(df)
        
        return prediction[0]
    
    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        n_iterations: int = 100
    ) -> tuple:
        """
        Make predictions with uncertainty estimates.
        
        Only works with tree-based models that support this.
        
        Args:
            X: Features dataframe
            n_iterations: Number of iterations for uncertainty estimation
        
        Returns:
            (predictions, lower_bound, upper_bound)
        """
        predictions = self.predict(X)
        
        # For tree-based models, we can use prediction intervals
        if hasattr(self.model, 'estimators_'):
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(self.preprocessor.transform(X))
                for tree in self.model.estimators_
            ])
            
            # Calculate confidence intervals
            lower_bound = np.percentile(tree_predictions, 2.5, axis=0)
            upper_bound = np.percentile(tree_predictions, 97.5, axis=0)
            
            return predictions, lower_bound, upper_bound
        
        else:
            logger.warning("Model does not support uncertainty estimation")
            return predictions, predictions, predictions
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Args:
            X: Input dataframe
        """
        # Check for required features
        expected_features = set(
            self.preprocessor.numeric_features + 
            self.preprocessor.categorical_features
        )
        actual_features = set(X.columns)
        
        missing_features = expected_features - actual_features
        
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Expected features: {expected_features}"
            )
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importance")
        
        feature_names = self.metadata['feature_names']
        importances = self.model.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def explain_prediction(
        self,
        features: Dict[str, Union[float, str]],
        top_n: int = 5
    ) -> Dict:
        """
        Explain a single prediction (simple version).
        
        Args:
            features: Feature dictionary
            top_n: Number of top contributing features
        
        Returns:
            Dictionary with prediction and contributing features
        """
        # Make prediction
        prediction = self.predict_single(features)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = self.get_feature_importance(top_n)
            
            return {
                'prediction': float(prediction),
                'top_features': importance_df.to_dict('records')
            }
        
        else:
            return {
                'prediction': float(prediction),
                'message': 'Model does not support feature importance'
            }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize inference pipeline
    pipeline = InferencePipeline('models/production')
    
    # Example: Make prediction on new data
    new_data = pd.DataFrame({
        'numeric1': [0.5],
        'numeric2': [-0.2],
        'category1': ['A'],
        'category2': ['X']
    })
    
    prediction = pipeline.predict(new_data)
    print(f"Prediction: {prediction[0]:.2f}")
    
    # Example: Single prediction
    features = {
        'numeric1': 0.5,
        'numeric2': -0.2,
        'category1': 'A',
        'category2': 'X'
    }
    
    single_pred = pipeline.predict_single(features)
    print(f"Single prediction: {single_pred:.2f}")
    
    # Example: Get feature importance
    try:
        importance = pipeline.get_feature_importance(top_n=5)
        print("\nTop 5 Important Features:")
        print(importance)
    except ValueError as e:
        print(f"Feature importance not available: {e}")
