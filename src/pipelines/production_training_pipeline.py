"""
Production-Grade Training Pipeline

This is the COMPLETE workflow that shows:
1. Where to split data (ANSWER TO YOUR QUESTION!)
2. How to prevent data leakage
3. How to make everything modular and reusable
4. How to prepare for CI/CD

WORKFLOW:
Raw Data → Clean → Split → Preprocess → Train → Evaluate → Save
          ↑              ↑           ↑
          No learning    Split here  Learn from train only!
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_split.data_splitter import DataSplitter
from src.features_engineer.production_preprocessor import ProductionPreprocessor, DataCleaner
from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles model training with multiple algorithms.
    
    Follows MLOps best practices:
    - Configuration-driven
    - Experiment tracking ready
    - Model versioning
    - Reproducible
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize model zoo based on config"""
        random_state = self.config.get('random_state', 42)
        
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=random_state),
            'lasso': Lasso(alpha=0.1, random_state=random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state
            ),
            'svr': SVR(kernel='rbf', C=100, epsilon=0.1)
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Training all models...")
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                logger.info(f"✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"✗ Error training {name}: {str(e)}")
    
    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of results for each model
        """
        logger.info("Evaluating all models...")
        
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                
                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred)
                }
                
                self.results[name] = metrics
                
                logger.info(
                    f"{name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}"
                )
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
        
        return self.results
    
    def select_best_model(
        self,
        metric: str = 'r2'
    ) -> Tuple[str, Any, Dict[str, float]]:
        """
        Select best model based on metric.
        
        Args:
            metric: Metric to optimize ('r2', 'rmse', 'mae', 'mse')
        
        Returns:
            (model_name, model, metrics)
        """
        if not self.results:
            raise ValueError("No results available. Evaluate models first.")
        
        if metric == 'r2':
            # Higher is better
            best_name = max(self.results, key=lambda x: self.results[x]['r2'])
        elif metric in ['rmse', 'mae', 'mse']:
            # Lower is better
            best_name = min(self.results, key=lambda x: self.results[x][metric])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        self.best_model_name = best_name
        self.best_model = self.trained_models[best_name]
        
        logger.info(f"Best model: {best_name} ({metric}={self.results[best_name][metric]:.4f})")
        
        return best_name, self.best_model, self.results[best_name]
    
    def save_best_model(self, filepath: str) -> None:
        """Save best model"""
        if self.best_model is None:
            raise ValueError("No best model selected")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        logger.info(f"Best model ({self.best_model_name}) saved to {filepath}")


class TrainingPipeline:
    """
    Complete production-grade training pipeline.
    
    This is the MAIN ORCHESTRATOR that shows you:
    1. WHERE to split data
    2. HOW to prevent data leakage
    3. HOW to make everything modular
    
    Usage:
        pipeline = TrainingPipeline('config/config.yaml')
        results = pipeline.run()
        
    The pipeline automatically:
    - Loads data
    - Cleans data (no learning)
    - Splits data (CRITICAL STEP)
    - Preprocesses (fits on train only)
    - Trains models
    - Evaluates models
    - Saves best model
    - Saves all artifacts
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize components
        self.data_splitter = DataSplitter(
            test_size=self.config.data.test_size,
            val_size=self.config.data.val_size,
            random_state=self.config.data.random_state,
            verbose=True
        )
        
        self.preprocessor = ProductionPreprocessor(
            scaling_method=self.config.preprocessing.scaling_method,
            encoding_method=self.config.preprocessing.encoding_method,
            verbose=True
        )
        
        self.trainer = ModelTrainer(config=self.config.model.__dict__)
        
        # Data storage
        self.df_raw = None
        self.df_clean = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.X_train_transformed = None
        self.X_test_transformed = None
        self.X_val_transformed = None
        
        logger.info("Training pipeline initialized")
    
    def load_data(self) -> pd.DataFrame:
        """
        Step 1: Load raw data.
        
        Returns:
            Raw dataframe
        """
        logger.info(f"Loading data from {self.config.data.raw_data_path}")
        
        # Handle different file formats
        data_path = Path(self.config.data.raw_data_path)
        
        if data_path.suffix == '.csv':
            self.df_raw = pd.read_csv(data_path)
        elif data_path.suffix == '.parquet':
            self.df_raw = pd.read_parquet(data_path)
        elif data_path.suffix == '.zip':
            # Extract and load
            import zipfile
            with zipfile.ZipFile(data_path, 'r') as zip_ref:
                zip_ref.extractall('data/extracted')
            
            # Find CSV in extracted data
            csv_files = list(Path('data/extracted').glob('*.csv'))
            if len(csv_files) != 1:
                raise ValueError(f"Expected 1 CSV file, found {len(csv_files)}")
            
            self.df_raw = pd.read_csv(csv_files[0])
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info(f"Loaded data: {self.df_raw.shape}")
        return self.df_raw
    
    def clean_data(self) -> pd.DataFrame:
        """
        Step 2: Clean data (operations that DON'T learn from data).
        
        These operations can happen BEFORE splitting:
        - Remove duplicates
        - Fix data types
        - Basic validation
        
        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning data...")
        
        cleaner = DataCleaner()
        
        # Remove duplicates
        self.df_clean = cleaner.remove_duplicates(self.df_raw)
        
        logger.info(f"Data cleaned: {self.df_clean.shape}")
        return self.df_clean
    
    def split_data(self) -> None:
        """
        Step 3: SPLIT DATA (ANSWER TO YOUR QUESTION!)
        
        This is where we split into train/val/test.
        
        WHY HERE?
        - After operations that don't learn (cleaning)
        - Before operations that learn (scaling, encoding, imputation)
        
        This prevents DATA LEAKAGE!
        """
        logger.info("=" * 80)
        logger.info("SPLITTING DATA - This is the critical step!")
        logger.info("=" * 80)
        
        # Split the data
        self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val = \
            self.data_splitter.split_dataframe(
                self.df_clean,
                target_col=self.config.data.target_column
            )
        
        # Get and log statistics
        stats = self.data_splitter.get_split_statistics(
            self.y_train, self.y_test, self.y_val
        )
        
        logger.info("\nSplit Statistics:")
        logger.info(f"Train: {stats['train']}")
        logger.info(f"Val: {stats['val']}")
        logger.info(f"Test: {stats['test']}")
        
        logger.info("=" * 80)
    
    def preprocess_data(self) -> None:
        """
        Step 4: Preprocess data (operations that LEARN from data).
        
        CRITICAL:
        - FIT on training data ONLY
        - TRANSFORM train, val, test using SAME fitted preprocessor
        
        This is how we prevent data leakage!
        """
        logger.info("Preprocessing data...")
        
        # Fit preprocessor on TRAINING data ONLY
        logger.info("Fitting preprocessor on TRAINING data only...")
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        
        # Transform validation and test data using FITTED preprocessor
        logger.info("Transforming validation data...")
        self.X_val_transformed = self.preprocessor.transform(self.X_val)
        
        logger.info("Transforming test data...")
        self.X_test_transformed = self.preprocessor.transform(self.X_test)
        
        logger.info(
            f"Preprocessing complete:\n"
            f"  Train: {self.X_train_transformed.shape}\n"
            f"  Val: {self.X_val_transformed.shape}\n"
            f"  Test: {self.X_test_transformed.shape}"
        )
    
    def train_models(self) -> None:
        """
        Step 5: Train all models.
        """
        logger.info("Training models...")
        self.trainer.train_all(self.X_train_transformed, self.y_train)
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Step 6: Evaluate models on TEST set.
        
        Returns:
            Evaluation results
        """
        logger.info("Evaluating models on TEST set...")
        results = self.trainer.evaluate_all(self.X_test_transformed, self.y_test)
        
        # Also evaluate on validation set
        logger.info("\nEvaluating on VALIDATION set...")
        for name, model in self.trainer.trained_models.items():
            y_pred = model.predict(self.X_val_transformed)
            val_r2 = r2_score(self.y_val, y_pred)
            logger.info(f"{name} validation R²: {val_r2:.4f}")
        
        return results
    
    def select_best_model(self) -> Tuple[str, Any, Dict[str, float]]:
        """
        Step 7: Select best model.
        
        Returns:
            (model_name, model, metrics)
        """
        return self.trainer.select_best_model(
            metric=self.config.model.optimize_metric
        )
    
    def save_artifacts(self, output_dir: str = "models/production") -> None:
        """
        Step 8: Save all artifacts for production.
        
        Saves:
        - Best model
        - Preprocessor
        - Configuration
        - Results
        - Metadata
        
        Args:
            output_dir: Directory to save artifacts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving artifacts to {output_dir}...")
        
        # Save best model
        model_path = output_path / "model.pkl"
        self.trainer.save_best_model(str(model_path))
        
        # Save preprocessor
        preprocessor_path = output_path / "preprocessor.pkl"
        self.preprocessor.save(str(preprocessor_path))
        
        # Save configuration
        config_path = output_path / "config.yaml"
        self.config_manager.save_config(str(config_path))
        
        # Save results
        results_path = output_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.trainer.results, f, indent=2)
        
        # Save metadata
        metadata = {
            'best_model': self.trainer.best_model_name,
            'best_metrics': self.trainer.results[self.trainer.best_model_name],
            'feature_names': self.preprocessor.get_feature_names(),
            'train_size': len(self.X_train),
            'val_size': len(self.X_val),
            'test_size': len(self.X_test)
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ All artifacts saved to {output_dir}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        This is the main entry point that executes all steps in order.
        
        Returns:
            Dictionary with training results
        """
        logger.info("=" * 80)
        logger.info("STARTING PRODUCTION TRAINING PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Clean data (no learning)
            self.clean_data()
            
            # Step 3: Split data (CRITICAL!)
            self.split_data()
            
            # Step 4: Preprocess data (fit on train, transform on all)
            self.preprocess_data()
            
            # Step 5: Train models
            self.train_models()
            
            # Step 6: Evaluate models
            results = self.evaluate_models()
            
            # Step 7: Select best model
            best_name, best_model, best_metrics = self.select_best_model()
            
            # Step 8: Save artifacts
            self.save_artifacts()
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Best Model: {best_name}")
            logger.info(f"Test R²: {best_metrics['r2']:.4f}")
            logger.info(f"Test RMSE: {best_metrics['rmse']:.4f}")
            logger.info("=" * 80)
            
            return {
                'best_model_name': best_name,
                'best_metrics': best_metrics,
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    pipeline = TrainingPipeline("config/config.yaml")
    results = pipeline.run()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Best Model: {results['best_model_name']}")
    print(f"Test R²: {results['best_metrics']['r2']:.4f}")
    print("=" * 80)
