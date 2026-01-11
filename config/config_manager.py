"""
Configuration Management Module

Handles all configuration loading and validation.
Ensures consistent configuration across training and inference.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
@dataclass
class DataConfig:
    """Data-related configuration"""
    raw_data_path: str
    # processed_data_path: str
    # train_data_path: str
    # test_data_path: str
    target_column: str
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""
    handle_missing: str = "mean"  # Options: mean, median, mode, drop
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # Options: iqr, zscore
    scale_features: bool = True
    scaling_method: str = "standard"  # Options: standard, minmax, robust
    encode_categorical: bool = True
    encoding_method: str = "onehot"  # Options: onehot, label, target


@dataclass
class ModelConfig:
    """Model training configuration"""
    model_type: str = "random_forest"
    random_state: int = 42
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    cv_folds: int = 5
    optimize_metric: str = "r2"


@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    experiment_name: str = "house_price_prediction"
    run_name: str = "run_001"
    log_metrics: bool = True
    save_artifacts: bool = True
    track_experiments: bool = True
    model_output_dir: str = "models"
    log_dir: str = "logs"
@dataclass
class FeatureSelectionConfig:
    numeric: list = field(default_factory=list)
    categorical: list = field(default_factory=list)

@dataclass
class MLConfig:
    """Complete ML configuration"""
    data: DataConfig
    features: FeatureSelectionConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig


class ConfigManager:
    """
    Configuration manager that loads and validates configs.
    
    Usage:
        config_manager = ConfigManager("config/config.yaml")
        config = config_manager.get_config()
        
        # Access specific configs
        print(config.data.target_column)
        print(config.model.model_type)
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._validate_config_exists()
        self.config = self._load_config()
    
    def _validate_config_exists(self) -> None:
        """Validate that config file exists"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def _load_config(self) -> MLConfig:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return MLConfig(
            data=DataConfig(**config_dict['data']),
            features=FeatureSelectionConfig(**config_dict['features']),
            preprocessing=PreprocessingConfig(**config_dict['preprocessing']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training'])
        )
    
    def get_config(self) -> MLConfig:
        """Get complete configuration"""
        return self.config
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to file.
        Useful for versioning experiments.
        
        Args:
            output_path: Path to save configuration
        """
        config_dict = {
            'data': self.config.data.__dict__,
            'preprocessing': self.config.preprocessing.__dict__,
            'model': self.config.model.__dict__,
            'training': self.config.training.__dict__
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    # @staticmethod
    # def create_default_config(output_path: str = "config/default_config.yaml") -> None:
    #     """Create a default configuration file"""
    #     default_config = {
    #         'data': {
    #             'raw_data_path': 'data/raw/data.csv',
    #             'processed_data_path': 'data/processed/',
    #             'train_data_path': 'data/processed/train.csv',
    #             'test_data_path': 'data/processed/test.csv',
    #             'target_column': 'SalePrice',
    #             'test_size': 0.2,
    #             'val_size': 0.1,
    #             'random_state': 42
    #         },
    #         'preprocessing': {
    #             'handle_missing': 'mean',
    #             'handle_outliers': True,
    #             'outlier_method': 'iqr',
    #             'scale_features': True,
    #             'scaling_method': 'standard',
    #             'encode_categorical': True,
    #             'encoding_method': 'onehot'
    #         },
    #         'model': {
    #             'model_type': 'random_forest',
    #             'random_state': 42,
    #             'hyperparameters': {
    #                 'n_estimators': 100,
    #                 'max_depth': 10,
    #                 'min_samples_split': 2
    #             },
    #             'cv_folds': 5,
    #             'optimize_metric': 'r2'
    #         },
    #         'training': {
    #             'experiment_name': 'house_price_prediction',
    #             'run_name': 'run_001',
    #             'log_metrics': True,
    #             'save_artifacts': True,
    #             'track_experiments': True
    #         }
    #     }
        
    #     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    #     with open(output_path, 'w') as f:
    #         yaml.dump(default_config, f, default_flow_style=False)


if __name__ == "__main__":
    # Create default config
    # ConfigManager.create_default_config()
    
    # Load and use config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print(f"Target column: {config.data.target_column}")
    print(f"Feature selection: {config.features.numeric}")
    print(f"Model type: {config.model.model_type}")
    print(f"Test size: {config.data.test_size}")
