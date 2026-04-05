"""Configuration management helpers for training and serving."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    raw_data_path: str
    target_column: str
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class PreprocessingConfig:
    handle_missing: str = "mean"
    handle_outliers: bool = True
    outlier_method: str = "iqr"
    outlier_iqr_multiplier: float = 1.5
    outlier_ppsf_min: float = 80.0
    outlier_ppsf_max: float = 1000.0
    outlier_min_livingarea: float = 100.0
    exclude_property_types: list[str] = field(default_factory=list)
    scale_features: bool = True
    scaling_method: str = "standard"
    encode_categorical: bool = True
    encoding_method: str = "onehot"
    target_transform: str = "log1p"
    interval_num_segments: int = 5
    interval_min_segment_size: int = 10


@dataclass
class ModelConfig:
    model_type: str = "random_forest"
    random_state: int = 42
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    cv_folds: int = 5
    optimize_metric: str = "r2"


@dataclass
class TrainingConfig:
    experiment_name: str = "house_price_prediction"
    run_name: str = "run_001"
    registry_model_name: str = "housing_price_predictor"
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
    data: DataConfig
    features: FeatureSelectionConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig


class ConfigManager:
    """Loads a single YAML config file."""

    def __init__(self, config_path: str = "conf/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> MLConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        return MLConfig(
            data=DataConfig(**config_dict["data"]),
            features=FeatureSelectionConfig(**config_dict.get("features", {})),
            preprocessing=PreprocessingConfig(**config_dict["preprocessing"]),
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
        )

    def get_config(self) -> MLConfig:
        return self.config

    def save_config(self, output_path: str) -> None:
        config_dict = {
            "data": self.config.data.__dict__,
            "features": self.config.features.__dict__,
            "preprocessing": self.config.preprocessing.__dict__,
            "model": self.config.model.__dict__,
            "training": self.config.training.__dict__,
        }
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
