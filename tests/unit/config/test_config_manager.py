from housing_predictor.config_manager import ConfigManager
from housing_predictor.features.training_schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def test_config_manager_loads_single_yaml_contract():
    """Config is a versioned runtime contract, so loading must stay explicit."""
    cfg = ConfigManager("conf/config.yaml").get_config()

    assert cfg.data.raw_data_path
    assert cfg.data.target_column == "price"
    assert cfg.features.numeric == NUMERIC_FEATURES
    assert cfg.features.categorical == CATEGORICAL_FEATURES
    assert cfg.model.model_type
    assert cfg.training.registry_model_name
