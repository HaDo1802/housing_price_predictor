from predictor.config import ConfigManager


def test_config_manager_loads_single_yaml_contract():
    """Config is a versioned runtime contract, so loading must stay explicit."""
    cfg = ConfigManager("conf/config.yaml").config

    assert cfg.data.target_column == "price"
    assert not hasattr(cfg, "features")
    assert cfg.preprocessing.exclude_property_types == ["MOBILE"]
    assert cfg.model.model_type
    assert cfg.training.registry_model_name
