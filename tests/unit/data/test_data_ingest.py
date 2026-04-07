import pytest

from predictor.config import ConfigManager
from predictor.data_ingest import DataIngestor
from predictor.schema import MODEL_FEATURES, NUMERIC_FEATURES


def test_clean_applies_business_rules_and_casts_numerics(raw_training_df):
    config = ConfigManager("conf/config.yaml").config
    ingestor = DataIngestor(config)

    duplicated = raw_training_df.iloc[[0]].copy()
    df = raw_training_df.copy()
    df = df._append(duplicated, ignore_index=True)

    cleaned = ingestor.clean(df)

    assert len(cleaned) < len(df)
    assert "property_id" not in cleaned.columns
    assert "price_per_sqft" not in cleaned.columns
    assert "MOBILE" not in set(cleaned["property_type"])
    assert all(str(cleaned[col].dtype) == "float64" for col in NUMERIC_FEATURES)


def test_select_training_columns_keeps_exact_contract(raw_training_df):
    config = ConfigManager("conf/config.yaml").config
    ingestor = DataIngestor(config)

    cleaned = ingestor.clean(raw_training_df)
    selected = ingestor.select_training_columns(cleaned)

    assert list(selected.columns) == list(MODEL_FEATURES) + [config.data.target_column]


def test_select_training_columns_fails_loudly_when_required_feature_missing(raw_training_df):
    config = ConfigManager("conf/config.yaml").config
    ingestor = DataIngestor(config)

    cleaned = ingestor.clean(raw_training_df).drop(columns=["living_area"])

    with pytest.raises(ValueError, match="Missing training columns"):
        ingestor.select_training_columns(cleaned)
