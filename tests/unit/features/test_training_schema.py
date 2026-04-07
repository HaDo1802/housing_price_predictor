from predictor.schema import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def test_training_schema_lists_are_consistent():
    """The training schema is the core contract between data, training, and serving."""
    assert MODEL_FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))
    assert TARGET_COLUMN not in MODEL_FEATURES


def test_drop_columns_do_not_overlap_model_features():
    """A feature cannot be both required for training and dropped before selection."""
    assert set(DROP_COLUMNS).isdisjoint(MODEL_FEATURES)
