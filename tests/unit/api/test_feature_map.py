from housing_predictor.features.training_schema import MODEL_FEATURES
from serving.api.feature_map import (
    API_TO_MODEL_FIELDS,
    CATEGORICAL_OPTIONS,
    FEATURE_DISPLAY_LABELS,
)


def test_api_field_map_targets_valid_training_features():
    """API aliases may evolve, but they must always resolve to the training contract."""
    assert set(API_TO_MODEL_FIELDS.values()).issubset(set(MODEL_FEATURES))
    assert API_TO_MODEL_FIELDS["livingarea"] == "living_area"
    assert API_TO_MODEL_FIELDS["propertytype"] == "property_type"


def test_api_display_metadata_matches_real_features():
    """UI metadata should never drift away from the model's raw input schema."""
    assert set(FEATURE_DISPLAY_LABELS).issubset(set(MODEL_FEATURES))
    assert set(CATEGORICAL_OPTIONS).issubset(set(MODEL_FEATURES))
