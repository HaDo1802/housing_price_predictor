import pytest

from predictor.preprocessor import ProductionPreprocessor


def test_preprocessor_fits_on_canonical_training_columns(sample_housing_df):
    """Protects the expected raw training feature contract."""
    X = sample_housing_df.drop(columns=["price"])

    preprocessor = ProductionPreprocessor(verbose=False)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert preprocessor.is_fitted is True
    assert preprocessor.numeric_features + preprocessor.categorical_features == list(
        X.columns
    )


def test_preprocessor_rejects_missing_required_feature(sample_housing_df):
    """Missing columns should fail early with a diagnostic error."""
    X = sample_housing_df.drop(columns=["price", "vegas_district"])

    preprocessor = ProductionPreprocessor(verbose=False)

    with pytest.raises(ValueError, match="vegas_district"):
        preprocessor.fit_transform(X)


def test_preprocessor_ignores_extra_columns_after_fit(sample_housing_df):
    """Serving and feature stores often attach extra columns; preprocessing should ignore them."""
    X = sample_housing_df.drop(columns=["price"])
    preprocessor = ProductionPreprocessor(verbose=False)
    preprocessor.fit_transform(X)

    X_extra = X.copy()
    X_extra["unexpected_debug_column"] = 123
    transformed = preprocessor.transform(X_extra)

    assert transformed.shape[0] == len(X_extra)
