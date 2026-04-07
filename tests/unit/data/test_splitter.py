from predictor.training_pipeline import train_test_split


def test_train_test_split_preserves_total_row_count(sample_housing_df):
    X = sample_housing_df.drop(columns=["price"])
    y = sample_housing_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) + len(X_test) == len(sample_housing_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
