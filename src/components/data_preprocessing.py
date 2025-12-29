"""
Data Preprocessing Module

This module handles all data preprocessing steps including:
- Data loading and cleaning
- Missing value handling
- Outlier detection and removal
- Feature engineering (scaling, encoding, transformations)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline that handles cleaning,
    standardization, encoding, and feature engineering.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the DataPreprocessor.

        Parameters:
        verbose (bool): Whether to log processing steps
        """
        self.verbose = verbose
        self.scaler = None
        self.encoder = None
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.

        Parameters:
        file_path (str): Path to the data file

        Returns:
        pd.DataFrame: Loaded dataframe
        """
        import zipfile
        import os

        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall("extracted_data")
            extracted_files = os.listdir("extracted_data")
            csv_files = [f for f in extracted_files if f.endswith(".csv")]

            if len(csv_files) != 1:
                raise FileNotFoundError(
                    f"Expected 1 CSV file in zip, found {len(csv_files)}"
                )

            csv_file_path = os.path.join("extracted_data", csv_files[0])
            df = pd.read_csv(csv_file_path)
        else:
            df = pd.read_csv(file_path)

        if self.verbose:
            logger.info(f"Loaded data with shape: {df.shape}")

        return df

    def handle_missing_values(
        self, df: pd.DataFrame, method: str = "mean"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): Input dataframe
        method (str): 'mean', 'median', 'mode', or 'drop'

        Returns:
        pd.DataFrame: Dataframe with missing values handled
        """
        df_cleaned = df.copy()

        missing_count = df_cleaned.isnull().sum().sum()
        if missing_count == 0:
            if self.verbose:
                logger.info("No missing values found")
            return df_cleaned

        if method == "drop":
            df_cleaned = df_cleaned.dropna()
            if self.verbose:
                logger.info(
                    f"Dropped rows with missing values. New shape: {df_cleaned.shape}"
                )

        elif method == "mean":
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
                df_cleaned[numeric_cols].mean()
            )
            if self.verbose:
                logger.info("Filled numeric missing values with mean")

        elif method == "median":
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
                df_cleaned[numeric_cols].median()
            )
            if self.verbose:
                logger.info("Filled numeric missing values with median")

        elif method == "mode":
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().any():
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            if self.verbose:
                logger.info("Filled missing values with mode")

        return df_cleaned

    def detect_and_handle_outliers(
        self, df: pd.DataFrame, method: str = "iqr", remove: bool = True
    ) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR or Z-score method.

        Parameters:
        df (pd.DataFrame): Input dataframe
        method (str): 'iqr' or 'zscore'
        remove (bool): Whether to remove outliers or cap them

        Returns:
        pd.DataFrame: Dataframe with outliers handled
        """
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

        if method == "iqr":
            for col in numeric_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                if remove:
                    df_cleaned = df_cleaned[
                        (df_cleaned[col] >= lower_bound)
                        & (df_cleaned[col] <= upper_bound)
                    ]
                else:
                    df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)

        elif method == "zscore":
            for col in numeric_cols:
                z_scores = np.abs(
                    (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()
                )
                if remove:
                    df_cleaned = df_cleaned[z_scores <= 3]
                else:
                    df_cleaned[col] = df_cleaned[col].clip(
                        df_cleaned[col].mean() - 3 * df_cleaned[col].std(),
                        df_cleaned[col].mean() + 3 * df_cleaned[col].std(),
                    )

        if self.verbose:
            logger.info(f"Outlier handling completed. New shape: {df_cleaned.shape}")

        return df_cleaned

    def identify_feature_types(self, df: pd.DataFrame, target_col: str = None) -> None:
        """
        Identify numeric and categorical features.

        Parameters:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column to exclude
        """
        if target_col:
            df = df.drop(columns=[target_col])

        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        if self.verbose:
            logger.info(f"Numeric features: {len(self.numeric_features)}")
            logger.info(f"Categorical features: {len(self.categorical_features)}")

    def apply_log_transformation(
        self, df: pd.DataFrame, features: list = None
    ) -> pd.DataFrame:
        """
        Apply log transformation to specified features.

        Parameters:
        df (pd.DataFrame): Input dataframe
        features (list): List of feature names to transform

        Returns:
        pd.DataFrame: Dataframe with log-transformed features
        """
        df_transformed = df.copy()

        if features is None:
            features = self.numeric_features

        for feature in features:
            if feature in df_transformed.columns and df_transformed[feature].min() >= 0:
                df_transformed[feature] = np.log1p(df_transformed[feature])

        if self.verbose:
            logger.info(f"Applied log transformation to {len(features)} features")

        return df_transformed

    def create_preprocessor(
        self, scale_numeric: bool = True, encode_categorical: bool = True
    ) -> ColumnTransformer:
        """
        Create a scikit-learn ColumnTransformer for feature preprocessing.

        Parameters:
        scale_numeric (bool): Whether to scale numeric features
        encode_categorical (bool): Whether to encode categorical features

        Returns:
        ColumnTransformer: Fitted preprocessor
        """
        transformers = []

        if scale_numeric and self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))

        if encode_categorical and self.categorical_features:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    self.categorical_features,
                )
            )

        if not transformers:
            raise ValueError("No features to transform")

        self.preprocessor = ColumnTransformer(transformers=transformers)

        if self.verbose:
            logger.info("Created preprocessing pipeline")

        return self.preprocessor

    def fit_preprocessor(self, X: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.

        Parameters:
        X (pd.DataFrame): Training features

        Returns:
        DataPreprocessor: Self for method chaining
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not created. Call create_preprocessor first."
            )

        self.preprocessor.fit(X)
        if self.verbose:
            logger.info("Preprocessor fitted on training data")

        return self

    def transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.

        Parameters:
        X (pd.DataFrame): Features to transform

        Returns:
        np.ndarray: Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_preprocessor first.")

        return self.preprocessor.transform(X)

    def preprocess(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        handle_missing_method: str = "mean",
        handle_outliers: bool = True,
        scale_numeric: bool = True,
        encode_categorical: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline.

        Parameters:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        test_size (float): Test set fraction
        random_state (int): Random seed
        handle_missing_method (str): Method for missing value handling
        handle_outliers (bool): Whether to handle outliers
        scale_numeric (bool): Whether to scale numeric features
        encode_categorical (bool): Whether to encode categorical features

        Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_train_df, X_test_df)
        """
        # Handle missing values
        df = self.handle_missing_values(df, method=handle_missing_method)

        # Handle outliers
        if handle_outliers:
            df = self.detect_and_handle_outliers(df, method="iqr", remove=True)

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Identify feature types
        self.identify_feature_types(X)

        # Create and fit preprocessor
        self.create_preprocessor(
            scale_numeric=scale_numeric, encode_categorical=encode_categorical
        )
        self.fit_preprocessor(X)

        # Transform features
        X_transformed = self.transform_data(X)

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=test_size, random_state=random_state
        )

        if self.verbose:
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")

        return X_train, X_test, y_train, y_test, X, y

    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk.

        Parameters:
        filepath (str): Path to save the preprocessor
        """
        import pickle

        if self.preprocessor is None:
            raise ValueError("No preprocessor to save")

        with open(filepath, "wb") as f:
            pickle.dump(self.preprocessor, f)

        if self.verbose:
            logger.info(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath: str) -> None:
        """
        Load a fitted preprocessor from disk.

        Parameters:
        filepath (str): Path to load the preprocessor
        """
        import pickle

        with open(filepath, "rb") as f:
            self.preprocessor = pickle.load(f)

        if self.verbose:
            logger.info(f"Preprocessor loaded from {filepath}")
