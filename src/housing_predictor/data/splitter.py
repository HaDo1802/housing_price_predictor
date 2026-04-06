"""Data splitting helpers."""

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """Handle train, test, and validation splitting."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose

        self._validate_split_sizes()

    def _validate_split_sizes(self) -> None:
        """Validate configured split sizes."""
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")

        if not 0 <= self.val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {self.val_size}")

        if self.test_size + self.val_size >= 1:
            raise ValueError(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) must be < 1"
            )

    def split(self, X: pd.DataFrame, y: pd.Series, return_val: bool = True) -> Tuple:
        """Split features and target into train, test, and optional validation sets."""
        # First split: Train+Val vs Test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                f"Initial split - Train+Val: {len(X_temp)}, Test: {len(X_test)}"
            )

        if not return_val:
            if self.verbose:
                logger.info(f"Final split - Train: {len(X_temp)}, Test: {len(X_test)}")
            return X_temp, X_test, y_temp, y_test

        # Second split: Train vs Val (from the Train+Val set)
        # Calculate validation size relative to temp set
        val_size_adjusted = self.val_size / (1 - self.test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )

        if self.verbose:
            logger.info(
                f"Final split - Train: {len(X_train)}, "
                f"Val: {len(X_val)}, Test: {len(X_test)}"
            )
            logger.info(
                f"Split proportions - Train: {len(X_train)/len(X):.2%}, "
                f"Val: {len(X_val)/len(X):.2%}, Test: {len(X_test)/len(X):.2%}"
            )

        return X_train, X_test, X_val, y_train, y_test, y_val

    def split_dataframe(
        self, df: pd.DataFrame, target_col: str, return_val: bool = True
    ) -> Tuple:
        """Split a dataframe by separating the target column first."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        return self.split(X, y, return_val=return_val)
