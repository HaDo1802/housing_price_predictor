"""Data loading utilities."""

from pathlib import Path

import pandas as pd


def load_dataframe(data_path: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame."""
    path = Path(data_path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path.suffix}")
