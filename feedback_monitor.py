"""
Feedback monitoring and simple drift analysis.

Usage:
    python feedback_monitor.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.config_manager import ConfigManager
from src.monitoring.feedback_collector import load_feedback
from typing import Optional


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> Optional[float]:
    """Compute Population Stability Index for a numeric feature."""
    reference = reference.dropna()
    current = current.dropna()

    if reference.nunique() < 2 or current.nunique() < 2:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = reference.quantile(quantiles).values
    cut_points = np.unique(cut_points)

    if len(cut_points) < 3:
        return None

    ref_counts, _ = np.histogram(reference, bins=cut_points)
    cur_counts, _ = np.histogram(current, bins=cut_points)

    ref_perc = ref_counts / max(ref_counts.sum(), 1)
    cur_perc = cur_counts / max(cur_counts.sum(), 1)

    ref_perc = np.where(ref_perc == 0, 1e-6, ref_perc)
    cur_perc = np.where(cur_perc == 0, 1e-6, cur_perc)

    psi = np.sum((ref_perc - cur_perc) * np.log(ref_perc / cur_perc))
    return float(psi)


def evaluate_feedback(df: pd.DataFrame) -> dict:
    """Compute basic feedback metrics."""
    metrics = {}

    metrics["total_feedback"] = int(len(df))
    if "agree_with_prediction" in df.columns:
        metrics["agree_rate"] = float(df["agree_with_prediction"].mean())

    if {"suggested_min", "suggested_max", "predicted_price"}.issubset(df.columns):
        mask = df["suggested_min"].notna() & df["suggested_max"].notna()
        if mask.any():
            within = (
                (df.loc[mask, "predicted_price"] >= df.loc[mask, "suggested_min"])
                & (df.loc[mask, "predicted_price"] <= df.loc[mask, "suggested_max"])
            )
            metrics["pred_within_user_range_rate"] = float(within.mean())
            metrics["avg_user_range_width"] = float(
                (df.loc[mask, "suggested_max"] - df.loc[mask, "suggested_min"]).mean()
            )

    return metrics


def main() -> None:
    config = ConfigManager("config/config.yaml").get_config()
    feedback_df = load_feedback()

    if feedback_df.empty:
        print("No feedback data found.")
        return

    # Load reference dataset for drift
    raw_data_path = Path(config.data.raw_data_path)
    if not raw_data_path.exists():
        print(f"Reference data not found: {raw_data_path}")
        return

    reference_df = pd.read_csv(raw_data_path)

    # Compute feedback metrics
    metrics = evaluate_feedback(feedback_df)

    # Compute drift for numeric features
    drift = {}
    numeric_features = config.features.numeric
    for feature in numeric_features:
        if feature in reference_df.columns and feature in feedback_df.columns:
            psi = compute_psi(reference_df[feature], feedback_df[feature])
            if psi is not None:
                drift[feature] = psi

    report = {
        "feedback_metrics": metrics,
        "numeric_feature_psi": drift,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
