"""Training utility functions."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_INTERVAL_SEGMENTS = 5
DEFAULT_INTERVAL_MIN_SEGMENT_SIZE = 10


def regression_metrics(y_true, y_pred) -> dict:
    """Return common regression metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }


def evaluate_predictions(y_true, y_pred) -> tuple[dict, dict]:
    """Return evaluation metrics and segmented relative-error intervals."""
    test_pred = np.asarray(y_pred, dtype=float)
    y_test = np.asarray(y_true, dtype=float)

    alpha = 0.05
    coverage = float(1 - alpha)
    pred_nonneg = np.maximum(test_pred, 0.0)
    rel_err = np.abs(y_test - pred_nonneg) / np.maximum(np.abs(y_test), 1.0)
    global_q = float(np.quantile(rel_err, 1 - alpha, method="higher"))

    num_segments = DEFAULT_INTERVAL_SEGMENTS
    min_seg_size = DEFAULT_INTERVAL_MIN_SEGMENT_SIZE
    probs = np.linspace(0, 1, num_segments + 1)[1:-1]
    edges = np.quantile(pred_nonneg, probs).astype(float).tolist() if len(probs) else []
    edges = np.maximum.accumulate(np.asarray(edges, dtype=float)).tolist()

    q_by_segment = []
    for seg_idx in range(num_segments):
        if seg_idx == 0:
            mask = (
                pred_nonneg < edges[0]
                if edges
                else np.ones(len(pred_nonneg), dtype=bool)
            )
        elif seg_idx == num_segments - 1:
            mask = (
                pred_nonneg >= edges[-1]
                if edges
                else np.ones(len(pred_nonneg), dtype=bool)
            )
        else:
            mask = (pred_nonneg >= edges[seg_idx - 1]) & (pred_nonneg < edges[seg_idx])
        seg_err = rel_err[mask]
        q_by_segment.append(
            float(np.quantile(seg_err, 1 - alpha, method="higher"))
            if seg_err.size >= min_seg_size
            else global_q
        )

    prediction_interval = {
        "method": "segmented_relative_error_quantile",
        "alpha": alpha,
        "coverage": coverage,
        "num_segments": num_segments,
        "segment_edges": edges,
        "relative_error_quantiles_by_segment": q_by_segment,
        "relative_error_quantiles": {"global": global_q},
        "calibration_size": int(len(rel_err)),
    }
    metrics = {
        "test": regression_metrics(y_test, test_pred),
        "validation": None,
    }

    cheap_cutoff = float(np.percentile(y_test, 25))
    cheap_mask = y_test <= cheap_cutoff
    if np.any(cheap_mask):
        metrics["test"]["cheap_segment_mae"] = float(
            np.mean(np.abs(y_test[cheap_mask] - test_pred[cheap_mask]))
        )
        metrics["test"]["cheap_segment_cutoff_p25"] = cheap_cutoff

    return metrics, prediction_interval
