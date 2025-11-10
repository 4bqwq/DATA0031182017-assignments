from __future__ import annotations

from typing import Dict

import numpy as np


def compute_rank_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute regression metrics (MSE, RMSE, MAE, MAPE) on rank predictions."""

    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    errors = pred - true
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / np.clip(np.abs(true), 1.0, None)) * 100.0)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
