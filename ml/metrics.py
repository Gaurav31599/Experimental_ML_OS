"""Metric computation for both task types.

Primary ranking metric: ``f1_weighted`` (classification, sort desc) and
``rmse`` (regression, sort asc).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRIC = {"classification": "f1_weighted", "regression": "rmse"}
# True = higher is better (sort descending).
PRIMARY_ASCENDING = {"classification": False, "regression": True}


def compute_metrics(
    task_type: str,
    y_true,
    y_pred,
    y_proba=None,
) -> dict[str, float]:
    """Compute the metric dict for the given task type."""
    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }
        # ROC AUC only for binary problems with probability estimates.
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                pass
        return metrics

    # regression
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
