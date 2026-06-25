"""Candidate model registry.

Returns fresh, unfitted estimator instances each call so a fitted model is never
reused across improvement rounds. XGBoost is optional — if the import fails we
fall back to scikit-learn's GradientBoosting estimators.
"""

from __future__ import annotations

from typing import Optional

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:  # pragma: no cover - depends on optional dependency
    HAS_XGB = False


# Estimators whose constructor accepts a ``class_weight`` argument.
_SUPPORTS_CLASS_WEIGHT = {"LogisticRegression", "RandomForest"}


def get_candidate_models(task_type: str, class_weight: Optional[str] = None) -> dict:
    """Return a dict of {name: unfitted estimator} for the task.

    ``class_weight`` (e.g. "balanced") is applied only to estimators whose
    constructor accepts it. XGBoost / GradientBoosting do not take
    ``class_weight`` — imbalance is instead handled via sample weighting at fit
    time (see ``graph/nodes.py``), so they are returned unchanged here.
    """
    if task_type == "classification":
        models: dict = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                random_state=42, eval_metric="logloss"
            )
        else:
            models["GradientBoosting"] = GradientBoostingClassifier(random_state=42)

        if class_weight is not None:
            for name in list(models):
                if name in _SUPPORTS_CLASS_WEIGHT:
                    models[name].set_params(class_weight=class_weight)
        return models

    # regression
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(random_state=42)
    else:
        models["GradientBoosting"] = GradientBoostingRegressor(random_state=42)
    return models
