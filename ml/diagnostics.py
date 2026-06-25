"""Deterministic dataset diagnostics.

These heuristics drive the graph's routing decisions. The thresholds are fixed
on purpose — the improvement loop is deterministic, so detection must be too.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from scipy.stats import skew


def detect_task_type(df: pd.DataFrame, target_column: str) -> Literal["classification", "regression"]:
    """Classify the modelling task from the target column.

    - object / bool / category dtype  -> classification
    - numeric with <= 20 unique values, or < 5% unique ratio -> classification
      (covers integer-coded categories)
    - otherwise -> regression
    """
    target = df[target_column]
    if target.dtype == object or target.dtype == bool or str(target.dtype) == "category":
        return "classification"

    nunique = target.nunique()
    if nunique <= 20 or (nunique / len(df)) < 0.05:
        return "classification"
    return "regression"


def split_feature_columns(
    df: pd.DataFrame, target_column: str
) -> tuple[list[str], list[str]]:
    """Return ``(numeric_cols, categorical_cols)`` for the feature columns.

    The target column is always excluded from both lists.
    """
    feature_df = df.drop(columns=[target_column])
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def detect_imbalance(
    df: pd.DataFrame, target_column: str
) -> tuple[bool, float | None]:
    """Detect class imbalance for classification targets.

    Returns ``(imbalance_detected, imbalance_ratio)`` where the ratio is
    ``majority_count / minority_count``. Flags True when ratio > 3.0.
    """
    counts = df[target_column].value_counts()
    if len(counts) < 2:
        return False, None
    ratio = float(counts.max() / counts.min())
    return ratio > 3.0, ratio


def detect_skew(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[bool, list[str]]:
    """Detect skewed numeric feature columns.

    Returns ``(skew_detected, skewed_cols)``. A column is skewed when
    ``abs(scipy.stats.skew()) > 1.0``.
    """
    skewed_cols: list[str] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        if abs(float(skew(series))) > 1.0:
            skewed_cols.append(col)
    return len(skewed_cols) > 0, skewed_cols
