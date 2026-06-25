"""State schema for the ExperimentOS LangGraph workflow.

The graph is a deterministic state machine: every node reads from and writes to
this shared ``ExperimentState`` dict. Two fields (``results`` and
``improvement_log``) use an ``operator.add`` reducer so that nodes *append* to
them across rounds instead of overwriting.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

import pandas as pd


class RunResult(TypedDict):
    """A single trained-model result, logged once to MLflow."""

    model_name: str
    round: int  # 0 = baseline, 1/2 = improvement rounds
    stage_label: str  # "baseline" | "class_weight_balanced" | "robust_scaler"
    metrics: dict[str, float]
    mlflow_run_id: str


class ExperimentState(TypedDict):
    """Shared state passed between every node in the workflow."""

    raw_df: pd.DataFrame
    target_column: str
    task_type: Literal["classification", "regression"]
    numeric_cols: list[str]
    categorical_cols: list[str]
    imbalance_detected: bool
    imbalance_ratio: Optional[float]
    skew_detected: bool
    skewed_cols: list[str]
    round_count: int  # 0, 1, 2 — controls the improvement loop
    imbalance_addressed: bool
    skew_addressed: bool
    # Appended (never overwritten) thanks to the operator.add reducer.
    results: Annotated[list[RunResult], operator.add]
    leaderboard: Optional[pd.DataFrame]
    improvement_log: Annotated[list[str], operator.add]
    mlflow_experiment_id: str
    report_markdown: Optional[str]
    # Internal scaler choice — flips to "robust" during the skew-fix round.
    scaler_type: str
    # Surfaced to the UI when a stratified split was not possible.
    split_warning: Optional[str]
