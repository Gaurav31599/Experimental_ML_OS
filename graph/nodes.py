"""Node functions for the ExperimentOS workflow.

Every node is plain deterministic Python — no LLM, no randomness beyond the
fixed ``random_state=42``. Each node receives the full ``ExperimentState`` and
returns a partial dict of updates (LangGraph merges them, appending to the
reducer fields ``results`` and ``improvement_log``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from ml import diagnostics, metrics as metrics_mod
from ml.models import get_candidate_models
from ml.preprocessing import build_preprocessor
from mlflow_utils import log_run

from .state import ExperimentState, RunResult

RANDOM_STATE = 42
# Models that need imbalance handling via sample weights rather than class_weight.
_SAMPLE_WEIGHT_MODELS = {"XGBoost", "GradientBoosting"}


# --------------------------------------------------------------------------- #
# Pipeline nodes (run once, in order)
# --------------------------------------------------------------------------- #
def ingest_node(state: ExperimentState) -> dict:
    """Validate the dataframe and drop rows with a missing target."""
    df = state["raw_df"]
    target = state["target_column"]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    return {"raw_df": df}


def detect_task_type_node(state: ExperimentState) -> dict:
    """Decide classification vs regression."""
    task_type = diagnostics.detect_task_type(state["raw_df"], state["target_column"])
    return {"task_type": task_type}


def profile_data_node(state: ExperimentState) -> dict:
    """Detect feature column types, class imbalance, and numeric skew."""
    df = state["raw_df"]
    target = state["target_column"]
    numeric_cols, categorical_cols = diagnostics.split_feature_columns(df, target)

    imbalance_detected, imbalance_ratio = False, None
    if state["task_type"] == "classification":
        imbalance_detected, imbalance_ratio = diagnostics.detect_imbalance(df, target)

    skew_detected, skewed_cols = diagnostics.detect_skew(df, numeric_cols)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "imbalance_detected": imbalance_detected,
        "imbalance_ratio": imbalance_ratio,
        "skew_detected": skew_detected,
        "skewed_cols": skewed_cols,
    }


def build_preprocessor_node(state: ExperimentState) -> dict:
    """Record the scaler choice (preprocessor is rebuilt per train call)."""
    # The actual ColumnTransformer is rebuilt inside training so each pipeline
    # is fresh; here we just ensure scaler_type has a default value.
    return {"scaler_type": state.get("scaler_type") or "standard"}


def train_baseline_node(state: ExperimentState) -> dict:
    """Train all candidate models (round 0) and log them to MLflow."""
    results, warning = _train_candidates(
        state,
        round_idx=0,
        stage_label="baseline",
        class_weight=None,
        scaler=state["scaler_type"],
    )
    return {"results": results, "split_warning": warning}


def rank_node(state: ExperimentState) -> dict:
    """Rebuild the leaderboard dataframe from all results so far."""
    leaderboard = _build_leaderboard(state["results"], state["task_type"])
    return {"leaderboard": leaderboard}


def apply_improvement_node(state: ExperimentState) -> dict:
    """Apply one deterministic fix per call, then bump round_count.

    Priority: imbalance fix first, then skew fix. Exactly one fix is applied
    per visit so the loop terminates after at most 2 rounds.
    """
    round_idx = state["round_count"] + 1
    updates: dict = {"round_count": round_idx}

    needs_imbalance = state["imbalance_detected"] and not state["imbalance_addressed"]
    needs_skew = state["skew_detected"] and not state["skew_addressed"]

    if needs_imbalance:
        results, _ = _train_candidates(
            state,
            round_idx=round_idx,
            stage_label="class_weight_balanced",
            class_weight="balanced",
            scaler=state["scaler_type"],
        )
        ratio = state["imbalance_ratio"] or 0.0
        note = (
            f"Round {round_idx}: detected class imbalance "
            f"(ratio {ratio:.1f}:1) — retrained LogisticRegression and "
            "RandomForest with class_weight='balanced' (tree boosters used "
            "balanced sample weights)."
        )
        updates.update(
            {"results": results, "improvement_log": [note], "imbalance_addressed": True}
        )
        return updates

    if needs_skew:
        results, _ = _train_candidates(
            state,
            round_idx=round_idx,
            stage_label="robust_scaler",
            class_weight=None,
            scaler="robust",
        )
        cols = ", ".join(state["skewed_cols"])
        note = (
            f"Round {round_idx}: detected numeric skew in [{cols}] — "
            "rebuilt the preprocessor with RobustScaler and retrained all "
            "candidates."
        )
        updates.update(
            {
                "results": results,
                "improvement_log": [note],
                "skew_addressed": True,
                "scaler_type": "robust",
            }
        )
        return updates

    # Nothing actionable (router should normally prevent this).
    updates["improvement_log"] = [
        f"Round {round_idx}: no further deterministic fixes available."
    ]
    return updates


def generate_report_node(state: ExperimentState) -> dict:
    """Build the final markdown report string."""
    # Imported lazily to avoid a matplotlib import on every graph build.
    from ml.report import build_markdown_report

    return {"report_markdown": build_markdown_report(state)}


# --------------------------------------------------------------------------- #
# Conditional edge router
# --------------------------------------------------------------------------- #
def should_improve_router(state: ExperimentState) -> str:
    """Route to 'improve' or 'report'. The only conditional edge in the graph."""
    if state["round_count"] >= 2:
        return "report"
    needs_imbalance_fix = state["imbalance_detected"] and not state["imbalance_addressed"]
    needs_skew_fix = state["skew_detected"] and not state["skew_addressed"]
    if needs_imbalance_fix or needs_skew_fix:
        return "improve"
    return "report"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _split_xy(state: ExperimentState):
    """Build X/y and a train/test split, falling back to non-stratified."""
    df = state["raw_df"]
    target = state["target_column"]
    X = df.drop(columns=[target])  # target always dropped from features
    y = df[target]

    stratify = y if state["task_type"] == "classification" else None
    warning = None
    try:
        return (
            *train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
            ),
            warning,
        )
    except ValueError:
        # Too few samples in a class for a stratified split — fall back.
        warning = (
            "Dataset too small for a stratified split; used a plain "
            "random 80/20 split instead."
        )
        return (
            *train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
            ),
            warning,
        )


def _train_candidates(
    state: ExperimentState,
    round_idx: int,
    stage_label: str,
    class_weight: str | None,
    scaler: str,
) -> tuple[list[RunResult], str | None]:
    """Train, evaluate, and log every candidate model.

    Returns ``(results, split_warning)``.
    """
    X_train, X_test, y_train, y_test, warning = _split_xy(state)
    task_type = state["task_type"]
    models = get_candidate_models(task_type, class_weight=class_weight)

    results: list[RunResult] = []
    for name, estimator in models.items():
        preprocessor = build_preprocessor(
            state["numeric_cols"], state["categorical_cols"], scaler=scaler
        )
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )

        # Tree boosters get imbalance handling via sample weights.
        fit_kwargs = {}
        if (
            class_weight == "balanced"
            and task_type == "classification"
            and name in _SAMPLE_WEIGHT_MODELS
        ):
            sample_weight = compute_sample_weight("balanced", y_train)
            fit_kwargs["model__sample_weight"] = sample_weight

        pipeline.fit(X_train, y_train, **fit_kwargs)
        y_pred = pipeline.predict(X_test)

        y_proba = None
        if task_type == "classification" and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]

        run_metrics = metrics_mod.compute_metrics(task_type, y_test, y_pred, y_proba)

        n_features = len(state["numeric_cols"]) + len(state["categorical_cols"])
        run_id = log_run(
            pipeline=pipeline,
            model_name=name,
            round_idx=round_idx,
            stage_label=stage_label,
            task_type=task_type,
            n_features=n_features,
            scaler_type=scaler,
            metrics=run_metrics,
        )

        results.append(
            RunResult(
                model_name=name,
                round=round_idx,
                stage_label=stage_label,
                metrics=run_metrics,
                mlflow_run_id=run_id,
            )
        )
    return results, warning


def _build_leaderboard(results: list[RunResult], task_type: str) -> pd.DataFrame:
    """Flatten results into a sorted leaderboard dataframe (best first)."""
    rows = []
    for r in results:
        row = {
            "model": r["model_name"],
            "round": r["round"],
            "stage": r["stage_label"],
            "mlflow_run_id": r["mlflow_run_id"],
        }
        row.update(r["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    primary = metrics_mod.PRIMARY_METRIC[task_type]
    ascending = metrics_mod.PRIMARY_ASCENDING[task_type]
    if primary in df.columns:
        df = df.sort_values(primary, ascending=ascending).reset_index(drop=True)
    return df
