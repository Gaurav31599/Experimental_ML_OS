"""Builds and compiles the ExperimentOS StateGraph.

Topology::

    START -> ingest -> detect_task_type -> profile_data -> build_preprocessor
          -> train_baseline -> rank -> [should_improve_router]
                                          ├─ improve -> apply_improvement -> rank
                                          └─ report  -> generate_report -> END

The single conditional edge (``should_improve_router``) is the only branch.
Everything else is a straight, deterministic chain. The improve branch loops
back to ``rank`` so the leaderboard is rebuilt after each round.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import ExperimentState


def build_workflow():
    """Construct and compile the workflow graph."""
    graph = StateGraph(ExperimentState)

    graph.add_node("ingest", nodes.ingest_node)
    graph.add_node("detect_task_type", nodes.detect_task_type_node)
    graph.add_node("profile_data", nodes.profile_data_node)
    graph.add_node("build_preprocessor", nodes.build_preprocessor_node)
    graph.add_node("train_baseline", nodes.train_baseline_node)
    graph.add_node("rank", nodes.rank_node)
    graph.add_node("apply_improvement", nodes.apply_improvement_node)
    graph.add_node("generate_report", nodes.generate_report_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "detect_task_type")
    graph.add_edge("detect_task_type", "profile_data")
    graph.add_edge("profile_data", "build_preprocessor")
    graph.add_edge("build_preprocessor", "train_baseline")
    graph.add_edge("train_baseline", "rank")

    graph.add_conditional_edges(
        "rank",
        nodes.should_improve_router,
        {"improve": "apply_improvement", "report": "generate_report"},
    )
    graph.add_edge("apply_improvement", "rank")
    graph.add_edge("generate_report", END)

    return graph.compile()


def initial_state(
    raw_df, target_column: str, mlflow_experiment_id: str
) -> ExperimentState:
    """Build a fully-populated initial state for ``graph.invoke``."""
    return ExperimentState(
        raw_df=raw_df,
        target_column=target_column,
        task_type="classification",
        numeric_cols=[],
        categorical_cols=[],
        imbalance_detected=False,
        imbalance_ratio=None,
        skew_detected=False,
        skewed_cols=[],
        round_count=0,
        imbalance_addressed=False,
        skew_addressed=False,
        results=[],
        leaderboard=None,
        improvement_log=[],
        mlflow_experiment_id=mlflow_experiment_id,
        report_markdown=None,
        scaler_type="standard",
        split_warning=None,
    )
