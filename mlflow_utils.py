"""Local file-based MLflow helpers.

Tracking is always local (``./mlruns``) — no remote server. Each model/round
combination becomes one MLflow run.
"""

from __future__ import annotations

import os

# MLflow 3.x guards the file-based tracking store behind this opt-in. We use a
# local ./mlruns file store by design (single-user MVP), so enable it before the
# mlflow import resolves any store.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from datetime import datetime

import mlflow
from sklearn.pipeline import Pipeline

TRACKING_URI = "./mlruns"


def init_experiment(uploaded_filename: str) -> str:
    """Point MLflow at the local store and create/select an experiment.

    Returns the experiment id.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"ExperimentOS_{uploaded_filename}_{timestamp}"
    experiment = mlflow.set_experiment(name)
    return experiment.experiment_id


def log_run(
    pipeline: Pipeline,
    model_name: str,
    round_idx: int,
    stage_label: str,
    task_type: str,
    n_features: int,
    scaler_type: str,
    metrics: dict[str, float],
) -> str:
    """Log one fitted pipeline as an MLflow run; return the run id."""
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_name": model_name,
                "round": round_idx,
                "stage_label": stage_label,
                "task_type": task_type,
                "n_features": n_features,
                "scaler_type": scaler_type,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"round": round_idx, "stage": stage_label})
        # cloudpickle avoids MLflow 3.x's skops "untrusted types" gate on the
        # numpy dtypes inside fitted sklearn pipelines.
        mlflow.sklearn.log_model(
            pipeline, name="model", serialization_format="cloudpickle"
        )
        return run.info.run_id
