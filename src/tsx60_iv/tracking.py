"""Minimal MLflow experiment tracking interfaces."""

from typing import Any

import mlflow


def log_surface_run(
    parameters: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[str] | None = None,
    experiment_name: str = "tsx60-iv-surface",
) -> str:
    """Log a surface calculation and return its MLflow run identifier.

    Parameters
    ----------
    parameters
        Scalar model and data parameters.
    metrics
        Numeric fit and arbitrage diagnostics.
    artifacts
        Optional local artifact paths to upload.
    experiment_name
        MLflow experiment name.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        for artifact in artifacts or []:
            mlflow.log_artifact(artifact)
        return run.info.run_id
