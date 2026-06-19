"""
monitoring/mlflow.py
====================

MLflow utilities.

Responsibilities
----------------
1. Initialize MLflow
2. Log parameters
3. Log metrics
4. Save artifacts
5. Finish run
"""

from __future__ import annotations

import logging

import mlflow


logger = logging.getLogger(__name__)


def init_mlflow(config):
    """
    Initialize MLflow experiment.
    """

    if not config["mlflow"]["enabled"]:
        return

    mlflow.set_experiment(
        config["mlflow"]["experiment"]
    )

    mlflow.start_run(
        run_name=config["mlflow"]["run_name"]
    )

    params = {

        "model": config["model"]["name"],

        "dataset": config["dataset"]["name"],

        "batch_size": config["dataloader"]["batch_size"],

        "learning_rate": config["optimizer"]["learning_rate"],

        "epochs": config["training"]["epochs"],

        "strategy": config["distributed"]["strategy"],
    }

    mlflow.log_params(params)

    logger.info("MLflow initialized.")


def log_metrics(
    metrics: dict,
    step: int,
):
    """
    Log metrics to MLflow.
    """

    if mlflow.active_run() is None:
        return

    mlflow.log_metrics(
        metrics,
        step=step,
    )


def log_params(
    params: dict,
):
    """
    Log additional parameters.
    """

    if mlflow.active_run() is None:
        return

    mlflow.log_params(params)


def log_artifact(
    artifact_path: str,
):
    """
    Log artifact.
    """

    if mlflow.active_run() is None:
        return

    mlflow.log_artifact(
        artifact_path,
    )


def finish_mlflow():
    """
    End MLflow run.
    """

    if mlflow.active_run() is None:
        return

    mlflow.end_run()

    logger.info("MLflow run completed.")
