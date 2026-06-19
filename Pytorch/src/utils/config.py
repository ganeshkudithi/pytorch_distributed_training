"""
utils/config.py
===============

Configuration utilities.

Responsibilities
----------------
1. Load YAML configuration
2. Validate configuration
3. Pretty print configuration
"""

from __future__ import annotations

import os
import yaml
import logging
from pprint import pformat


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML config.

    Returns:
        dict: Configuration dictionary.
    """

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    validate_config(config)

    logger.info("Configuration loaded successfully.")

    return config


def validate_config(config: dict) -> None:
    """ 
    Validate mandatory configuration fields.
    """

    required_sections = [
        "model",
        "dataset",
        "dataloader",
        "optimizer",
        "scheduler",
        "training",
        "distributed",
        "checkpoint",
        "logging",
        "mlflow",
        "evaluation",
    ]

    missing = []

    for section in required_sections:
        if section not in config:
            missing.append(section)

    if missing:
        raise ValueError(
            f"Missing configuration sections: {missing}"
        )


def print_config(config: dict) -> None:
    """
    Pretty print configuration.
    """

    logger.info("\n%s", pformat(config))


def get_config_value(
    config: dict,
    key: str,
    default=None,
):
    """
    Safely retrieve configuration value.

    Example:
        lr = get_config_value(
            config,
            "optimizer.learning_rate"
        )
    """

    value = config

    for item in key.split("."):

        if isinstance(value, dict):

            value = value.get(item)

        else:

            return default

        if value is None:

            return default

    return value
