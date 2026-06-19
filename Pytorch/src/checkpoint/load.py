"""
checkpoint/load.py
==================

Checkpoint loading utilities.

Responsibilities
----------------
1. Load model checkpoint
2. Restore optimizer state
3. Restore scheduler state
4. Resume training from saved epoch
5. Support Single GPU, DDP and FSDP
"""

from __future__ import annotations

import os
import logging

import torch


logger = logging.getLogger(__name__)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    config,
):
    """
    Load checkpoint if resume path is provided.

    Returns
    -------
    int
        Starting epoch.
    """

    checkpoint_path = config["checkpoint"]["resume"]

    if checkpoint_path is None:

        logger.info("No checkpoint found. Starting fresh training.")

        return 0

    if not os.path.isfile(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    logger.info(
        f"Loading checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    strategy = config["distributed"]["strategy"]

    if strategy == "ddp":

        if hasattr(model, "module"):
            model.module.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

    elif strategy == "fsdp":

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    start_epoch = checkpoint["epoch"] + 1

    logger.info(
        f"Successfully resumed training from epoch {start_epoch}"
    )

    return start_epoch
