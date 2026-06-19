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
import torch.distributed as dist


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

    # Map to appropriate device to avoid GPU memory spike on rank 0
    rank = dist.get_rank() if dist.is_initialized() else 0
    map_location = {"cpu": "cpu"} if rank != 0 else "cpu"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
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

        # For FSDP, load via FSDP's own API with FULL_STATE_DICT
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        load_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            load_policy,
        ):
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
