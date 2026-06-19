"""
checkpoint/save.py
==================

Checkpoint saving utilities.

Responsibilities
----------------
1. Save model checkpoint
2. Save optimizer state
3. Save scheduler state
4. Save training metadata
5. Support DDP/FSDP
"""

from __future__ import annotations

import os
import logging

import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


logger = logging.getLogger(__name__)


def is_main_process():

    if not dist.is_initialized():
        return True

    return dist.get_rank() == 0


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    config,
):
    """
    Save training checkpoint.
    """

    if not is_main_process():
        return

    checkpoint_dir = config["checkpoint"]["save_dir"]

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_epoch_{epoch + 1}.pt",
    )

    strategy = config["distributed"]["strategy"]

    if strategy == "fsdp":

        save_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):

            model_state = model.state_dict()

    else:

        if hasattr(model, "module"):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model_state,

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "config": config,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    logger.info(
        f"Checkpoint saved: {checkpoint_path}"
    )
