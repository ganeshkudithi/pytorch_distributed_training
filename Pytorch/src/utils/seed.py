"""
utils/seed.py
=============

Utilities for reproducibility.

Responsibilities
----------------
1. Set random seed
2. Configure CUDA determinism
3. Configure cuDNN
"""

from __future__ import annotations

import os
import random
import logging

import numpy as np
import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def set_seed(
    seed: int = 42,
    deterministic: bool = False,
) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
        deterministic (bool): Enable deterministic training.
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    logger.info(f"Random seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """
    Seed DataLoader workers.

    Args:
        worker_id (int): Worker ID.
    """

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Create a seeded torch Generator.

    Returns:
        torch.Generator
    """

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def is_main_process() -> bool:
    """
    Returns True if current process is rank 0.
    """

    if not dist.is_initialized():
        return True

    return dist.get_rank() == 0


def synchronize() -> None:
    """
    Synchronize all distributed processes.
    """

    if dist.is_initialized():
        dist.barrier()
