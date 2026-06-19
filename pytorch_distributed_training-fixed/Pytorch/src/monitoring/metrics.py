"""
monitoring/metrics.py
=====================

Training metric utilities.

Responsibilities
----------------
1. Gradient norm
2. Tokens / second
3. Samples / second
4. Step time
5. Epoch time
6. Model FLOP Utilization (MFU)
"""

from __future__ import annotations

import torch


def compute_grad_norm(model) -> float:
    """
    Compute L2 gradient norm.
    """

    total_norm = 0.0

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        param_norm = parameter.grad.detach().data.norm(2)

        total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


def compute_tokens_per_second(
    batch,
    elapsed_time: float,
    world_size: int = 1,
) -> float:
    """
    Compute throughput in tokens/sec.
    """

    if elapsed_time <= 0:
        return 0.0

    tokens = batch["input_ids"].numel()

    tokens *= world_size

    return tokens / elapsed_time


def compute_samples_per_second(
    batch,
    elapsed_time: float,
    world_size: int = 1,
) -> float:
    """
    Compute throughput in samples/sec.
    """

    if elapsed_time <= 0:
        return 0.0

    samples = batch["input_ids"].size(0)

    samples *= world_size

    return samples / elapsed_time


def compute_step_time(
    start_time: float,
    end_time: float,
) -> float:
    """
    Step execution time.
    """

    return end_time - start_time


def compute_epoch_time(
    start_time: float,
    end_time: float,
) -> float:
    """
    Epoch execution time.
    """

    return end_time - start_time


def compute_mfu(
    model_parameters: int,
    tokens_processed: int,
    elapsed_time: float,
    peak_flops: float,
) -> float:
    """
    Approximate Model FLOP Utilization (MFU).

    FLOPs ≈ 6 × Parameters × Tokens
    """

    if elapsed_time <= 0:
        return 0.0

    achieved_flops = (
        6
        * model_parameters
        * tokens_processed
    ) / elapsed_time

    return achieved_flops / peak_flops
