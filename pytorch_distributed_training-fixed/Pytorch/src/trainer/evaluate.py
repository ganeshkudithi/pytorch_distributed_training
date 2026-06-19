"""
trainer/evaluate.py
===================

Evaluation loop for Supervised Fine-Tuning (SFT).

Responsibilities
----------------
1. Run model in evaluation mode
2. Disable gradient computation
3. Compute validation loss
4. Compute perplexity
5. Return evaluation metrics
"""

from __future__ import annotations

import math
import logging

import torch
import torch.distributed as dist

from tqdm import tqdm


logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
):
    """
    Evaluate model on validation dataset.

    Args:
        model: HuggingFace Causal LM
        dataloader: Validation DataLoader

    Returns:
        Dictionary containing validation metrics.
    """

    rank = (
        dist.get_rank()
        if dist.is_initialized()
        else 0
    )

    device = torch.cuda.current_device()

    model.eval()

    total_loss = 0.0
    total_steps = 0

    progress = tqdm(
        dataloader,
        disable=rank != 0,
        desc="Validation",
    )

    for batch in progress:

        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )

        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = outputs.loss

        total_loss += loss.item()
        total_steps += 1

        if rank == 0:

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    avg_loss = total_loss / total_steps

    perplexity = math.exp(avg_loss)

    if rank == 0:

        logger.info(
            f"Validation Loss : {avg_loss:.4f}"
        )

        logger.info(
            f"Perplexity      : {perplexity:.4f}"
        )

    model.train()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }
