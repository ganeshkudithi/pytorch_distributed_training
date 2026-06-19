"""
trainer/train.py
================

Main entry point for distributed SFT training.

Supports:
    - Single GPU
    - DDP
    - FSDP
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.utils.config import load_config
from src.utils.seed import set_seed

from src.data.tokenizer import load_tokenizer
from src.data.dataset import load_sft_dataset

from src.model.loader import load_model
from src.model.wrapping import (
    apply_liger,
    apply_compile,
)

from src.distributed.ddp import (
    setup_ddp,
    wrap_ddp,
    cleanup as cleanup_ddp,
)

from src.distributed.fsdp import (
    setup_fsdp,
    wrap_fsdp,
    cleanup as cleanup_fsdp,
)

from src.trainer.engine import Trainer
from src.trainer.evaluate import evaluate

from src.checkpoint.save import save_checkpoint
from src.checkpoint.load import load_checkpoint

from src.monitoring.mlflow import (
    init_mlflow,
    finish_mlflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Distributed SFT Training"
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration",
    )

    return parser.parse_args()


def build_dataloader(dataset, config, is_train=True):

    if is_train:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            shuffle=False,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config["dataloader"]["batch_size"],
        sampler=sampler,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        drop_last=config["dataloader"]["drop_last"],
    )

    return dataloader, sampler


def build_optimizer(model, config):

    return torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
    )


def build_scheduler(
    optimizer,
    train_loader,
    config,
):

    grad_accum_steps = config["dataloader"].get(
        "gradient_accumulation_steps", 1
    )

    total_steps = (
        len(train_loader)
        // grad_accum_steps
        * config["training"]["epochs"]
    )

    warmup_steps = int(
        total_steps
        * config["scheduler"]["warmup_ratio"]
    )

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

def main():

    # ---------------------------------------------------------
    # Parse Arguments
    # ---------------------------------------------------------

    args = parse_args()

    # ---------------------------------------------------------
    # Load Configuration
    # ---------------------------------------------------------

    config = load_config(args.config)

    # ---------------------------------------------------------
    # Set Random Seed
    # ---------------------------------------------------------

    set_seed(config["seed"])

    # ---------------------------------------------------------
    # Initialize Distributed Training
    # ---------------------------------------------------------

    strategy = config["distributed"]["strategy"].lower()

    if strategy == "ddp":

        logger.info("Initializing DDP...")

        local_rank = setup_ddp()

    elif strategy == "fsdp":

        logger.info("Initializing FSDP...")

        local_rank = setup_fsdp()

    else:

        raise ValueError(
            f"Unsupported distributed strategy: {strategy}"
        )

    # ---------------------------------------------------------
    # Initialize MLflow (rank 0 only)
    # ---------------------------------------------------------

    import torch.distributed as dist

    if dist.get_rank() == 0:
        init_mlflow(config)

    # ---------------------------------------------------------
    # Load Tokenizer
    # ---------------------------------------------------------

    logger.info("Loading tokenizer...")

    tokenizer = load_tokenizer(
        config["model"]["name"]
    )

    # ---------------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------------

    logger.info("Loading training dataset...")

    train_dataset = load_sft_dataset(
        config,
        tokenizer,
    )

    # ---------------------------------------------------------
    # Load Validation Dataset (if evaluation enabled)
    # ---------------------------------------------------------

    val_loader = None
    val_sampler = None

    if config["evaluation"]["enabled"]:

        logger.info("Loading validation dataset...")

        val_config = dict(config)
        val_dataset_config = dict(config["dataset"])

        # Use the validation split
        val_dataset_config["split"] = config["dataset"].get(
            "val_split", "test_sft"
        )

        val_config["dataset"] = val_dataset_config

        val_dataset = load_sft_dataset(
            val_config,
            tokenizer,
        )

        val_loader, val_sampler = build_dataloader(
            val_dataset,
            config,
            is_train=False,
        )

        logger.info(
            f"Validation samples : {len(val_dataset)}"
        )

    # ---------------------------------------------------------
    # Create DataLoader
    # ---------------------------------------------------------

    train_loader, train_sampler = build_dataloader(
        train_dataset,
        config,
    )

    logger.info(
        f"Training samples : {len(train_dataset)}"
    )

    logger.info(
        f"Steps/Epoch      : {len(train_loader)}"
    )

    # ---------------------------------------------------------
    # Load Model
    # ---------------------------------------------------------

    logger.info("Loading model...")

    model = load_model(config)

    # ---------------------------------------------------------
    # Apply Liger Kernel (before wrapping)
    # ---------------------------------------------------------

    model = apply_liger(
        model,
        config,
    )

    # ---------------------------------------------------------
    # Wrap Model
    # ---------------------------------------------------------

    if strategy == "ddp":

        logger.info("Wrapping model with DDP...")

        model = wrap_ddp(
            model,
            local_rank,
            config,
        )

    elif strategy == "fsdp":

        logger.info("Wrapping model with FSDP...")

        model = wrap_fsdp(
            model,
            config,
        )

    # ---------------------------------------------------------
    # torch.compile (after wrapping)
    # ---------------------------------------------------------

    model = apply_compile(
        model,
        config,
    )

    logger.info("Model initialization completed.")

    # ---------------------------------------------------------
    # Build Optimizer
    # ---------------------------------------------------------

    optimizer = build_optimizer(
        model,
        config,
    )

    # ---------------------------------------------------------
    # Build Scheduler
    # ---------------------------------------------------------

    scheduler = build_scheduler(
        optimizer,
        train_loader,
        config,
    )

    logger.info("Optimizer initialized.")

    logger.info("Scheduler initialized.")


    # ---------------------------------------------------------
    # Resume Checkpoint (Optional)
    # ---------------------------------------------------------

    start_epoch = 0

    if config["checkpoint"]["resume"] is not None:

        logger.info("Loading checkpoint...")

        start_epoch = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
        )

    # ---------------------------------------------------------
    # Create Trainer
    # ---------------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    logger.info("Trainer initialized.")

    # ---------------------------------------------------------
    # Start Training
    # ---------------------------------------------------------

    logger.info("=" * 80)
    logger.info("Starting Supervised Fine-Tuning")
    logger.info("=" * 80)

    for epoch in range(
        start_epoch,
        config["training"]["epochs"],
    ):

        logger.info(
            f"Epoch [{epoch + 1}/{config['training']['epochs']}]"
        )

        # Required for DistributedSampler
        train_sampler.set_epoch(epoch)

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        train_metrics = trainer.train_epoch(epoch)

        logger.info(
            f"Training Loss : {train_metrics['loss']:.4f}"
        )

        logger.info(
            f"Epoch Time    : {train_metrics['epoch_time']:.2f} sec"
        )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        if config["evaluation"]["enabled"] and val_loader is not None:

            logger.info("Running validation...")

            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            eval_metrics = evaluate(
                model=model,
                dataloader=val_loader,
            )

            logger.info(
                f"Validation Loss : {eval_metrics['loss']:.4f}"
            )

            logger.info(
                f"Perplexity      : {eval_metrics['perplexity']:.4f}"
            )

        # -----------------------------------------------------
        # Save Checkpoint
        # -----------------------------------------------------

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
        )

    logger.info("=" * 80)
    logger.info("Training Finished Successfully.")
    logger.info("=" * 80)

    # ---------------------------------------------------------
    # Finish MLflow
    # ---------------------------------------------------------

    if dist.get_rank() == 0:
        finish_mlflow()

    # ---------------------------------------------------------
    # Cleanup Distributed Training
    # ---------------------------------------------------------

    if strategy == "ddp":

        cleanup_ddp()

    elif strategy == "fsdp":

        cleanup_fsdp()

    logger.info("Distributed resources released.")


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        logger.info("Training interrupted by user.")

    except Exception as e:

        logger.exception(
            f"Training failed: {e}"
        )

        raise
