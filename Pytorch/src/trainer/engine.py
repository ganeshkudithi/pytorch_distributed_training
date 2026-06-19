"""
trainer/engine.py

Training engine for Supervised Fine-Tuning (SFT).

Responsibilities:
    - Forward pass
    - Backward pass
    - Gradient clipping
    - Optimizer step
    - Scheduler step
    - Metric logging
    - MLflow logging
"""

from __future__ import annotations

import time
import logging

import torch
import torch.distributed as dist

from tqdm import tqdm

from src.monitoring.metrics import (
    compute_grad_norm,
    compute_tokens_per_second,
)

from src.monitoring.gpu import (
    get_gpu_utilization,
    get_gpu_memory,
)

from src.monitoring.mlflow import log_metrics


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        config,
    ):

        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = (
            dist.get_world_size()
            if dist.is_initialized()
            else 1
        )

        self.device = torch.cuda.current_device()

    def train_epoch(self, epoch):

        self.model.train()

        running_loss = 0.0

        epoch_start = time.perf_counter()

        progress_bar = tqdm(
            self.train_loader,
            disable=self.rank != 0,
            desc=f"Epoch {epoch}",
        )

        for step, batch in enumerate(progress_bar):

            step_start = time.perf_counter()

            input_ids = batch["input_ids"].to(
                self.device,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["max_grad_norm"],
            )

            self.optimizer.step()

            self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)

            step_time = time.perf_counter() - step_start

            running_loss += loss.item()

            lr = self.scheduler.get_last_lr()[0]

            tokens_per_second = compute_tokens_per_second(
                batch,
                step_time,
                self.world_size,
            )

            gpu_util = get_gpu_utilization()

            gpu_memory = get_gpu_memory()

            if self.rank == 0:

                logger.info(
                    f"[epoch {epoch}] "
                    f"[step {step}] "
                    f"loss={loss.item():.4f} "
                    f"lr={lr:.2e} "
                    f"grad_norm={grad_norm:.4f} "
                    f"tok/s={tokens_per_second:.2f} "
                    f"GPU={gpu_util:.1f}% "
                    f"GPU_mem={gpu_memory:.2f}GB"
                )

                log_metrics(
                    {
                        "loss": loss.item(),
                        "learning_rate": lr,
                        "grad_norm": float(grad_norm),
                        "tokens_per_second": tokens_per_second,
                        "gpu_utilization": gpu_util,
                        "gpu_memory_gb": gpu_memory,
                    },
                    step=epoch * len(self.train_loader) + step,
                )

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    tok_s=f"{tokens_per_second:.1f}",
                )

        epoch_time = time.perf_counter() - epoch_start

        avg_loss = running_loss / len(self.train_loader)

        return avg_loss, epoch_time
