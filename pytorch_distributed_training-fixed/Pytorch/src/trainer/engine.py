"""
trainer/engine.py
=================

Core training engine for Supervised Fine-Tuning (SFT).

Responsibilities
----------------
1. Forward pass
2. Loss computation
3. Backward pass
4. Gradient accumulation
5. Gradient clipping
6. Optimizer step
7. Scheduler step
8. Compute training metrics
9. MLflow logging
10. Console logging
"""

from __future__ import annotations

import logging
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from src.monitoring.metrics import (
    compute_grad_norm,
    compute_tokens_per_second,
    compute_samples_per_second,
)

from src.monitoring.gpu import (
    get_gpu_memory,
    get_gpu_utilization,
    get_cpu_memory,
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

        self.rank = (
            dist.get_rank()
            if dist.is_initialized()
            else 0
        )

        self.world_size = (
            dist.get_world_size()
            if dist.is_initialized()
            else 1
        )

        self.device = torch.cuda.current_device()

        self.grad_accum_steps = config["dataloader"].get(
            "gradient_accumulation_steps", 1
        )

    def train_epoch(self, epoch):

        self.model.train()

        running_loss = 0.0

        epoch_start = time.perf_counter()

        progress = tqdm(
            self.train_loader,
            disable=self.rank != 0,
            desc=f"Epoch {epoch}",
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress):

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

            # Scale loss for gradient accumulation
            loss = outputs.loss / self.grad_accum_steps

            loss.backward()

            is_accumulation_step = (
                (step + 1) % self.grad_accum_steps != 0
                and (step + 1) != len(self.train_loader)
            )

            if not is_accumulation_step:

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["max_grad_norm"],
                )

                self.optimizer.step()

                self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

            else:
                grad_norm = torch.tensor(0.0)

            step_time = (
                time.perf_counter()
                - step_start
            )

            # Unscaled loss for logging
            unscaled_loss = loss.item() * self.grad_accum_steps

            running_loss += unscaled_loss

            lr = self.scheduler.get_last_lr()[0]

            tokens_per_sec = compute_tokens_per_second(
                batch=batch,
                elapsed_time=step_time,
                world_size=self.world_size,
            )

            samples_per_sec = compute_samples_per_second(
                batch=batch,
                elapsed_time=step_time,
                world_size=self.world_size,
            )

            gpu_util = get_gpu_utilization()

            gpu_memory = get_gpu_memory()

            cpu_memory = get_cpu_memory()

            if self.rank == 0:

                logger.info(
                    f"[epoch {epoch}] "
                    f"[step {step}] "
                    f"loss={unscaled_loss:.4f} "
                    f"lr={lr:.2e} "
                    f"grad_norm={grad_norm:.4f} "
                    f"tok/s={tokens_per_sec:.2f} "
                    f"samples/s={samples_per_sec:.2f} "
                    f"GPU={gpu_util:.1f}% "
                    f"GPU_mem={gpu_memory:.2f}GB "
                    f"RAM={cpu_memory:.2f}GB"
                )

                progress.set_postfix(
                    loss=f"{unscaled_loss:.4f}",
                    tok_s=f"{tokens_per_sec:.1f}",
                    gpu=f"{gpu_util:.0f}%",
                )

                log_metrics(
                    {
                        "loss": unscaled_loss,
                        "learning_rate": lr,
                        "grad_norm": float(grad_norm),
                        "tokens_per_second": tokens_per_sec,
                        "samples_per_second": samples_per_sec,
                        "gpu_utilization": gpu_util,
                        "gpu_memory_gb": gpu_memory,
                        "cpu_memory_gb": cpu_memory,
                        "step_time": step_time,
                    },
                    step=epoch * len(self.train_loader) + step,
                )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        average_loss = (
            running_loss
            / len(self.train_loader)
        )

        return {
            "loss": average_loss,
            "epoch_time": epoch_time,
        }
