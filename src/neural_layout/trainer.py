"""Trainer for Neural Poster Layout Generation with Aim Tracking and Multi-Task Metrics.

Supports:
  - Reproducible training with random seeds
  - Train/Dev split evaluation
  - Full checkpointing (latest and best models with optimizer state)
  - Aim experiment tracking for every individual loss and evaluation metric
  - Gradient clipping and AdamW optimization with cosine annealing
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.neural_layout.dataset import PosterLayoutDataset, collate_layout_batch
from src.neural_layout.evaluator import LayoutEvaluator
from src.neural_layout.layout_model import TemplateConditionedLayoutModel
from src.neural_layout.losses import MultiTaskLayoutLoss, LossWeights


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    """Hyperparameters for neural layout generator training."""

    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    grad_clip_norm: float = 1.0
    d_model: int = 128
    num_heads: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 3
    ffn_dim: int = 256
    dropout: float = 0.1
    loss_weights: LossWeights = field(default_factory=LossWeights)
    checkpoint_dir: Path = Path("models/checkpoints/neural_layout")
    aim_repo: str = ".aim"
    experiment_name: str = "neural_layout_v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LayoutTrainer:
    """End-to-End Trainer for TemplateConditionedLayoutModel."""

    def __init__(
        self,
        model: TemplateConditionedLayoutModel,
        config: TrainingConfig,
        train_dataset: PosterLayoutDataset,
        val_dataset: PosterLayoutDataset | None = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss_fn = MultiTaskLayoutLoss(weights=config.loss_weights)
        self.evaluator = LayoutEvaluator()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.num_epochs),
            eta_min=1e-5,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_layout_batch,
        )

        self.val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_layout_batch,
            )

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_iou = -1.0
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        # Initialize Aim tracking
        self.aim_run = None
        self._init_aim()

    def _init_aim(self) -> None:
        """Initialize Aim run safely."""
        try:
            from aim import Run
            self.aim_run = Run(
                repo=self.config.aim_repo,
                experiment=self.config.experiment_name,
            )
            # Log hyperparameters
            self.aim_run["hparams"] = {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "num_epochs": self.config.num_epochs,
                "d_model": self.config.d_model,
                "encoder_layers": self.config.encoder_layers,
                "decoder_layers": self.config.decoder_layers,
                "device": self.config.device,
            }
        except Exception as e:
            print(f"[Warning] Aim tracking could not be initialized: {e}")
            self.aim_run = None

    def _log_metrics(self, metrics: dict[str, float], step: int, epoch: int, context: dict[str, str]) -> None:
        """Log metrics to Aim if available."""
        if self.aim_run is None:
            return
        try:
            for name, val in metrics.items():
                self.aim_run.track(
                    val,
                    name=name,
                    step=step,
                    epoch=epoch,
                    context=context,
                )
        except Exception:
            pass

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        accum_losses: dict[str, float] = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch tensors to device
            batch_dev = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()
            predictions = self.model(
                content_feats=batch_dev["content_feats"],
                template_feats=batch_dev["template_feats"],
                content_mask=batch_dev["content_mask"],
                template_mask=batch_dev["template_mask"],
            )

            total_loss, loss_dict = self.loss_fn(predictions, batch_dev)
            total_loss.backward()

            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )

            self.optimizer.step()

            for k, v in loss_dict.items():
                accum_losses[k] = accum_losses.get(k, 0.0) + v
            num_batches += 1

            step = epoch * len(self.train_loader) + batch_idx
            self._log_metrics(loss_dict, step=step, epoch=epoch, context={"subset": "train_step"})

        self.scheduler.step()

        epoch_metrics = {k: v / max(1, num_batches) for k, v in accum_losses.items()}
        epoch_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self._log_metrics(epoch_metrics, step=epoch, epoch=epoch, context={"subset": "train_epoch"})
        return epoch_metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Execute validation over the validation dataset."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        accum_eval: dict[str, float] = {}
        accum_losses: dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            batch_dev = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            predictions = self.model(
                content_feats=batch_dev["content_feats"],
                template_feats=batch_dev["template_feats"],
                content_mask=batch_dev["content_mask"],
                template_mask=batch_dev["template_mask"],
            )

            _, loss_dict = self.loss_fn(predictions, batch_dev)
            eval_metrics = self.evaluator.evaluate_batch(predictions, batch_dev)

            for k, v in loss_dict.items():
                accum_losses[k] = accum_losses.get(k, 0.0) + v
            for k, v in eval_metrics.items():
                accum_eval[k] = accum_eval.get(k, 0.0) + v
            num_batches += 1

        val_metrics: dict[str, float] = {}
        for k, v in accum_losses.items():
            val_metrics[f"val_{k}"] = v / max(1, num_batches)
        for k, v in accum_eval.items():
            val_metrics[f"val_{k}"] = v / max(1, num_batches)

        self._log_metrics(val_metrics, step=epoch, epoch=epoch, context={"subset": "val_epoch"})
        return val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        filename: str = "latest_checkpoint.pt",
    ) -> Path:
        """Save training checkpoint."""
        ckpt_path = self.config.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(state, ckpt_path)

        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            return best_path

        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load state from checkpoint."""
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.start_epoch = state.get("epoch", 0) + 1
        return state

    def train(self) -> dict[str, list[float]]:
        """Run complete training cycle."""
        set_seed(self.config.seed)
        history: dict[str, list[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_giou": [],
            "val_boundary_violations": [],
            "val_unintended_overlaps": [],
            "val_col_acc": [],
            "val_order_tau": [],
        }

        print(f"Starting Neural Layout Training for {self.config.num_epochs} epochs on {self.device}...")
        for epoch in range(self.start_epoch, self.config.num_epochs):
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_m.get("loss_total", 0.0))
            history["val_loss"].append(val_m.get("val_loss_total", 0.0))
            history["val_iou"].append(val_m.get("val_iou_mean", 0.0))
            history["val_giou"].append(val_m.get("val_giou_mean", 0.0))
            history["val_boundary_violations"].append(val_m.get("val_canvas_violation_rate", 0.0))
            history["val_unintended_overlaps"].append(val_m.get("val_unintended_overlap_rate", 0.0))
            history["val_col_acc"].append(val_m.get("val_column_accuracy", 0.0))
            history["val_order_tau"].append(val_m.get("val_reading_order_kendall_tau", 0.0))

            val_iou = val_m.get("val_iou_mean", 0.0)
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou

            self.save_checkpoint(epoch=epoch, metrics=val_m, is_best=is_best)

            # Concise epoch progress log
            print(
                f"[Epoch {epoch + 1:02d}/{self.config.num_epochs:02d}] "
                f"Train Loss: {train_m.get('loss_total', 0.0):.4f} | "
                f"Val Loss: {val_m.get('val_loss_total', 0.0):.4f} | "
                f"Val IoU: {val_iou:.4f} | "
                f"Overlap Rate: {val_m.get('val_unintended_overlap_rate', 0.0):.3f} | "
                f"Col Acc: {val_m.get('val_column_accuracy', 0.0):.3f} | "
                f"Order Tau: {val_m.get('val_reading_order_kendall_tau', 0.0):.3f} "
                f"{'(*best*)' if is_best else ''}"
            )

        if self.aim_run is not None:
            try:
                self.aim_run.close()
            except Exception:
                pass

        return history
