"""Trainer for Neural Poster Layout Generation with ClearML Tracking and Multi-Task Metrics.

Supports:
  - Reproducible training with random seeds
  - Train/Dev split evaluation
  - Full checkpointing (latest and best models with optimizer state)
  - ClearML experiment tracking for all individual loss components, evaluation metrics,
    hyperparameters, environment metadata, and visualizations
  - Gradient clipping and AdamW optimization with cosine annealing
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from clearml import Task
except ImportError:
    Task = None

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


def get_git_info() -> dict[str, str]:
    """Retrieve git commit hash and branch safely."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {
            "git_commit": commit,
            "git_branch": branch,
            "git_dirty": str(bool(status)),
        }
    except Exception:
        return {
            "git_commit": "unknown",
            "git_branch": "unknown",
            "git_dirty": "unknown",
        }


@dataclass
class TrainingConfig:
    """Hyperparameters and ClearML tracking configuration for neural layout training."""

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
    project_name: str = "Infographix"
    task_name: str = "Neural Layout Training - Local CPU"
    task_type: str = "training"
    tags: list[str] = field(default_factory=lambda: ["neural_layout", "stage_b", "cpu", "local"])
    reuse_last_task_id: bool = False
    log_to_clearml: bool = True
    execution_env: str = "local_cpu"
    dataset_version: str = "SciPostLayout v2.0.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LayoutTrainer:
    """End-to-End Trainer for TemplateConditionedLayoutModel."""

    def __init__(
        self,
        model: TemplateConditionedLayoutModel,
        config: TrainingConfig,
        train_dataset: PosterLayoutDataset,
        val_dataset: PosterLayoutDataset | None = None,
        clearml_task: Any | None = None,
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

        # Initialize ClearML tracking
        self.clearml_task = clearml_task
        self.clearml_logger = None
        self._owns_clearml_task = False
        if self.config.log_to_clearml:
            self._init_clearml()

    def _init_clearml(self) -> None:
        """Connect to existing ClearML Task or initialize a new one, and record config."""
        if Task is None:
            print("[Warning] clearml package is not installed; skipping experiment tracking.")
            return

        try:
            if self.clearml_task is None:
                self.clearml_task = Task.current_task()
                if self.clearml_task is None:
                    self.clearml_task = Task.init(
                        project_name=self.config.project_name,
                        task_name=self.config.task_name,
                        task_type=self.config.task_type,
                        reuse_last_task_id=self.config.reuse_last_task_id,
                        tags=self.config.tags,
                        auto_connect_frameworks=False,
                    )
                    self._owns_clearml_task = True

            self.clearml_logger = self.clearml_task.get_logger()
            self._log_configuration()
        except Exception as e:
            print(f"[Warning] ClearML tracking could not be initialized: {e}")
            self.clearml_task = None
            self.clearml_logger = None

    def _log_configuration(self) -> None:
        """Capture full training, model, loss, dataset, and environment configuration in ClearML."""
        if self.clearml_task is None:
            return

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        git_info = get_git_info()
        device_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and "cuda" in str(self.device)
            else "CPU"
        )

        hyperparams = {
            "model/architecture": "TemplateConditionedLayoutModel",
            "model/d_model": self.config.d_model,
            "model/num_heads": self.config.num_heads,
            "model/encoder_layers": self.config.encoder_layers,
            "model/decoder_layers": self.config.decoder_layers,
            "model/ffn_dim": self.config.ffn_dim,
            "model/num_columns": self.model.num_columns,
            "model/dropout": self.config.dropout,
            "model/trainable_parameters": total_params,
            "training/batch_size": self.config.batch_size,
            "training/learning_rate": self.config.learning_rate,
            "training/weight_decay": self.config.weight_decay,
            "training/num_epochs": self.config.num_epochs,
            "training/grad_clip_norm": self.config.grad_clip_norm,
            "training/seed": self.config.seed,
            "training/optimizer": "AdamW",
            "training/scheduler": f"CosineAnnealingLR(T_max={self.config.num_epochs}, eta_min=1e-5)",
            "training/device": str(self.device),
            "training/device_name": device_name,
            "training/execution_env": self.config.execution_env,
            "loss_weights/w_box": self.config.loss_weights.w_box,
            "loss_weights/w_giou": self.config.loss_weights.w_giou,
            "loss_weights/w_bound": self.config.loss_weights.w_bound,
            "loss_weights/w_overlap": self.config.loss_weights.w_overlap,
            "loss_weights/w_col": self.config.loss_weights.w_col,
            "loss_weights/w_aspect": self.config.loss_weights.w_aspect,
            "loss_weights/w_order": self.config.loss_weights.w_order,
            "loss_weights/w_macro": self.config.loss_weights.w_macro,
            "dataset/version": self.config.dataset_version,
            "dataset/train_samples": len(self.train_dataset),
            "dataset/val_samples": len(self.val_dataset) if self.val_dataset is not None else 0,
            "environment/git_commit": git_info["git_commit"],
            "environment/git_branch": git_info["git_branch"],
            "environment/git_dirty": git_info["git_dirty"],
        }

        try:
            self.clearml_task.connect(hyperparams, name="Experiment Configuration")
        except Exception as e:
            print(f"[Warning] Failed to connect configuration to ClearML: {e}")

    def _log_epoch_metrics(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        epoch: int,
    ) -> None:
        """Report epoch-level training and validation metrics to ClearML using consistent names."""
        if self.clearml_logger is None:
            return

        epoch_iter = epoch + 1
        try:
            # 1. Combined Total Loss (Train vs Val)
            if "loss_total" in train_metrics:
                self.clearml_logger.report_scalar(
                    title="Loss - Total",
                    series="train",
                    value=train_metrics["loss_total"],
                    iteration=epoch_iter,
                )
            if "val_loss_total" in val_metrics:
                self.clearml_logger.report_scalar(
                    title="Loss - Total",
                    series="val",
                    value=val_metrics["val_loss_total"],
                    iteration=epoch_iter,
                )

            # 2. All Eight Individual Loss Components
            loss_components = {
                "Loss - Box Regression": ("loss_box", "val_loss_box"),
                "Loss - GIoU": ("loss_giou", "val_loss_giou"),
                "Loss - Boundary": ("loss_bound", "val_loss_bound"),
                "Loss - Overlap": ("loss_overlap", "val_loss_overlap"),
                "Loss - Column Assignment": ("loss_col", "val_loss_col"),
                "Loss - Aspect Ratio": ("loss_aspect", "val_loss_aspect"),
                "Loss - Reading Order": ("loss_order", "val_loss_order"),
                "Loss - Macro Column": ("loss_macro", "val_loss_macro"),
            }
            for title, (tr_key, val_key) in loss_components.items():
                if tr_key in train_metrics:
                    self.clearml_logger.report_scalar(
                        title=title,
                        series="train",
                        value=train_metrics[tr_key],
                        iteration=epoch_iter,
                    )
                if val_key in val_metrics:
                    self.clearml_logger.report_scalar(
                        title=title,
                        series="val",
                        value=val_metrics[val_key],
                        iteration=epoch_iter,
                    )

            # 3. Evaluation Metrics
            eval_metric_specs = [
                ("Metrics - Alignment", "val_iou_mean", "val_iou_mean"),
                ("Metrics - Alignment", "val_giou_mean", "val_giou_mean"),
                ("Metrics - Geometric Constraints", "val_canvas_violation_rate", "val_canvas_violation_rate"),
                ("Metrics - Geometric Constraints", "val_unintended_overlap_rate", "val_unintended_overlap_rate"),
                ("Metrics - Structure", "val_column_accuracy", "val_column_accuracy"),
                ("Metrics - Structure", "val_reading_order_kendall_tau", "val_reading_order_kendall_tau"),
                ("Metrics - Template Baseline", "val_template_divergence_iou", "val_template_divergence_iou"),
                ("Metrics - Template Baseline", "val_template_baseline_gt_iou", "val_template_baseline_gt_iou"),
            ]
            for title, series, key in eval_metric_specs:
                if key in val_metrics:
                    self.clearml_logger.report_scalar(
                        title=title,
                        series=series,
                        value=val_metrics[key],
                        iteration=epoch_iter,
                    )

            # 4. Learning Rate
            if "learning_rate" in train_metrics:
                self.clearml_logger.report_scalar(
                    title="Hyperparameters",
                    series="learning_rate",
                    value=train_metrics["learning_rate"],
                    iteration=epoch_iter,
                )
        except Exception as e:
            print(f"[Warning] ClearML scalar reporting failed: {e}")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        accum_losses: dict[str, float] = {}
        num_batches = 0

        for batch in self.train_loader:
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

        self.scheduler.step()

        epoch_metrics = {k: v / max(1, num_batches) for k, v in accum_losses.items()}
        epoch_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
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

        return val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        filename: str = "latest_checkpoint.pt",
    ) -> Path:
        """Save local training checkpoint (model weights + optimizer state)."""
        ckpt_path = self.config.checkpoint_dir / filename
        clearml_tid = (
            str(self.clearml_task.id)
            if (self.clearml_task is not None and getattr(self.clearml_task, "id", None) is not None)
            else None
        )
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "clearml_task_id": clearml_tid,
        }
        torch.save(state, ckpt_path)

        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            return best_path

        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load state from local checkpoint."""
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

            self._log_epoch_metrics(train_metrics=train_m, val_metrics=val_m, epoch=epoch)

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

        if self.clearml_logger is not None:
            try:
                self.clearml_logger.flush()
            except Exception:
                pass

        if self._owns_clearml_task and self.clearml_task is not None:
            try:
                self.clearml_task.close()
            except Exception:
                pass

        return history
