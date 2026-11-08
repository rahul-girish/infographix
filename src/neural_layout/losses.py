"""Multi-Task Differentiable Losses for Retrieval-Augmented Neural Poster Layout Generation.

Implements:
  1. SmoothL1 Bounding Box Loss (L_box)
  2. Generalized IoU Loss (L_giou)
  3. Differentiable Canvas Boundary Violation Penalty (L_bound)
  4. Unintended Pairwise Overlap Penalty (L_overlap, ignoring legitimate figure-caption containment)
  5. Column-Assignment Cross-Entropy Loss (L_col)
  6. Intrinsic Aspect-Ratio Preservation Loss (L_aspect, masked for unavailable items)
  7. Continuous Reading-Order Pairwise Ranking Loss (L_order)
  8. Macro Column Planning Loss (L_macro)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neural_layout.dataset import CATEGORY_TO_IDX


def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x0, y0, w, h] to [x0, y0, x1, y1]."""
    x0, y0, w, h = boxes.unbind(-1)
    x1 = x0 + w
    y1 = y0 + h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def compute_giou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Compute Generalized IoU between predicted and target boxes of shape [..., 4]."""
    b1 = box_xywh_to_xyxy(pred_boxes)
    b2 = box_xywh_to_xyxy(target_boxes)

    x0_1, y0_1, x1_1, y1_1 = b1.unbind(-1)
    x0_2, y0_2, x1_2, y1_2 = b2.unbind(-1)

    # Intersection
    inter_x0 = torch.max(x0_1, x0_2)
    inter_y0 = torch.max(y0_1, y0_2)
    inter_x1 = torch.min(x1_1, x1_2)
    inter_y1 = torch.min(y1_1, y1_2)

    inter_w = torch.clamp(inter_x1 - inter_x0, min=0.0)
    inter_h = torch.clamp(inter_y1 - inter_y0, min=0.0)
    inter_area = inter_w * inter_h

    # Areas and Union
    area1 = torch.clamp(x1_1 - x0_1, min=0.0) * torch.clamp(y1_1 - y0_1, min=0.0)
    area2 = torch.clamp(x1_2 - x0_2, min=0.0) * torch.clamp(y1_2 - y0_2, min=0.0)
    union_area = area1 + area2 - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    # Smallest enclosing convex box
    enc_x0 = torch.min(x0_1, x0_2)
    enc_y0 = torch.min(y0_1, y0_2)
    enc_x1 = torch.max(x1_1, x1_2)
    enc_y1 = torch.max(y1_1, y1_2)
    enc_w = torch.clamp(enc_x1 - enc_x0, min=0.0)
    enc_h = torch.clamp(enc_y1 - enc_y0, min=0.0)
    enc_area = enc_w * enc_h

    giou = iou - (torch.clamp(enc_area - union_area, min=0.0) / torch.clamp(enc_area, min=1e-6))
    return giou


@dataclass
class LossWeights:
    """Configurable weights for the multi-task layout loss."""

    w_box: float = 1.0
    w_giou: float = 1.5
    w_bound: float = 2.5      # Hard boundary enforcement
    w_overlap: float = 2.0    # Hard collision prevention
    w_col: float = 1.0
    w_aspect: float = 1.2
    w_order: float = 0.8
    w_macro: float = 0.5


class MultiTaskLayoutLoss(nn.Module):
    """Computes all individual layout loss components with strict masking."""

    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.weights = weights or LossWeights()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=0.05)
        self.fig_idx = CATEGORY_TO_IDX.get("figure", 4)
        self.cap_idx = CATEGORY_TO_IDX.get("caption", 6)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute all loss terms.

        Args:
            predictions: Output dict from TemplateConditionedLayoutModel.
            batch: Collated batch dictionary from DataLoader.

        Returns:
            total_loss: Scalar torch.Tensor with backward gradient.
            loss_dict: Dictionary mapping metric name to python float for Aim logging.
        """
        pred_boxes = predictions["pred_boxes"]               # [B, M, 4]
        pred_col_logits = predictions["pred_column_logits"]  # [B, M, 5]
        pred_orders = predictions["pred_order_scores"]       # [B, M]
        pred_macro = predictions["pred_macro_column_logits"] # [B, 5]

        target_boxes = batch["target_boxes"]                 # [B, M, 4]
        target_cols = batch["target_columns"]               # [B, M]
        target_orders = batch["target_orders"]               # [B, M]
        macro_targets = batch["macro_targets"]               # [B]
        content_mask = batch["content_mask"]                 # [B, M] bool
        content_feats = batch["content_feats"]               # [B, M, 18]

        B, M, _ = pred_boxes.shape
        num_valid_elements = torch.clamp(content_mask.sum().float(), min=1.0)

        # -------------------------------------------------------------
        # 1. SmoothL1 Bounding Box Loss
        # -------------------------------------------------------------
        l1_diff = self.smooth_l1(pred_boxes, target_boxes)  # [B, M, 4]
        loss_box = (l1_diff.sum(dim=-1) * content_mask.float()).sum() / num_valid_elements

        # -------------------------------------------------------------
        # 2. GIoU Loss
        # -------------------------------------------------------------
        giou_vals = compute_giou(pred_boxes, target_boxes)  # [B, M]
        loss_giou = ((1.0 - giou_vals) * content_mask.float()).sum() / num_valid_elements

        # -------------------------------------------------------------
        # 3. Canvas Boundary Violation Penalty (Hard Constraint)
        # -------------------------------------------------------------
        # Normalized poster bounds are [0, 1]
        px0 = pred_boxes[..., 0]
        py0 = pred_boxes[..., 1]
        pw = pred_boxes[..., 2]
        ph = pred_boxes[..., 3]
        px1 = px0 + pw
        py1 = py0 + ph

        # Penalize protruding outside [0, 1] canvas boundaries
        bound_violation = (
            F.relu(-px0)
            + F.relu(-py0)
            + F.relu(px1 - 1.0)
            + F.relu(py1 - 1.0)
        )  # [B, M]
        loss_bound = (bound_violation * content_mask.float()).sum() / num_valid_elements

        # -------------------------------------------------------------
        # 4. Pairwise Overlap Penalty (excluding legitimate figure-caption containment)
        # -------------------------------------------------------------
        # Calculate pairwise IoU between all (i, j) where i != j in each poster
        # px0: [B, M, 1], px0_t: [B, 1, M]
        ix0 = torch.max(px0.unsqueeze(2), px0.unsqueeze(1))
        iy0 = torch.max(py0.unsqueeze(2), py0.unsqueeze(1))
        ix1 = torch.min(px1.unsqueeze(2), px1.unsqueeze(1))
        iy1 = torch.min(py1.unsqueeze(2), py1.unsqueeze(1))

        inter_w = torch.clamp(ix1 - ix0, min=0.0)
        inter_h = torch.clamp(iy1 - iy0, min=0.0)
        inter_area = inter_w * inter_h  # [B, M, M]

        # Categories for legitimate overlap filtering
        # content_feats[..., 0..8] is one-hot category
        cat_indices = torch.argmax(content_feats[..., :9], dim=-1)  # [B, M]
        cat_i = cat_indices.unsqueeze(2)  # [B, M, 1]
        cat_j = cat_indices.unsqueeze(1)  # [B, 1, M]

        # Mask out diagonal (self-overlap) and invalid padded elements
        valid_pair_mask = content_mask.unsqueeze(2) & content_mask.unsqueeze(1)  # [B, M, M]
        diag_mask = torch.eye(M, device=pred_boxes.device, dtype=torch.bool).unsqueeze(0)
        pair_active = valid_pair_mask & (~diag_mask)

        # Allow legitimate overlaps: figure + caption pairs
        is_fig_cap = (
            ((cat_i == self.fig_idx) & (cat_j == self.cap_idx))
            | ((cat_i == self.cap_idx) & (cat_j == self.fig_idx))
        )
        # Suppress overlap penalty for legitimate figure-caption pairs
        overlap_weight_matrix = torch.where(is_fig_cap, torch.tensor(0.05, device=pred_boxes.device), torch.tensor(1.0, device=pred_boxes.device))

        penalized_overlap = inter_area * pair_active.float() * overlap_weight_matrix
        num_valid_pairs = torch.clamp(pair_active.float().sum(), min=1.0)
        loss_overlap = penalized_overlap.sum() / num_valid_pairs

        # -------------------------------------------------------------
        # 5. Column Assignment Cross-Entropy
        # -------------------------------------------------------------
        flat_logits = pred_col_logits.view(-1, pred_col_logits.shape[-1])
        flat_targets = target_cols.view(-1)
        flat_mask = content_mask.view(-1).float()

        ce_col = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss_col = (ce_col * flat_mask).sum() / num_valid_elements

        # -------------------------------------------------------------
        # 6. Intrinsic Aspect Ratio Preservation Loss
        # -------------------------------------------------------------
        has_aspect = content_feats[..., 13] > 0.5  # [B, M] bool
        intrinsic_aspect = content_feats[..., 12]  # [B, M] float
        valid_aspect_mask = has_aspect & content_mask

        pred_aspect = pw / torch.clamp(ph, min=1e-3)
        aspect_error = torch.abs(torch.log(torch.clamp(pred_aspect, min=0.1, max=10.0)) - torch.log(torch.clamp(intrinsic_aspect, min=0.1, max=10.0)))
        num_valid_aspects = torch.clamp(valid_aspect_mask.float().sum(), min=1.0)
        loss_aspect = (aspect_error * valid_aspect_mask.float()).sum() / num_valid_aspects

        # -------------------------------------------------------------
        # 7. Pairwise Reading Order Ranking Loss
        # -------------------------------------------------------------
        # For valid pairs (i, j) where target_orders[i] < target_orders[j],
        # we want pred_orders[j] > pred_orders[i] with margin
        ord_diff_target = target_orders.unsqueeze(2) - target_orders.unsqueeze(1)  # [B, M, M] (j - i)
        ord_diff_pred = pred_orders.unsqueeze(2) - pred_orders.unsqueeze(1)        # [B, M, M] (j - i)

        # Pair condition: j comes after i in true reading order
        pairs_to_rank = pair_active & (ord_diff_target > 0)
        # Margin ranking loss: log(1 + exp(-(pred_j - pred_i)))
        rank_loss = F.softplus(-ord_diff_pred)
        num_rank_pairs = torch.clamp(pairs_to_rank.float().sum(), min=1.0)
        loss_order = (rank_loss * pairs_to_rank.float()).sum() / num_rank_pairs

        # -------------------------------------------------------------
        # 8. Macro Column Planning Cross-Entropy
        # -------------------------------------------------------------
        loss_macro = F.cross_entropy(pred_macro, macro_targets)

        # -------------------------------------------------------------
        # Weighted Total Multi-Task Loss
        # -------------------------------------------------------------
        total_loss = (
            self.weights.w_box * loss_box
            + self.weights.w_giou * loss_giou
            + self.weights.w_bound * loss_bound
            + self.weights.w_overlap * loss_overlap
            + self.weights.w_col * loss_col
            + self.weights.w_aspect * loss_aspect
            + self.weights.w_order * loss_order
            + self.weights.w_macro * loss_macro
        )

        loss_dict = {
            "loss_total": float(total_loss.detach().cpu().item()),
            "loss_box": float(loss_box.detach().cpu().item()),
            "loss_giou": float(loss_giou.detach().cpu().item()),
            "loss_bound": float(loss_bound.detach().cpu().item()),
            "loss_overlap": float(loss_overlap.detach().cpu().item()),
            "loss_col": float(loss_col.detach().cpu().item()),
            "loss_aspect": float(loss_aspect.detach().cpu().item()),
            "loss_order": float(loss_order.detach().cpu().item()),
            "loss_macro": float(loss_macro.detach().cpu().item()),
        }

        return total_loss, loss_dict
