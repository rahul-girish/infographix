"""Multi-Task Differentiable Losses for Retrieval-Augmented Neural Poster Layout Generation.

Implements:
  1. SmoothL1 Bounding Box Loss (L_box)
  2. Log-Dimension Anti-Collapse Loss (L_size)
  3. Generalized IoU Loss (L_giou)
  4. Differentiable Canvas Boundary Violation Penalty (L_bound)
  5. Scale-Invariant Pairwise IoU Overlap Penalty (L_overlap, exempting only explicit GT overlaps)
  6. Column-Assignment Cross-Entropy Loss (L_col)
  7. Intrinsic Aspect-Ratio Preservation Loss (L_aspect, masked for unavailable items)
  8. Continuous Reading-Order Pairwise Ranking Loss (L_order)
  9. Macro Column Planning Loss (L_macro)
"""

from __future__ import annotations

from dataclasses import dataclass

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

    inter_x0 = torch.max(x0_1, x0_2)
    inter_y0 = torch.max(y0_1, y0_2)
    inter_x1 = torch.min(x1_1, x1_2)
    inter_y1 = torch.min(y1_1, y1_2)

    inter_w = torch.clamp(inter_x1 - inter_x0, min=0.0)
    inter_h = torch.clamp(inter_y1 - inter_y0, min=0.0)
    inter_area = inter_w * inter_h

    area1 = torch.clamp(x1_1 - x0_1, min=0.0) * torch.clamp(y1_1 - y0_1, min=0.0)
    area2 = torch.clamp(x1_2 - x0_2, min=0.0) * torch.clamp(y1_2 - y0_2, min=0.0)
    union_area = area1 + area2 - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)

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

    w_box: float = 1.5
    w_size: float = 0.5
    w_giou: float = 1.5
    w_bound: float = 2.5
    w_overlap: float = 1.5
    w_col: float = 0.8
    w_aspect: float = 1.0
    w_order: float = 0.8
    w_macro: float = 0.4


class MultiTaskLayoutLoss(nn.Module):
    """Computes all individual layout loss components with strict masking."""

    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.weights = weights or LossWeights()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=0.05)
        self.size_l1 = nn.SmoothL1Loss(reduction="none", beta=0.1)
        self.fig_idx = CATEGORY_TO_IDX.get("figure", 4)
        self.cap_idx = CATEGORY_TO_IDX.get("caption", 6)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute all loss terms with strict padding masking and per-element normalization."""
        pred_boxes = predictions["pred_boxes"]                  # [B, M, 4]
        pred_col_logits = predictions["pred_column_logits"]     # [B, M, C]
        pred_orders = predictions["pred_order_scores"]          # [B, M]
        pred_macro = predictions["pred_macro_column_logits"]    # [B, C]

        target_boxes = batch["target_boxes"]                    # [B, M, 4]
        target_cols = batch["target_columns"]                   # [B, M]
        target_orders = batch["target_orders"]                  # [B, M]
        macro_targets = batch["macro_targets"]                  # [B]
        content_mask = batch["content_mask"]                    # [B, M] bool
        content_feats = batch["content_feats"]                  # [B, M, 18]

        B, M, _ = pred_boxes.shape
        mask_f = content_mask.float()
        num_valid_elements = torch.clamp(mask_f.sum(), min=1.0)

        # 1. SmoothL1 Bounding Box Loss (mean across 4 coords)
        l1_diff = self.smooth_l1(pred_boxes, target_boxes)  # [B, M, 4]
        loss_box = (l1_diff.mean(dim=-1) * mask_f).sum() / num_valid_elements

        # 2. Log-Dimension Loss (prevents shrinking to tiny degenerate boxes)
        pred_wh = torch.clamp(pred_boxes[..., 2:4], min=1e-3)
        target_wh = torch.clamp(target_boxes[..., 2:4], min=1e-3)
        log_wh_diff = self.size_l1(torch.log(pred_wh), torch.log(target_wh))
        loss_size = (log_wh_diff.mean(dim=-1) * mask_f).sum() / num_valid_elements

        # 3. GIoU Loss
        giou_vals = compute_giou(pred_boxes, target_boxes)  # [B, M]
        loss_giou = ((1.0 - giou_vals) * mask_f).sum() / num_valid_elements

        # 4. Canvas Boundary Violation Penalty
        px0, py0, pw, ph = pred_boxes.unbind(-1)
        px1 = px0 + pw
        py1 = py0 + ph

        bound_violation = (
            F.relu(-px0)
            + F.relu(-py0)
            + F.relu(px1 - 1.0)
            + F.relu(py1 - 1.0)
        )
        loss_bound = (bound_violation * mask_f).sum() / num_valid_elements

        # 5. Scale-Invariant Pairwise IoU Overlap Penalty
        # Excludes self-pairs and padded elements; only exempts explicit GT overlapping pairs.
        ix0 = torch.max(px0.unsqueeze(2), px0.unsqueeze(1))
        iy0 = torch.max(py0.unsqueeze(2), py0.unsqueeze(1))
        ix1 = torch.min(px1.unsqueeze(2), px1.unsqueeze(1))
        iy1 = torch.min(py1.unsqueeze(2), py1.unsqueeze(1))

        inter_w = torch.clamp(ix1 - ix0, min=0.0)
        inter_h = torch.clamp(iy1 - iy0, min=0.0)
        inter_area = inter_w * inter_h  # [B, M, M]

        areas = torch.clamp(pw, min=0.0) * torch.clamp(ph, min=0.0)
        union_area = areas.unsqueeze(2) + areas.unsqueeze(1) - inter_area
        pair_iou = inter_area / torch.clamp(union_area, min=1e-6)  # [B, M, M]

        valid_pair_mask = content_mask.unsqueeze(2) & content_mask.unsqueeze(1)
        diag_mask = torch.eye(M, device=pred_boxes.device, dtype=torch.bool).unsqueeze(0)
        pair_active = valid_pair_mask & (~diag_mask)

        exempt_mask = batch.get("target_overlap_exempt_mask")
        if exempt_mask is not None:
            exempt_mask = exempt_mask.to(device=pred_boxes.device, dtype=torch.bool)
            overlap_weight_matrix = torch.where(
                exempt_mask,
                torch.tensor(0.05, device=pred_boxes.device, dtype=pred_boxes.dtype),
                torch.tensor(1.0, device=pred_boxes.device, dtype=pred_boxes.dtype),
            )
        else:
            overlap_weight_matrix = torch.ones_like(pair_iou)

        per_element_overlap = (pair_iou * pair_active.float() * overlap_weight_matrix).sum(dim=-1)
        loss_overlap = (per_element_overlap * mask_f).sum() / num_valid_elements

        # 6. Column Assignment Cross-Entropy
        num_cols = pred_col_logits.shape[-1]
        safe_target_cols = target_cols.clamp(0, num_cols - 1)
        flat_logits = pred_col_logits.view(-1, num_cols)
        flat_targets = safe_target_cols.view(-1)
        flat_mask = mask_f.view(-1)

        ce_col = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss_col = (ce_col * flat_mask).sum() / num_valid_elements

        # 7. Intrinsic Aspect Ratio Preservation Loss (masked where unavailable)
        has_aspect = content_feats[..., 13] > 0.5
        intrinsic_aspect = content_feats[..., 12]
        valid_aspect_mask = has_aspect & content_mask

        pred_aspect = pw / torch.clamp(ph, min=1e-3)
        aspect_error = torch.abs(
            torch.log(torch.clamp(pred_aspect, min=0.1, max=10.0))
            - torch.log(torch.clamp(intrinsic_aspect, min=0.1, max=10.0))
        )
        num_valid_aspects = torch.clamp(valid_aspect_mask.float().sum(), min=1.0)
        loss_aspect = (aspect_error * valid_aspect_mask.float()).sum() / num_valid_aspects

        # 8. Pairwise Reading Order Ranking Loss
        ord_diff_target = target_orders.unsqueeze(2) - target_orders.unsqueeze(1)
        ord_diff_pred = pred_orders.unsqueeze(2) - pred_orders.unsqueeze(1)
        pairs_to_rank = pair_active & (ord_diff_target > 0)
        rank_loss = F.softplus(-ord_diff_pred)
        num_rank_pairs = torch.clamp(pairs_to_rank.float().sum(), min=1.0)
        loss_order = (rank_loss * pairs_to_rank.float()).sum() / num_rank_pairs

        # 9. Macro Column Planning Cross-Entropy
        safe_macro_targets = macro_targets.clamp(0, pred_macro.shape[-1] - 1)
        loss_macro = F.cross_entropy(pred_macro, safe_macro_targets)

        total_loss = (
            self.weights.w_box * loss_box
            + self.weights.w_size * loss_size
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
            "loss_size": float(loss_size.detach().cpu().item()),
            "loss_giou": float(loss_giou.detach().cpu().item()),
            "loss_bound": float(loss_bound.detach().cpu().item()),
            "loss_overlap": float(loss_overlap.detach().cpu().item()),
            "loss_col": float(loss_col.detach().cpu().item()),
            "loss_aspect": float(loss_aspect.detach().cpu().item()),
            "loss_order": float(loss_order.detach().cpu().item()),
            "loss_macro": float(loss_macro.detach().cpu().item()),
        }

        return total_loss, loss_dict
