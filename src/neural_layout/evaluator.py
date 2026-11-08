"""Multi-Metric Evaluator for Neural Layout Generation.

Evaluates:
  - Bounding-box IoU and GIoU
  - Canvas-boundary violation rate
  - Unintended overlap rate and area (excluding legitimate figure-caption pairs)
  - Column assignment accuracy
  - Reading order Kendall's Tau rank correlation
  - Intrinsic aspect ratio fidelity
  - Dual Baseline Comparison: Ground Truth vs. Neural Prediction vs. Retrieved Template Baseline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.stats as stats
import torch

from src.neural_layout.dataset import CATEGORY_TO_IDX, KNOWN_CATEGORIES
from src.neural_layout.losses import box_xywh_to_xyxy, compute_giou


class LayoutEvaluator:
    """Evaluates neural layout generation against ground truth and retrieved template baselines."""

    def __init__(self):
        self.fig_idx = CATEGORY_TO_IDX.get("figure", 4)
        self.cap_idx = CATEGORY_TO_IDX.get("caption", 6)

    @torch.no_grad()
    def evaluate_batch(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Evaluate a single batch of predictions.

        Returns comprehensive metric dictionary.
        """
        pred_boxes = predictions["pred_boxes"].cpu()               # [B, M, 4]
        pred_col_logits = predictions["pred_column_logits"].cpu()  # [B, M, 5]
        pred_orders = predictions["pred_order_scores"].cpu()       # [B, M]
        pred_macro = predictions["pred_macro_column_logits"].cpu() # [B, 5]

        target_boxes = batch["target_boxes"].cpu()                 # [B, M, 4]
        target_cols = batch["target_columns"].cpu()               # [B, M]
        target_orders = batch["target_orders"].cpu()               # [B, M]
        macro_targets = batch["macro_targets"].cpu()               # [B]
        content_mask = batch["content_mask"].cpu()                 # [B, M] bool
        content_feats = batch["content_feats"].cpu()               # [B, M, 18]
        template_feats = batch["template_feats"].cpu()             # [B, K, 16]
        template_mask = batch["template_mask"].cpu()               # [B, K] bool

        B, M, _ = pred_boxes.shape

        ious_list: list[float] = []
        gious_list: list[float] = []
        boundary_violations_list: list[float] = []
        col_correct_list: list[float] = []
        order_taus_list: list[float] = []
        aspect_errors_list: list[float] = []

        # Overlap tracking
        overlap_pair_indicators: list[float] = []
        overlap_areas: list[float] = []

        # Template baseline comparisons
        template_divergence_ious: list[float] = []
        template_vs_gt_ious: list[float] = []

        for b in range(B):
            mask_b = content_mask[b]
            valid_indices = torch.nonzero(mask_b, as_tuple=True)[0]
            m_valid = len(valid_indices)
            if m_valid == 0:
                continue

            p_boxes = pred_boxes[b, valid_indices]        # [m_valid, 4]
            t_boxes = target_boxes[b, valid_indices]      # [m_valid, 4]
            p_cols = torch.argmax(pred_col_logits[b, valid_indices], dim=-1) # [m_valid]
            t_cols = target_cols[b, valid_indices]        # [m_valid]
            p_ord = pred_orders[b, valid_indices]         # [m_valid]
            t_ord = target_orders[b, valid_indices]       # [m_valid]
            c_feats = content_feats[b, valid_indices]     # [m_valid, 18]

            # 1. IoU and GIoU vs Ground Truth
            giou_vals = compute_giou(p_boxes, t_boxes)
            # Standard IoU:
            b1 = box_xywh_to_xyxy(p_boxes)
            b2 = box_xywh_to_xyxy(t_boxes)
            ix0 = torch.max(b1[:, 0], b2[:, 0])
            iy0 = torch.max(b1[:, 1], b2[:, 1])
            ix1 = torch.min(b1[:, 2], b2[:, 2])
            iy1 = torch.min(b1[:, 3], b2[:, 3])
            inter_a = torch.clamp(ix1 - ix0, min=0.0) * torch.clamp(iy1 - iy0, min=0.0)
            area1 = p_boxes[:, 2] * p_boxes[:, 3]
            area2 = t_boxes[:, 2] * t_boxes[:, 3]
            union_a = area1 + area2 - inter_a
            sample_ious = inter_a / torch.clamp(union_a, min=1e-6)

            ious_list.extend(sample_ious.tolist())
            gious_list.extend(giou_vals.tolist())

            # 2. Canvas Boundary Violation Rate (overflow outside [0, 1] by > 0.01)
            px0, py0, pw, ph = p_boxes.unbind(-1)
            px1 = px0 + pw
            py1 = py0 + ph
            violates = (px0 < -0.01) | (py0 < -0.01) | (px1 > 1.01) | (py1 > 1.01)
            boundary_violations_list.extend(violates.float().tolist())

            # 3. Column Assignment Accuracy
            col_match = (p_cols == t_cols).float()
            col_correct_list.extend(col_match.tolist())

            # 4. Reading Order Kendall's Tau
            if m_valid >= 2:
                tau, _ = stats.kendalltau(p_ord.numpy(), t_ord.numpy())
                if not math.isnan(tau):
                    order_taus_list.append(tau)

            # 5. Aspect Ratio Preservation for intrinsic figures
            has_aspect = c_feats[:, 13] > 0.5
            if has_aspect.any():
                intr_aspect = c_feats[has_aspect, 12]
                pred_aspect = pw[has_aspect] / torch.clamp(ph[has_aspect], min=1e-3)
                log_err = torch.abs(torch.log(torch.clamp(pred_aspect, min=0.1, max=10.0)) - torch.log(torch.clamp(intr_aspect, min=0.1, max=10.0)))
                aspect_errors_list.extend(log_err.tolist())

            # 6. Unintended Overlaps (IoU > 0.05 between non-figure-caption elements)
            cats = torch.argmax(c_feats[:, :9], dim=-1)
            for i in range(m_valid):
                for j in range(i + 1, m_valid):
                    # Check if legitimate figure-caption pair
                    is_fig_cap = (
                        (cats[i] == self.fig_idx and cats[j] == self.cap_idx)
                        or (cats[i] == self.cap_idx and cats[j] == self.fig_idx)
                    )
                    if is_fig_cap:
                        continue  # Legitimate containment

                    box_i = b1[i]
                    box_j = b1[j]
                    o_x0 = max(float(box_i[0]), float(box_j[0]))
                    o_y0 = max(float(box_i[1]), float(box_j[1]))
                    o_x1 = min(float(box_i[2]), float(box_j[2]))
                    o_y1 = min(float(box_i[3]), float(box_j[3]))

                    if o_x1 > o_x0 and o_y1 > o_y0:
                        inter_ij = (o_x1 - o_x0) * (o_y1 - o_y0)
                        u_ij = float(area1[i] + area1[j]) - inter_ij
                        iou_ij = inter_ij / max(1e-6, u_ij)
                        overlap_areas.append(inter_ij)
                        overlap_pair_indicators.append(1.0 if iou_ij > 0.05 else 0.0)
                    else:
                        overlap_pair_indicators.append(0.0)

            # 7. Comparison with Retrieved Template Slots
            t_mask_b = template_mask[b]
            k_valid = torch.nonzero(t_mask_b, as_tuple=True)[0]
            if len(k_valid) > 0:
                tmpl_boxes = template_feats[b, k_valid, 9:13]  # [k_valid, 4] [x0, y0, w, h]
                # Compare neural predictions against nearest template slots (Template Divergence)
                # Divergence proves the model adapts rather than memorizing template slots
                min_m_k = min(m_valid, len(k_valid))
                p_sub = p_boxes[:min_m_k]
                t_sub = tmpl_boxes[:min_m_k]
                gt_sub = t_boxes[:min_m_k]

                # Neural vs Template overlap
                b_p = box_xywh_to_xyxy(p_sub)
                b_t = box_xywh_to_xyxy(t_sub)
                b_gt = box_xywh_to_xyxy(gt_sub)

                inter_pt = torch.clamp(torch.min(b_p[:, 2], b_t[:, 2]) - torch.max(b_p[:, 0], b_t[:, 0]), min=0.0) * \
                           torch.clamp(torch.min(b_p[:, 3], b_t[:, 3]) - torch.max(b_p[:, 1], b_t[:, 1]), min=0.0)
                u_pt = (p_sub[:, 2] * p_sub[:, 3]) + (t_sub[:, 2] * t_sub[:, 3]) - inter_pt
                template_divergence_ious.extend((inter_pt / torch.clamp(u_pt, min=1e-6)).tolist())

                # Template baseline vs Ground Truth
                inter_tgt = torch.clamp(torch.min(b_t[:, 2], b_gt[:, 2]) - torch.max(b_t[:, 0], b_gt[:, 0]), min=0.0) * \
                            torch.clamp(torch.min(b_t[:, 3], b_gt[:, 3]) - torch.max(b_t[:, 1], b_gt[:, 1]), min=0.0)
                u_tgt = (t_sub[:, 2] * t_sub[:, 3]) + (gt_sub[:, 2] * gt_sub[:, 3]) - inter_tgt
                template_vs_gt_ious.extend((inter_tgt / torch.clamp(u_tgt, min=1e-6)).tolist())

        # Macro accuracy
        pred_macro_cls = torch.argmax(pred_macro, dim=-1)
        macro_acc = (pred_macro_cls == macro_targets).float().mean().item()

        return {
            "iou_mean": float(np.mean(ious_list)) if ious_list else 0.0,
            "giou_mean": float(np.mean(gious_list)) if gious_list else 0.0,
            "canvas_violation_rate": float(np.mean(boundary_violations_list)) if boundary_violations_list else 0.0,
            "unintended_overlap_rate": float(np.mean(overlap_pair_indicators)) if overlap_pair_indicators else 0.0,
            "unintended_overlap_mean_area": float(np.mean(overlap_areas)) if overlap_areas else 0.0,
            "column_accuracy": float(np.mean(col_correct_list)) if col_correct_list else 0.0,
            "reading_order_kendall_tau": float(np.mean(order_taus_list)) if order_taus_list else 0.0,
            "aspect_ratio_log_error": float(np.mean(aspect_errors_list)) if aspect_errors_list else 0.0,
            "macro_column_accuracy": macro_acc,
            "template_divergence_iou": float(np.mean(template_divergence_ious)) if template_divergence_ious else 0.0,
            "template_baseline_gt_iou": float(np.mean(template_vs_gt_ious)) if template_vs_gt_ious else 0.0,
        }
