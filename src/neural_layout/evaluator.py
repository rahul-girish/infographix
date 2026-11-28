"""Multi-Metric Evaluator for Neural Layout Generation.

Evaluates:
  - Bounding-box IoU, GIoU, center L1 error, and width/height L1 error
  - Canvas-boundary violation rate
  - Unintended overlap rate, mean pairwise IoU, and overlap area (exempting only explicit GT overlaps)
  - Column assignment accuracy and macro column accuracy
  - Reading order Kendall's Tau rank correlation
  - Intrinsic aspect ratio fidelity
  - Collapse diagnostics (center std, width/height std, near-identical box rate)
  - Dual Baseline Comparison: Ground Truth vs. Neural Prediction vs. Retrieved Template Baseline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import scipy.stats as stats
import torch

from src.neural_layout.losses import box_xywh_to_xyxy, compute_giou


def _pairwise_iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute [N, M] pairwise IoU between two [x0, y0, w, h] box tensors."""
    a_xyxy = box_xywh_to_xyxy(boxes_a)
    b_xyxy = box_xywh_to_xyxy(boxes_b)
    ix0 = torch.max(a_xyxy[:, None, 0], b_xyxy[None, :, 0])
    iy0 = torch.max(a_xyxy[:, None, 1], b_xyxy[None, :, 1])
    ix1 = torch.min(a_xyxy[:, None, 2], b_xyxy[None, :, 2])
    iy1 = torch.min(a_xyxy[:, None, 3], b_xyxy[None, :, 3])
    inter = torch.clamp(ix1 - ix0, min=0.0) * torch.clamp(iy1 - iy0, min=0.0)
    area_a = torch.clamp(boxes_a[:, 2], min=0.0) * torch.clamp(boxes_a[:, 3], min=0.0)
    area_b = torch.clamp(boxes_b[:, 2], min=0.0) * torch.clamp(boxes_b[:, 3], min=0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / torch.clamp(union, min=1e-6)


def compute_within_category_hungarian_ious(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    cat_ids: torch.Tensor,
) -> list[float]:
    """Compute optimal within-category bipartite-matched IoUs for a single poster."""
    matched_ious: list[float] = []
    for cat_val in torch.unique(cat_ids).tolist():
        idx = torch.nonzero(cat_ids == cat_val, as_tuple=True)[0]
        if len(idx) == 0:
            continue
        p_sub = pred_boxes[idx]
        t_sub = target_boxes[idx]
        iou_mat = _pairwise_iou_matrix(p_sub, t_sub).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
        for r, c in zip(row_ind, col_ind):
            matched_ious.append(float(iou_mat[r, c]))
    return matched_ious


class LayoutEvaluator:
    """Evaluates neural layout generation against ground truth and retrieved template baselines."""

    @torch.no_grad()
    def evaluate_batch(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Evaluate a single batch of predictions and return comprehensive metric dictionary."""
        pred_boxes = predictions["pred_boxes"].detach().cpu()               # [B, M, 4]
        pred_col_logits = predictions["pred_column_logits"].detach().cpu()  # [B, M, C]
        pred_orders = predictions["pred_order_scores"].detach().cpu()       # [B, M]
        pred_macro = predictions["pred_macro_column_logits"].detach().cpu() # [B, C]

        target_boxes = batch["target_boxes"].detach().cpu()                 # [B, M, 4]
        target_cols = batch["target_columns"].detach().cpu()               # [B, M]
        target_orders = batch["target_orders"].detach().cpu()               # [B, M]
        macro_targets = batch["macro_targets"].detach().cpu()               # [B]
        content_mask = batch["content_mask"].detach().cpu()                 # [B, M] bool
        content_feats = batch["content_feats"].detach().cpu()               # [B, M, 18]
        template_feats = batch["template_feats"].detach().cpu()             # [B, K, 16]
        template_mask = batch["template_mask"].detach().cpu()               # [B, K] bool
        exempt_mask_batch = batch.get("target_overlap_exempt_mask")
        if exempt_mask_batch is not None:
            exempt_mask_batch = exempt_mask_batch.detach().cpu()

        B = pred_boxes.shape[0]

        ious_list: list[float] = []
        hungarian_ious_list: list[float] = []
        gious_list: list[float] = []
        center_err_list: list[float] = []
        wh_err_list: list[float] = []
        boundary_violations_list: list[float] = []
        col_correct_list: list[float] = []
        order_taus_list: list[float] = []
        aspect_errors_list: list[float] = []

        overlap_pair_indicators: list[float] = []
        overlap_ious_all: list[float] = []
        overlap_areas: list[float] = []
        near_identical_indicators: list[float] = []

        center_std_x_list: list[float] = []
        center_std_y_list: list[float] = []
        wh_std_w_list: list[float] = []
        wh_std_h_list: list[float] = []

        template_divergence_ious: list[float] = []
        template_vs_gt_ious: list[float] = []

        for b in range(B):
            mask_b = content_mask[b]
            valid_indices = torch.nonzero(mask_b, as_tuple=True)[0]
            m_valid = len(valid_indices)
            if m_valid == 0:
                continue

            p_boxes = pred_boxes[b, valid_indices]
            t_boxes = target_boxes[b, valid_indices]
            p_cols = torch.argmax(pred_col_logits[b, valid_indices], dim=-1)
            t_cols = target_cols[b, valid_indices]
            p_ord = pred_orders[b, valid_indices]
            t_ord = target_orders[b, valid_indices]
            c_feats = content_feats[b, valid_indices]

            # 1. IoU, GIoU, Center & WH errors vs Ground Truth
            giou_vals = compute_giou(p_boxes, t_boxes)
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

            # 1b. Within-category Hungarian-matched IoU (permutation-invariant set alignment)
            cat_ids = torch.argmax(c_feats[:, :9], dim=-1)
            hungarian_ious_list.extend(
                compute_within_category_hungarian_ious(p_boxes, t_boxes, cat_ids)
            )

            p_cx = p_boxes[:, 0] + 0.5 * p_boxes[:, 2]
            p_cy = p_boxes[:, 1] + 0.5 * p_boxes[:, 3]
            t_cx = t_boxes[:, 0] + 0.5 * t_boxes[:, 2]
            t_cy = t_boxes[:, 1] + 0.5 * t_boxes[:, 3]
            c_err = 0.5 * (torch.abs(p_cx - t_cx) + torch.abs(p_cy - t_cy))
            wh_err = 0.5 * (torch.abs(p_boxes[:, 2] - t_boxes[:, 2]) + torch.abs(p_boxes[:, 3] - t_boxes[:, 3]))
            center_err_list.extend(c_err.tolist())
            wh_err_list.extend(wh_err.tolist())

            if m_valid >= 2:
                center_std_x_list.append(float(p_cx.std(unbiased=False).item()))
                center_std_y_list.append(float(p_cy.std(unbiased=False).item()))
                wh_std_w_list.append(float(p_boxes[:, 2].std(unbiased=False).item()))
                wh_std_h_list.append(float(p_boxes[:, 3].std(unbiased=False).item()))

            # 2. Canvas Boundary Violation Rate
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
                log_err = torch.abs(
                    torch.log(torch.clamp(pred_aspect, min=0.1, max=10.0))
                    - torch.log(torch.clamp(intr_aspect, min=0.1, max=10.0))
                )
                aspect_errors_list.extend(log_err.tolist())

            # 6. Pairwise Unintended Overlaps & Near-Identical Box Collapse
            exempt_b = exempt_mask_batch[b] if exempt_mask_batch is not None else None
            for idx_i in range(m_valid):
                orig_i = int(valid_indices[idx_i])
                for idx_j in range(idx_i + 1, m_valid):
                    orig_j = int(valid_indices[idx_j])
                    max_coord_diff = float(torch.max(torch.abs(p_boxes[idx_i] - p_boxes[idx_j])).item())
                    near_identical_indicators.append(1.0 if max_coord_diff < 0.02 else 0.0)

                    if exempt_b is not None and bool(exempt_b[orig_i, orig_j]):
                        continue

                    box_i = b1[idx_i]
                    box_j = b1[idx_j]
                    o_x0 = max(float(box_i[0]), float(box_j[0]))
                    o_y0 = max(float(box_i[1]), float(box_j[1]))
                    o_x1 = min(float(box_i[2]), float(box_j[2]))
                    o_y1 = min(float(box_i[3]), float(box_j[3]))

                    if o_x1 > o_x0 and o_y1 > o_y0:
                        inter_ij = (o_x1 - o_x0) * (o_y1 - o_y0)
                        u_ij = float(area1[idx_i] + area1[idx_j]) - inter_ij
                        iou_ij = inter_ij / max(1e-6, u_ij)
                        overlap_areas.append(inter_ij)
                        overlap_ious_all.append(iou_ij)
                        overlap_pair_indicators.append(1.0 if iou_ij > 0.05 else 0.0)
                    else:
                        overlap_ious_all.append(0.0)
                        overlap_pair_indicators.append(0.0)

            # 7. Comparison with Retrieved Template Slots
            t_mask_b = template_mask[b]
            k_valid = torch.nonzero(t_mask_b, as_tuple=True)[0]
            if len(k_valid) > 0:
                tmpl_boxes = template_feats[b, k_valid, 9:13]
                min_m_k = min(m_valid, len(k_valid))
                p_sub = p_boxes[:min_m_k]
                t_sub = tmpl_boxes[:min_m_k]
                gt_sub = t_boxes[:min_m_k]

                b_p = box_xywh_to_xyxy(p_sub)
                b_t = box_xywh_to_xyxy(t_sub)
                b_gt = box_xywh_to_xyxy(gt_sub)

                inter_pt = (
                    torch.clamp(torch.min(b_p[:, 2], b_t[:, 2]) - torch.max(b_p[:, 0], b_t[:, 0]), min=0.0)
                    * torch.clamp(torch.min(b_p[:, 3], b_t[:, 3]) - torch.max(b_p[:, 1], b_t[:, 1]), min=0.0)
                )
                u_pt = (p_sub[:, 2] * p_sub[:, 3]) + (t_sub[:, 2] * t_sub[:, 3]) - inter_pt
                template_divergence_ious.extend((inter_pt / torch.clamp(u_pt, min=1e-6)).tolist())

                inter_tgt = (
                    torch.clamp(torch.min(b_t[:, 2], b_gt[:, 2]) - torch.max(b_t[:, 0], b_gt[:, 0]), min=0.0)
                    * torch.clamp(torch.min(b_t[:, 3], b_gt[:, 3]) - torch.max(b_t[:, 1], b_gt[:, 1]), min=0.0)
                )
                u_tgt = (t_sub[:, 2] * t_sub[:, 3]) + (gt_sub[:, 2] * gt_sub[:, 3]) - inter_tgt
                template_vs_gt_ious.extend((inter_tgt / torch.clamp(u_tgt, min=1e-6)).tolist())

        pred_macro_cls = torch.argmax(pred_macro, dim=-1)
        macro_acc = (pred_macro_cls == macro_targets).float().mean().item()

        return {
            "iou_mean": float(np.mean(ious_list)) if ious_list else 0.0,
            "hungarian_iou_mean": float(np.mean(hungarian_ious_list)) if hungarian_ious_list else 0.0,
            "giou_mean": float(np.mean(gious_list)) if gious_list else 0.0,
            "center_l1_error": float(np.mean(center_err_list)) if center_err_list else 0.0,
            "wh_l1_error": float(np.mean(wh_err_list)) if wh_err_list else 0.0,
            "canvas_violation_rate": float(np.mean(boundary_violations_list)) if boundary_violations_list else 0.0,
            "unintended_overlap_rate": float(np.mean(overlap_pair_indicators)) if overlap_pair_indicators else 0.0,
            "unintended_overlap_mean_area": float(np.mean(overlap_areas)) if overlap_areas else 0.0,
            "pairwise_iou_mean": float(np.mean(overlap_ious_all)) if overlap_ious_all else 0.0,
            "near_identical_box_rate": float(np.mean(near_identical_indicators)) if near_identical_indicators else 0.0,
            "pred_center_std_x": float(np.mean(center_std_x_list)) if center_std_x_list else 0.0,
            "pred_center_std_y": float(np.mean(center_std_y_list)) if center_std_y_list else 0.0,
            "pred_wh_std_w": float(np.mean(wh_std_w_list)) if wh_std_w_list else 0.0,
            "pred_wh_std_h": float(np.mean(wh_std_h_list)) if wh_std_h_list else 0.0,
            "column_accuracy": float(np.mean(col_correct_list)) if col_correct_list else 0.0,
            "reading_order_kendall_tau": float(np.mean(order_taus_list)) if order_taus_list else 0.0,
            "aspect_ratio_log_error": float(np.mean(aspect_errors_list)) if aspect_errors_list else 0.0,
            "macro_column_accuracy": macro_acc,
            "template_divergence_iou": float(np.mean(template_divergence_ious)) if template_divergence_ious else 0.0,
            "template_baseline_gt_iou": float(np.mean(template_vs_gt_ious)) if template_vs_gt_ious else 0.0,
        }


def evaluate_category_mean_baseline(
    train_elements_df: pd.DataFrame,
    eval_dataset: Any,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Trivial baseline: predict each category's training-set mean box and majority column."""
    from src.neural_layout.dataset import KNOWN_CATEGORIES, collate_layout_batch

    cat_to_idx = {c: i for i, c in enumerate(KNOWN_CATEGORIES)}
    df = train_elements_df.copy()
    if "category_slug" in df.columns:
        cat_col = df["category_slug"].astype(str).str.lower()
    elif "category_name" in df.columns:
        cat_col = df["category_name"].astype(str).str.lower()
    else:
        cat_col = pd.Series(["unknown"] * len(df))

    x_col = "x0" if "x0" in df.columns else "x0_norm"
    y_col = "y0" if "y0" in df.columns else "y0_norm"
    w_col = "w" if "w" in df.columns else "w_norm"
    h_col = "h" if "h" in df.columns else "h_norm"

    mean_box_by_cat: dict[int, list[float]] = {}
    global_mean_box = [
        float(df[x_col].mean()),
        float(df[y_col].mean()),
        float(df[w_col].mean()),
        float(df[h_col].mean()),
    ]
    for cat_name, cat_idx in cat_to_idx.items():
        sub = df[cat_col == cat_name]
        if len(sub) > 0:
            mean_box_by_cat[cat_idx] = [
                float(sub[x_col].mean()),
                float(sub[y_col].mean()),
                float(sub[w_col].mean()),
                float(sub[h_col].mean()),
            ]
        else:
            mean_box_by_cat[cat_idx] = global_mean_box

    # Majority column class across training elements
    if "column_id" in df.columns:
        maj_col = int(df["column_id"].clip(0, 4).mode().iloc[0])
    else:
        maj_col = 0

    evaluator = LayoutEvaluator()
    n = len(eval_dataset) if max_samples is None else min(len(eval_dataset), max_samples)
    samples = [eval_dataset[i] for i in range(n)]
    batch = collate_layout_batch(samples)

    B, M, _ = batch["content_feats"].shape
    pred_boxes = torch.zeros((B, M, 4), dtype=torch.float32)
    pred_col_logits = torch.zeros((B, M, 5), dtype=torch.float32)
    pred_col_logits[:, :, maj_col] = 5.0
    pred_orders = torch.linspace(0.0, 1.0, steps=M, dtype=torch.float32).unsqueeze(0).expand(B, M).clone()
    pred_macro = torch.zeros((B, 5), dtype=torch.float32)
    pred_macro[:, 2] = 5.0

    cat_ids = torch.argmax(batch["content_feats"][:, :, :9], dim=-1)
    for b in range(B):
        for m in range(M):
            c_idx = int(cat_ids[b, m].item())
            pred_boxes[b, m] = torch.tensor(mean_box_by_cat.get(c_idx, global_mean_box), dtype=torch.float32)

    preds = {
        "pred_boxes": pred_boxes,
        "pred_column_logits": pred_col_logits,
        "pred_order_scores": pred_orders,
        "pred_macro_column_logits": pred_macro,
    }
    return evaluator.evaluate_batch(preds, batch)


def evaluate_template_copy_baseline(
    eval_dataset: Any,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Category-matched template copy baseline: assign each item the next same-category slot from its retrieved template."""
    from src.neural_layout.dataset import collate_layout_batch

    evaluator = LayoutEvaluator()
    n = len(eval_dataset) if max_samples is None else min(len(eval_dataset), max_samples)
    samples = [eval_dataset[i] for i in range(n)]
    batch = collate_layout_batch(samples)

    B, M, _ = batch["content_feats"].shape
    pred_boxes = torch.zeros((B, M, 4), dtype=torch.float32)
    pred_col_logits = torch.zeros((B, M, 5), dtype=torch.float32)
    pred_orders = torch.zeros((B, M), dtype=torch.float32)
    pred_macro = torch.zeros((B, 5), dtype=torch.float32)
    pred_macro[:, 2] = 5.0

    content_cats = torch.argmax(batch["content_feats"][:, :, :9], dim=-1)
    tmpl_cats = torch.argmax(batch["template_feats"][:, :, :9], dim=-1)
    tmpl_boxes = batch["template_feats"][:, :, 9:13]
    tmpl_mask = batch["template_mask"]
    content_mask = batch["content_mask"]

    for b in range(B):
        k_valid = torch.nonzero(tmpl_mask[b], as_tuple=True)[0]
        slots_by_cat: dict[int, list[torch.Tensor]] = {}
        all_slots: list[torch.Tensor] = []
        for k in k_valid.tolist():
            c_k = int(tmpl_cats[b, k].item())
            box_k = tmpl_boxes[b, k]
            slots_by_cat.setdefault(c_k, []).append(box_k)
            all_slots.append(box_k)
        cat_cursor: dict[int, int] = {}

        # Sort valid content items by (section_group_norm, in_cat_rank_norm) when present
        m_valid = torch.nonzero(content_mask[b], as_tuple=True)[0].tolist()
        m_valid_sorted = sorted(
            m_valid,
            key=lambda idx: (
                float(batch["content_feats"][b, idx, 14].item()),
                float(batch["content_feats"][b, idx, 15].item()),
            ),
        )
        for rank_i, m in enumerate(m_valid_sorted):
            c_m = int(content_cats[b, m].item())
            pool = slots_by_cat.get(c_m) or all_slots
            if pool:
                cur = cat_cursor.get(c_m, 0)
                chosen = pool[cur % len(pool)]
                cat_cursor[c_m] = cur + 1
                pred_boxes[b, m] = chosen
                cx = float(chosen[0].item() + 0.5 * chosen[2].item())
                est_col = min(4, max(0, int(cx * 3)))
                pred_col_logits[b, m, est_col] = 5.0
                pred_orders[b, m] = float(chosen[1].item()) + float(est_col)
            else:
                pred_boxes[b, m] = torch.tensor([0.1, 0.1, 0.3, 0.2], dtype=torch.float32)
                pred_col_logits[b, m, 0] = 5.0
                pred_orders[b, m] = float(rank_i) / max(1, len(m_valid_sorted))

    preds = {
        "pred_boxes": pred_boxes,
        "pred_column_logits": pred_col_logits,
        "pred_order_scores": pred_orders,
        "pred_macro_column_logits": pred_macro,
    }
    return evaluator.evaluate_batch(preds, batch)
