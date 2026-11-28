"""Structured Column-Grid Post-Processor for Neural Poster Layout Predictions.

Converts raw continuous predictions from `TemplateConditionedLayoutModel` into
clean, publication-ready, 0-overlap, column-aligned poster layouts by leveraging
the model's strong structural predictions:
  - Macro column count (`pred_macro_column_logits`)
  - Per-element column assignment (`pred_column_logits`)
  - Reading order (`pred_order_scores`)
  - Element heights & intrinsic figure aspect ratios (`pred_boxes`, `content_feats`)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.neural_layout.dataset import KNOWN_CATEGORIES


@dataclass
class RefinerConfig:
    """Layout geometry and spacing parameters for structured column-grid packing."""

    margin_x: float = 0.025
    margin_top: float = 0.020
    margin_bottom: float = 0.020
    col_gap: float = 0.020
    header_gap: float = 0.018
    item_gap: float = 0.010
    tight_gap: float = 0.004
    min_columns: int = 2
    max_columns: int = 4
    max_header_height: float = 0.20
    min_box_height: float = 0.015
    rebalance_columns: bool = True
    expand_sparse_columns: bool = True
    max_expand_factor: float = 1.25


class NeuralLayoutRefiner:
    """Snaps raw neural layout predictions onto a clean, 0-overlap column grid."""

    def __init__(self, config: RefinerConfig | None = None):
        self.config = config or RefinerConfig()

    def _choose_num_columns(
        self,
        macro_logits: np.ndarray,
        num_body: int,
    ) -> int:
        if num_body <= 2:
            return 1
        cfg = self.config
        lo = max(1, cfg.min_columns)
        hi = min(len(macro_logits), cfg.max_columns, num_body)
        if hi < lo:
            return max(1, hi)
        valid_logits = macro_logits[lo - 1 : hi]
        return int(lo + int(np.argmax(valid_logits)))

    def _desired_body_height(
        self,
        cat: str,
        pred_h: float,
        col_w: float,
        has_aspect: bool,
        intr_aspect: float,
        body_height: float,
    ) -> float:
        if cat in ("figure", "logo") and has_aspect and intr_aspect > 0.05:
            # Ideal height at full column width; cap so tall figures don't dominate a column
            h_full = col_w / float(np.clip(intr_aspect, 0.25, 4.0))
            return float(np.clip(h_full, 0.05, 0.42 * body_height))
        if cat == "section":
            return float(np.clip(pred_h, 0.022, 0.052))
        if cat == "caption":
            return float(np.clip(pred_h, 0.020, 0.060))
        if cat in ("figure", "logo"):
            return float(np.clip(pred_h, 0.060, 0.35 * body_height))
        return float(np.clip(pred_h, 0.035, 0.42 * body_height))

    def _pack_header(
        self,
        header_indices: list[int],
        cats: list[str],
        pred_boxes: np.ndarray,
        has_aspect: np.ndarray,
        intr_aspect: np.ndarray,
        out_boxes: np.ndarray,
    ) -> float:
        """Place title, author_info, and top logos cleanly in the banner zone.

        Returns the y-coordinate where the body area should begin.
        """
        cfg = self.config
        if not header_indices:
            return cfg.margin_top

        logos = [i for i in header_indices if cats[i] == "logo"]
        titles = [i for i in header_indices if cats[i] == "title"]
        authors = [i for i in header_indices if cats[i] == "author_info"]
        others = [i for i in header_indices if i not in logos and i not in titles and i not in authors]

        # Sort titles/authors by predicted y0
        titles.sort(key=lambda i: float(pred_boxes[i, 1]))
        authors.sort(key=lambda i: float(pred_boxes[i, 1]))
        text_header = titles + authors + others

        # Reserve horizontal slots for up to 2 header logos (left and right corners)
        left_logo = logos[0] if len(logos) >= 1 else None
        right_logo = logos[1] if len(logos) >= 2 else None
        extra_logos = logos[2:]

        logo_slot_w = 0.09
        banner_x0 = cfg.margin_x + (logo_slot_w + cfg.item_gap if left_logo is not None else 0.0)
        banner_x1 = (1.0 - cfg.margin_x) - (logo_slot_w + cfg.item_gap if right_logo is not None else 0.0)
        banner_w = max(0.30, banner_x1 - banner_x0)

        # Compute desired heights for stacked text header items
        desired_hs: list[float] = []
        for idx in text_header:
            c = cats[idx]
            ph = float(pred_boxes[idx, 3])
            if c == "title":
                desired_hs.append(float(np.clip(ph, 0.042, 0.095)))
            elif c == "author_info":
                desired_hs.append(float(np.clip(ph, 0.026, 0.065)))
            else:
                desired_hs.append(float(np.clip(ph, 0.025, 0.060)))

        total_gaps = cfg.tight_gap * max(0, len(text_header) - 1)
        total_h = sum(desired_hs) + total_gaps
        if total_h > cfg.max_header_height and sum(desired_hs) > 1e-6:
            scale = max(0.3, (cfg.max_header_height - total_gaps) / sum(desired_hs))
            desired_hs = [max(cfg.min_box_height, h * scale) for h in desired_hs]

        y_cur = cfg.margin_top
        for idx, h_val in zip(text_header, desired_hs):
            out_boxes[idx] = [banner_x0, y_cur, banner_w, h_val]
            y_cur += h_val + cfg.tight_gap

        header_h = max(0.06, (y_cur - cfg.tight_gap) - cfg.margin_top) if text_header else 0.07

        def _place_logo(idx: int, slot_x0: float, slot_w: float) -> None:
            asp = float(np.clip(intr_aspect[idx], 0.4, 2.5)) if has_aspect[idx] else 1.0
            lh = min(header_h, slot_w / asp)
            lw = min(slot_w, lh * asp)
            lx = slot_x0 + 0.5 * (slot_w - lw)
            ly = cfg.margin_top + 0.5 * (header_h - lh)
            out_boxes[idx] = [lx, ly, lw, lh]

        if left_logo is not None:
            _place_logo(left_logo, cfg.margin_x, logo_slot_w)
        if right_logo is not None:
            _place_logo(right_logo, 1.0 - cfg.margin_x - logo_slot_w, logo_slot_w)
        if extra_logos:
            # Place any additional header logos neatly in a slim strip below the banner
            strip_h = 0.035
            strip_y = cfg.margin_top + header_h + cfg.tight_gap
            slot_w = min(0.08, (1.0 - 2 * cfg.margin_x) / max(1, len(extra_logos)))
            for k_i, l_idx in enumerate(extra_logos):
                _place_logo(l_idx, cfg.margin_x + k_i * (slot_w + 0.008), slot_w)
                out_boxes[l_idx, 1] = strip_y
                out_boxes[l_idx, 3] = min(out_boxes[l_idx, 3], strip_h)
            header_h += cfg.tight_gap + strip_h

        return cfg.margin_top + header_h + cfg.header_gap

    def refine_single(
        self,
        pred_boxes: np.ndarray,
        pred_col_logits: np.ndarray,
        pred_orders: np.ndarray,
        pred_macro_logits: np.ndarray,
        content_feats: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refine a single poster's predictions into a 0-overlap, column-aligned layout.

        Returns:
            refined_boxes: [M, 4] in [x0, y0, w, h]
            refined_cols: [M] integer column indices
            refined_orders: [M] continuous reading-order scores
        """
        M = int(pred_boxes.shape[0])
        if M == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )

        cfg = self.config
        cat_ids = np.argmax(content_feats[:, :9], axis=-1)
        cats = [KNOWN_CATEGORIES[int(c)] if int(c) < len(KNOWN_CATEGORIES) else "text" for c in cat_ids]
        intr_aspect = content_feats[:, 12].astype(np.float32)
        has_aspect = content_feats[:, 13] > 0.5
        sec_group = content_feats[:, 14].astype(np.float32)
        in_cat_rank = content_feats[:, 15].astype(np.float32)

        # 1. Separate header vs body elements
        header_indices: list[int] = []
        body_indices: list[int] = []
        for i in range(M):
            c = cats[i]
            if c in ("title", "author_info") or (c == "logo" and float(pred_boxes[i, 1]) < 0.22):
                header_indices.append(i)
            else:
                body_indices.append(i)

        out_boxes = np.zeros((M, 4), dtype=np.float32)
        out_cols = np.zeros((M,), dtype=np.int64)
        out_orders = np.zeros((M,), dtype=np.float32)

        body_top = self._pack_header(
            header_indices=header_indices,
            cats=cats,
            pred_boxes=pred_boxes,
            has_aspect=has_aspect,
            intr_aspect=intr_aspect,
            out_boxes=out_boxes,
        )
        body_height = max(0.20, (1.0 - cfg.margin_bottom) - body_top)

        if not body_indices:
            for rank_i, idx in enumerate(header_indices):
                out_orders[idx] = float(rank_i) / max(1, M)
            return out_boxes, out_cols, out_orders

        # 2. Choose number of columns K and compute column geometry
        K = self._choose_num_columns(pred_macro_logits, len(body_indices))
        col_w = (1.0 - 2.0 * cfg.margin_x - (K - 1) * cfg.col_gap) / float(K)
        col_x0s = [cfg.margin_x + c * (col_w + cfg.col_gap) for c in range(K)]

        # 3. Assign body elements to columns using pred_column_logits (masked to 0..K-1)
        col_bins: list[list[int]] = [[] for _ in range(K)]
        for idx in body_indices:
            logits_k = pred_col_logits[idx, :K]
            c_pred = int(np.argmax(logits_k))
            col_bins[c_pred].append(idx)

        # Compute desired heights for every body element
        desired_h: dict[int, float] = {
            idx: self._desired_body_height(
                cat=cats[idx],
                pred_h=float(pred_boxes[idx, 3]),
                col_w=col_w,
                has_aspect=bool(has_aspect[idx]),
                intr_aspect=float(intr_aspect[idx]),
                body_height=body_height,
            )
            for idx in body_indices
        }

        # Sort helper within a column: reading order + section grouping + figure-caption pairing
        def _sort_column(items: list[int]) -> list[int]:
            if len(items) <= 1:
                return list(items)
            # Blend pred_order_scores rank and pred_y0 rank; keep section headings before their group items
            ord_vals = np.array([float(pred_orders[i]) for i in items])
            y_vals = np.array([float(pred_boxes[i, 1]) for i in items])
            ord_rank = np.argsort(np.argsort(ord_vals)).astype(float) / max(1, len(items) - 1)
            y_rank = np.argsort(np.argsort(y_vals)).astype(float) / max(1, len(items) - 1)
            score_map = {
                items[j]: 0.55 * float(ord_rank[j]) + 0.45 * float(y_rank[j])
                for j in range(len(items))
            }
            ordered = sorted(
                items,
                key=lambda i: (
                    score_map[i],
                    0 if cats[i] == "section" else 1,
                    float(in_cat_rank[i]),
                ),
            )
            # Attach captions directly beneath the nearest preceding figure/table in the same section group
            reordered: list[int] = []
            captions_by_group: dict[float, list[int]] = {}
            for idx in ordered:
                if cats[idx] == "caption":
                    captions_by_group.setdefault(round(float(sec_group[idx]), 3), []).append(idx)
            used_caps: set[int] = set()
            for idx in ordered:
                if idx in used_caps:
                    continue
                if cats[idx] == "caption":
                    reordered.append(idx)
                    used_caps.add(idx)
                    continue
                reordered.append(idx)
                if cats[idx] in ("figure", "table"):
                    gkey = round(float(sec_group[idx]), 3)
                    pool = captions_by_group.get(gkey, [])
                    for cap_idx in pool:
                        if cap_idx not in used_caps:
                            reordered.append(cap_idx)
                            used_caps.add(cap_idx)
                            break
            return reordered

        for c in range(K):
            col_bins[c] = _sort_column(col_bins[c])

        # 4. Gentle column height rebalancing so no column is empty or severely overflowing
        if cfg.rebalance_columns and K > 1:
            for _ in range(len(body_indices)):
                col_loads = [
                    sum(desired_h[i] for i in col_bins[c]) + cfg.item_gap * max(0, len(col_bins[c]) - 1)
                    for c in range(K)
                ]
                heaviest = int(np.argmax(col_loads))
                lightest = int(np.argmin(col_loads))
                if len(col_bins[heaviest]) <= 1:
                    break
                # Rebalance only if a column is empty or severely overloaded vs. another column
                needs_fill = len(col_bins[lightest]) == 0
                is_overloaded = (col_loads[heaviest] > body_height * 1.35) and (
                    col_loads[heaviest] - col_loads[lightest] > 0.30 * body_height
                )
                if not (needs_fill or is_overloaded):
                    break
                # Move a boundary element toward the lighter column to preserve reading order
                if heaviest < lightest:
                    moved = col_bins[heaviest].pop(-1)
                    col_bins[heaviest + 1].insert(0, moved)
                else:
                    moved = col_bins[heaviest].pop(0)
                    col_bins[heaviest - 1].append(moved)

            for c in range(K):
                col_bins[c] = _sort_column(col_bins[c])

        # 5. Vertically pack every column with guaranteed 0 overlap and exact aspect preservation
        order_counter = 0
        for idx in header_indices:
            out_cols[idx] = 0
            out_orders[idx] = float(order_counter) / max(1, M - 1)
            order_counter += 1

        for c in range(K):
            items = col_bins[c]
            if not items:
                continue

            # Determine inter-item vertical gaps (tighter under section headings and figure->caption)
            gaps: list[float] = []
            for j in range(len(items) - 1):
                cur_cat = cats[items[j]]
                nxt_cat = cats[items[j + 1]]
                if cur_cat == "section" or (cur_cat in ("figure", "table") and nxt_cat == "caption"):
                    gaps.append(cfg.tight_gap)
                else:
                    gaps.append(cfg.item_gap)

            total_gaps = sum(gaps)
            hs = [desired_h[i] for i in items]
            sum_h = sum(hs)
            avail_h = max(cfg.min_box_height * len(items), body_height - total_gaps)

            if sum_h > avail_h and sum_h > 1e-6:
                # Shrink proportionally to fit cleanly inside [body_top, 1 - margin_bottom]
                scale = avail_h / sum_h
                hs = [max(cfg.min_box_height, h * scale) for h in hs]
                # Final safeguard if min_box_height clamping slightly exceeded avail_h
                cur_sum = sum(hs)
                if cur_sum > avail_h and cur_sum > 1e-6:
                    hs = [h * (avail_h / cur_sum) for h in hs]
            elif cfg.expand_sparse_columns and sum_h < 0.85 * avail_h and sum_h > 1e-6:
                # Gently expand flexible text/list/table blocks so sparse columns feel balanced
                flex_mask = [cats[i] in ("text", "list", "table") for i in items]
                flex_sum = sum(h for h, is_f in zip(hs, flex_mask) if is_f)
                if flex_sum > 1e-6:
                    target_add = min(0.85 * avail_h - sum_h, flex_sum * (cfg.max_expand_factor - 1.0))
                    factor = 1.0 + target_add / flex_sum
                    hs = [h * factor if is_f else h for h, is_f in zip(hs, flex_mask)]

            y_cur = body_top
            col_x0 = col_x0s[c]
            for j, idx in enumerate(items):
                h_i = float(hs[j])
                w_i = col_w
                x_i = col_x0

                # Preserve intrinsic aspect ratio for figures/logos by centering inside the column
                if cats[idx] in ("figure", "logo") and bool(has_aspect[idx]) and float(intr_aspect[idx]) > 0.05:
                    asp = float(np.clip(intr_aspect[idx], 0.25, 4.0))
                    ideal_w = h_i * asp
                    if ideal_w < col_w:
                        w_i = max(0.03, ideal_w)
                        x_i = col_x0 + 0.5 * (col_w - w_i)

                out_boxes[idx] = [x_i, y_cur, w_i, h_i]
                out_cols[idx] = c
                out_orders[idx] = float(order_counter) / max(1, M - 1)
                order_counter += 1
                if j < len(gaps):
                    y_cur += h_i + gaps[j]

        # Final numerical clamp to strictly guarantee [0, 1] canvas bounds
        out_boxes[:, 0] = np.clip(out_boxes[:, 0], 0.0, 1.0 - 1e-4)
        out_boxes[:, 1] = np.clip(out_boxes[:, 1], 0.0, 1.0 - 1e-4)
        out_boxes[:, 2] = np.clip(out_boxes[:, 2], 1e-4, 1.0 - out_boxes[:, 0])
        out_boxes[:, 3] = np.clip(out_boxes[:, 3], 1e-4, 1.0 - out_boxes[:, 1])

        return out_boxes, out_cols, out_orders

    @torch.no_grad()
    def refine_batch(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Refine an entire batch of neural predictions into drop-in evaluator-compatible tensors."""
        pred_boxes = predictions["pred_boxes"].detach().cpu()
        pred_col_logits = predictions["pred_column_logits"].detach().cpu()
        pred_orders = predictions["pred_order_scores"].detach().cpu()
        pred_macro = predictions["pred_macro_column_logits"].detach().cpu()

        content_feats = batch["content_feats"].detach().cpu()
        content_mask = batch["content_mask"].detach().cpu()

        B, M, _ = pred_boxes.shape
        num_cols = pred_col_logits.shape[-1]

        ref_boxes = torch.zeros_like(pred_boxes)
        ref_col_logits = pred_col_logits.clone()
        ref_orders = pred_orders.clone()

        for b in range(B):
            valid_idx = torch.nonzero(content_mask[b], as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue
            rb, rc, ro = self.refine_single(
                pred_boxes=pred_boxes[b, valid_idx].numpy(),
                pred_col_logits=pred_col_logits[b, valid_idx].numpy(),
                pred_orders=pred_orders[b, valid_idx].numpy(),
                pred_macro_logits=pred_macro[b].numpy(),
                content_feats=content_feats[b, valid_idx].numpy(),
            )
            ref_boxes[b, valid_idx] = torch.from_numpy(rb)
            ref_orders[b, valid_idx] = torch.from_numpy(ro)
            # Encode refined column assignment cleanly as sharp logits
            col_one_hot = torch.full((len(valid_idx), num_cols), -5.0, dtype=pred_col_logits.dtype)
            for j, c_val in enumerate(rc):
                col_one_hot[j, min(num_cols - 1, max(0, int(c_val)))] = 5.0
            ref_col_logits[b, valid_idx] = col_one_hot

        device = predictions["pred_boxes"].device
        return {
            "pred_boxes": ref_boxes.to(device),
            "pred_column_logits": ref_col_logits.to(device),
            "pred_order_scores": ref_orders.to(device),
            "pred_macro_column_logits": predictions["pred_macro_column_logits"],
        }
