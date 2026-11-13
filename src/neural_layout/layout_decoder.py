"""LayoutDecoder: Cross-Attention Decoder for Multi-Task Neural Layout Prediction.

Performs cross-attention from user content queries into retrieved template slots,
followed by self-attention between content blocks to resolve placement and interactions.
Supports both the legacy independent [x0, y0, w, h] parameterization and the
canvas-bounded + template-anchor residual parameterization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention (content -> template) and self-attention."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        # Cross-attention: queries = content, keys/values = template
        self.cross_mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(d_model)

        # Self-attention: content elements interact with each other
        self.self_mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_self = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        content: torch.Tensor,
        template: torch.Tensor,
        content_mask: torch.Tensor,
        template_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            content: [B, M, d_model]
            template: [B, K, d_model]
            content_mask: [B, M] (True = valid)
            template_mask: [B, K] (True = valid)
        """
        key_padding_tmpl = ~template_mask
        key_padding_content = ~content_mask

        # Unmask first item if all padded to prevent NaNs in softmax
        all_padded_t = key_padding_tmpl.all(dim=1)
        if all_padded_t.any():
            key_padding_tmpl = key_padding_tmpl.clone()
            key_padding_tmpl[all_padded_t, 0] = False

        all_padded_c = key_padding_content.all(dim=1)
        if all_padded_c.any():
            key_padding_content = key_padding_content.clone()
            key_padding_content[all_padded_c, 0] = False

        # 1. Cross-attention into template
        cross_out, _ = self.cross_mha(
            query=content,
            key=template,
            value=template,
            key_padding_mask=key_padding_tmpl,
        )
        content = self.norm_cross(content + self.dropout(cross_out))

        # 2. Self-attention among content blocks
        self_out, _ = self.self_mha(
            query=content,
            key=content,
            value=content,
            key_padding_mask=key_padding_content,
        )
        content = self.norm_self(content + self.dropout(self_out))

        # 3. FFN
        ffn_out = self.ffn(content)
        content = self.norm_ffn(content + ffn_out)

        return content


class LayoutDecoder(nn.Module):
    """Decodes conditioned representations into multi-task poster layout predictions."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        num_columns: int = 5,
        dropout: float = 0.1,
        bbox_parameterization: str = "independent",
        use_anchor_conditioning: bool = False,
        min_box_width: float = 0.01,
        min_box_height: float = 0.005,
        num_categories: int = 9,
    ):
        super().__init__()
        if bbox_parameterization not in ("independent", "bounded"):
            raise ValueError(
                f"Unsupported bbox_parameterization '{bbox_parameterization}'. "
                "Expected 'independent' or 'bounded'."
            )

        self.d_model = d_model
        self.num_columns = num_columns
        self.bbox_parameterization = bbox_parameterization
        self.use_anchor_conditioning = use_anchor_conditioning
        self.min_box_width = float(min_box_width)
        self.min_box_height = float(min_box_height)
        self.num_categories = num_categories

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Prediction Heads:
        # 1. Bounding-Box Head
        if bbox_parameterization == "independent" and not use_anchor_conditioning:
            # Preserves exact layer structure for v0.1.0 checkpoint compatibility
            self.bbox_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 4),
                nn.Sigmoid(),
            )
        else:
            self.bbox_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 4),
            )

        if use_anchor_conditioning:
            self.slot_q = nn.Linear(d_model, d_model)
            self.slot_k = nn.Linear(d_model, d_model)
            # Mild positive prior on matching semantic categories
            self.cat_compat_bias = nn.Parameter(torch.eye(num_categories) * 1.0)
            # Start residual box refinements near zero so initial boxes follow retrieved anchors
            nn.init.zeros_(self.bbox_head[-1].weight)
            nn.init.zeros_(self.bbox_head[-1].bias)

        # 2. Column Assignment Head: predicts column index logits (0..num_columns - 1)
        self.column_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_columns),
        )

        # 3. Reading Order Head: predicts a continuous ordering score per element
        self.order_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # 4. Macro Planning Head: predicts global poster column count (1 to 5 columns)
        self.macro_column_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_columns),
        )

    def _compute_anchor_boxes(
        self,
        h_masked: torch.Tensor,
        template_embeds: torch.Tensor,
        template_mask: torch.Tensor,
        template_boxes: torch.Tensor,
        content_cat_ids: torch.Tensor | None = None,
        template_cat_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute category-aware soft template slot assignments and geometric anchor boxes."""
        q = self.slot_q(h_masked)          # [B, M, d_model]
        k = self.slot_k(template_embeds)   # [B, K, d_model]
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_model)

        if content_cat_ids is not None and template_cat_ids is not None:
            c_idx = content_cat_ids.clamp(0, self.num_categories - 1)
            t_idx = template_cat_ids.clamp(0, self.num_categories - 1)
            cat_bias = self.cat_compat_bias[c_idx.unsqueeze(-1), t_idx.unsqueeze(1)]
            attn_logits = attn_logits + cat_bias

        valid_tmpl = template_mask.clone()
        all_padded_t = ~valid_tmpl.any(dim=1)
        if all_padded_t.any():
            valid_tmpl[all_padded_t, 0] = True

        attn_logits = attn_logits.masked_fill(~valid_tmpl.unsqueeze(1), -1e9)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, M, K]
        anchor_boxes = torch.bmm(attn_weights, template_boxes)  # [B, M, 4]
        return anchor_boxes, attn_weights

    def _boxes_to_raw_logits(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert normalized [x0, y0, w, h] anchors into unconstrained parameter logits."""
        ax0, ay0, aw, ah = boxes.unbind(-1)
        eps = 1e-4
        if self.bbox_parameterization == "bounded":
            aw_c = aw.clamp(self.min_box_width + eps, 1.0 - eps)
            ah_c = ah.clamp(self.min_box_height + eps, 1.0 - eps)
            frac_w = ((aw_c - self.min_box_width) / max(eps, 1.0 - self.min_box_width)).clamp(eps, 1.0 - eps)
            frac_h = ((ah_c - self.min_box_height) / max(eps, 1.0 - self.min_box_height)).clamp(eps, 1.0 - eps)
            avail_x = torch.clamp(1.0 - aw_c, min=eps)
            avail_y = torch.clamp(1.0 - ah_c, min=eps)
            frac_x = (ax0 / avail_x).clamp(eps, 1.0 - eps)
            frac_y = (ay0 / avail_y).clamp(eps, 1.0 - eps)
            return torch.stack(
                [torch.logit(frac_x), torch.logit(frac_y), torch.logit(frac_w), torch.logit(frac_h)],
                dim=-1,
            )

        return torch.logit(boxes.clamp(eps, 1.0 - eps))

    def _decode_raw_boxes(self, raw_box_params: torch.Tensor) -> torch.Tensor:
        """Map unconstrained box parameters into valid normalized [x0, y0, w, h]."""
        if self.bbox_parameterization == "bounded":
            raw_x, raw_y, raw_w, raw_h = raw_box_params.unbind(-1)
            w = self.min_box_width + (1.0 - self.min_box_width) * torch.sigmoid(raw_w)
            h = self.min_box_height + (1.0 - self.min_box_height) * torch.sigmoid(raw_h)
            x0 = torch.sigmoid(raw_x) * (1.0 - w)
            y0 = torch.sigmoid(raw_y) * (1.0 - h)
            return torch.stack([x0, y0, w, h], dim=-1)

        return torch.sigmoid(raw_box_params)

    def forward(
        self,
        content_embeds: torch.Tensor,
        template_embeds: torch.Tensor,
        content_mask: torch.Tensor,
        template_mask: torch.Tensor,
        macro_embed: torch.Tensor | None = None,
        template_boxes: torch.Tensor | None = None,
        content_cat_ids: torch.Tensor | None = None,
        template_cat_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode predictions for content elements."""
        h = content_embeds
        for layer in self.layers:
            h = layer(
                content=h,
                template=template_embeds,
                content_mask=content_mask,
                template_mask=template_mask,
            )

        mask_expanded = content_mask.unsqueeze(-1).float()
        h_masked = h * mask_expanded

        anchor_boxes: torch.Tensor | None = None
        template_attn_weights: torch.Tensor | None = None

        if self.use_anchor_conditioning and template_boxes is not None:
            anchor_boxes, template_attn_weights = self._compute_anchor_boxes(
                h_masked=h_masked,
                template_embeds=template_embeds,
                template_mask=template_mask,
                template_boxes=template_boxes,
                content_cat_ids=content_cat_ids,
                template_cat_ids=template_cat_ids,
            )
            anchor_logits = self._boxes_to_raw_logits(anchor_boxes)
            delta_logits = self.bbox_head(h_masked)
            pred_boxes = self._decode_raw_boxes(anchor_logits + delta_logits)
        elif self.bbox_parameterization == "bounded":
            raw_box_params = self.bbox_head(h_masked)
            pred_boxes = self._decode_raw_boxes(raw_box_params)
        else:
            pred_boxes = self.bbox_head(h_masked)

        pred_column_logits = self.column_head(h_masked)
        pred_order_scores = self.order_head(h_masked).squeeze(-1)

        if macro_embed is not None:
            pred_macro_cols = self.macro_column_head(macro_embed)
        else:
            pooled = h_masked.sum(dim=1) / torch.clamp(content_mask.sum(dim=1, keepdim=True), min=1.0)
            pred_macro_cols = self.macro_column_head(pooled)

        out: dict[str, torch.Tensor] = {
            "pred_boxes": pred_boxes,
            "pred_column_logits": pred_column_logits,
            "pred_order_scores": pred_order_scores,
            "pred_macro_column_logits": pred_macro_cols,
        }
        if anchor_boxes is not None:
            out["anchor_boxes"] = anchor_boxes
        if template_attn_weights is not None:
            out["template_attn_weights"] = template_attn_weights
        return out
