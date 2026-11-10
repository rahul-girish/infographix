"""LayoutDecoder: Cross-Attention Decoder for Multi-Task Neural Layout Prediction.

Performs cross-attention from user content queries into retrieved template slots,
followed by self-attention between content blocks to resolve placement and interactions.
Predicts bounding boxes, column assignments, reading order scores, and macro columns.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_columns = num_columns

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Prediction Heads:
        # 1. Bounding-Box Head: predicts [x0, y0, w, h] normalized in [0, 1]
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),  # Guarantees valid [0, 1] normalized bounding box coordinates
        )

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

    def forward(
        self,
        content_embeds: torch.Tensor,
        template_embeds: torch.Tensor,
        content_mask: torch.Tensor,
        template_mask: torch.Tensor,
        macro_embed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode predictions for content elements.

        Args:
            content_embeds: [B, M, d_model]
            template_embeds: [B, K, d_model]
            content_mask: [B, M] (True = valid)
            template_mask: [B, K] (True = valid)
            macro_embed: [B, d_model] optional pooled macro representation.

        Returns:
            Dictionary containing:
                "pred_boxes": [B, M, 4] normalized [x0, y0, w, h] in (0, 1)
                "pred_column_logits": [B, M, num_columns]
                "pred_order_scores": [B, M] continuous ordering score
                "pred_macro_column_logits": [B, num_columns]
        """
        h = content_embeds
        for layer in self.layers:
            h = layer(
                content=h,
                template=template_embeds,
                content_mask=content_mask,
                template_mask=template_mask,
            )

        # Mask out padded positions
        mask_expanded = content_mask.unsqueeze(-1).float()
        h_masked = h * mask_expanded

        # Predictions
        pred_boxes = self.bbox_head(h_masked)                   # [B, M, 4] in (0, 1)
        pred_column_logits = self.column_head(h_masked)         # [B, M, 5]
        pred_order_scores = self.order_head(h_masked).squeeze(-1)  # [B, M]

        if macro_embed is not None:
            pred_macro_cols = self.macro_column_head(macro_embed)
        else:
            # Fallback pooling
            pooled = h_masked.sum(dim=1) / torch.clamp(content_mask.sum(dim=1, keepdim=True), min=1.0)
            pred_macro_cols = self.macro_column_head(pooled)

        return {
            "pred_boxes": pred_boxes,
            "pred_column_logits": pred_column_logits,
            "pred_order_scores": pred_order_scores,
            "pred_macro_column_logits": pred_macro_cols,
        }
