"""TemplateConditionedLayoutModel: Retrieval-Augmented Neural Poster Layout Generator.

Combines ContentEncoder, TemplateEncoder, and LayoutDecoder into an end-to-end differentiable
architecture for scientific poster composition.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.neural_layout.content_encoder import ContentEncoder
from src.neural_layout.template_encoder import TemplateEncoder
from src.neural_layout.layout_decoder import LayoutDecoder


class TemplateConditionedLayoutModel(nn.Module):
    """End-to-End Retrieval-Augmented Neural Poster Layout Generator."""

    def __init__(
        self,
        content_dim: int = 18,
        template_dim: int = 16,
        d_model: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        ffn_dim: int = 256,
        num_columns: int = 5,
        dropout: float = 0.1,
        bbox_parameterization: str = "independent",
        use_anchor_conditioning: bool = False,
        min_box_width: float = 0.01,
        min_box_height: float = 0.005,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.template_dim = template_dim
        self.d_model = d_model
        self.num_columns = num_columns
        self.bbox_parameterization = bbox_parameterization
        self.use_anchor_conditioning = use_anchor_conditioning
        self.min_box_width = float(min_box_width)
        self.min_box_height = float(min_box_height)

        self.content_encoder = ContentEncoder(
            input_dim=content_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=encoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self.template_encoder = TemplateEncoder(
            input_dim=template_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=encoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self.decoder = LayoutDecoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=decoder_layers,
            ffn_dim=ffn_dim,
            num_columns=num_columns,
            dropout=dropout,
            bbox_parameterization=bbox_parameterization,
            use_anchor_conditioning=use_anchor_conditioning,
            min_box_width=min_box_width,
            min_box_height=min_box_height,
        )

    def forward(
        self,
        content_feats: torch.Tensor,
        template_feats: torch.Tensor,
        content_mask: torch.Tensor,
        template_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            content_feats: [B, M, 18]
            template_feats: [B, K, 16]
            content_mask: [B, M] (bool)
            template_mask: [B, K] (bool)

        Returns:
            Dictionary of predictions:
                pred_boxes: [B, M, 4] normalized [x0, y0, w, h] in (0, 1)
                pred_column_logits: [B, M, 5]
                pred_order_scores: [B, M]
                pred_macro_column_logits: [B, 5]
        """
        content_embeds, macro_embed = self.content_encoder(content_feats, content_mask)
        template_embeds = self.template_encoder(template_feats, template_mask)

        template_boxes = template_feats[..., 9:13] if template_feats.shape[-1] >= 13 else None
        content_cat_ids = (
            torch.argmax(content_feats[..., :9], dim=-1)
            if content_feats.shape[-1] >= 9
            else None
        )
        template_cat_ids = (
            torch.argmax(template_feats[..., :9], dim=-1)
            if template_feats.shape[-1] >= 9
            else None
        )

        predictions = self.decoder(
            content_embeds=content_embeds,
            template_embeds=template_embeds,
            content_mask=content_mask,
            template_mask=template_mask,
            macro_embed=macro_embed,
            template_boxes=template_boxes,
            content_cat_ids=content_cat_ids,
            template_cat_ids=template_cat_ids,
        )

        return predictions
