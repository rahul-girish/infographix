"""TemplateEncoder: Encodes retrieved template elements and spatial geometry.

Encodes template slots (K elements) with category embeddings and normalized spatial
coordinates [x0, y0, w, h, cx, cy, area] via a Set Transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.neural_layout.content_encoder import SetAttentionBlock


class TemplateEncoder(nn.Module):
    """Encodes retrieved candidate template slots into dense latent representations."""

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        template_feats: torch.Tensor,
        template_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode template slots.

        Args:
            template_feats: [B, K, 16] template slot features.
            template_mask: [B, K] boolean mask where True = valid slot, False = padded slot.

        Returns:
            template_embeds: [B, K, d_model] latent embeddings for each template slot.
        """
        key_padding_mask = ~template_mask

        # Unmask first element if all are masked to prevent NaNs
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False

        h = self.input_proj(template_feats)  # [B, K, d_model]

        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)

        # Zero out embeddings for padded tokens
        h = h * template_mask.unsqueeze(-1).float()
        return h
