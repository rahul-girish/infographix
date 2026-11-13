"""ContentEncoder: Set Transformer with Multi-Head Self-Attention for User Content Demands.

Accepts variable numbers of content elements (M), respects explicit missingness flags,
handles padding masks, and guarantees zero target geometry leakage.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SetAttentionBlock(nn.Module):
    """Set Attention Block (SAB) for permutation-equivariant set processing."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [B, N, d_model].
            key_padding_mask: Boolean tensor of shape [B, N], True indicates padded/ignored tokens.

        Returns:
            Tensor of shape [B, N, d_model].
        """
        # Multi-head self-attention
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ContentEncoder(nn.Module):
    """Encodes user content elements into dense latent representations using a Set Transformer."""

    def __init__(
        self,
        input_dim: int = 18,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Project input demand features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Stack of Set Attention Blocks
        self.layers = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Global pooling query for macro-level representation (Pooling by Multihead Attention)
        self.macro_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.macro_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.macro_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        content_feats: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode content elements.

        Args:
            content_feats: [B, M, 18] content feature vectors.
            content_mask: [B, M] boolean mask where True = valid element, False = padded element.

        Returns:
            element_embeds: [B, M, d_model] latent embeddings for each content element.
            macro_embed: [B, d_model] pooled macro representation for global poster properties.
        """
        batch_size, seq_len, _ = content_feats.shape

        # PyTorch MultiheadAttention expects key_padding_mask where True = ignored/padded
        key_padding_mask = ~content_mask

        # If an entire batch row is masked (e.g. edge case), unmask first element to prevent NaNs in softmax
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False

        h = self.input_proj(content_feats)  # [B, M, d_model]

        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)

        # Zero out embeddings for padded tokens
        h = h * content_mask.unsqueeze(-1).float()

        # Global macro pooling across valid elements
        q = self.macro_query.repeat(batch_size, 1, 1)  # [B, 1, d_model]
        pooled_out, _ = self.macro_pool(
            query=q,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
        )
        macro_embed = self.macro_norm(pooled_out.squeeze(1))  # [B, d_model]

        return h, macro_embed
