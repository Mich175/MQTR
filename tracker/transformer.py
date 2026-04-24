"""
tracker/transformer.py
GTR Transformer core – pure PyTorch, no detectron2.

Key idea:
  - Input: Reid features from all frames in a window
  - Encoder: self-attention across all objects in the window
  - Decoder: trajectory queries (from selected frames) cross-attend to encoder output
  - Output: association matrix (n_queries x N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out    = nn.Linear(d_model, d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        Q = self.q_proj(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(k).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention across ALL objects in the window
        x = self.norm1(x + self.drop(self.self_attn(x, x, x)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,  # (B, n_q, d)  trajectory queries
        memory:  torch.Tensor,  # (B, N,   d)  encoder output
    ) -> torch.Tensor:
        # 1. Self-attention among queries
        queries = self.norm1(queries + self.drop(
            self.self_attn(queries, queries, queries)))
        # 2. Cross-attention: queries attend to all encoded objects
        queries = self.norm2(queries + self.drop(
            self.cross_attn(queries, memory, memory)))
        # 3. Feed-forward
        queries = self.norm3(queries + self.drop(self.ff(queries)))
        return queries


class GTRTransformer(nn.Module):
    """
    Global Tracking Transformer – pure PyTorch.

    Replaces roi_heads._forward_transformer() from the original GTR.

    Args:
        d_model:    feature dimension (must match reid feature dim)
        n_heads:    number of attention heads
        n_enc:      number of encoder layers
        n_dec:      number of decoder layers
        dim_ff:     feed-forward hidden dim
        dropout:    dropout rate
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_enc:   int = 6,
        n_dec:   int = 6,
        dim_ff:  int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_enc)
        ])

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_dec)
        ])

        # Association head: projects decoded queries to association logits
        self.asso_head = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode all object features in the window.

        Args:
            features: (1, N, d_model)  all objects across all frames

        Returns:
            memory: (1, N, d_model)
        """
        x = features
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(
        self,
        queries: torch.Tensor,   # (1, n_q, d_model)
        memory:  torch.Tensor,   # (1, N,   d_model)
    ) -> torch.Tensor:
        """
        Decode trajectory queries against encoded memory.

        Returns:
            decoded: (1, n_q, d_model)
        """
        x = queries
        for layer in self.decoder:
            x = layer(x, memory)
        return x

    def forward(
        self,
        all_features: torch.Tensor,   # (1, N, d_model)
        query_features: torch.Tensor, # (1, n_q, d_model)
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            all_features:   features of ALL objects in the window (Keys/Values)
            query_features: features of QUERY frame objects (Queries)

        Returns:
            asso_logits: (n_q, N) association scores
        """
        # Encode all objects
        memory = self.encode(all_features)             # (1, N, d)

        # Decode queries
        decoded = self.decode(query_features, memory)  # (1, n_q, d)

        # Association scores: dot product between decoded queries and memory
        q = self.asso_head(decoded[0])                 # (n_q, d)
        k = memory[0]                                  # (N, d)
        asso_logits = torch.mm(q, k.t())               # (n_q, N)

        return asso_logits
