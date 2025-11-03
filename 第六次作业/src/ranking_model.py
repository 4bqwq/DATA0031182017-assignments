from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn


class ResidualGatedBlock(nn.Module):
    """Feature interaction block combining residual skip with gating."""

    def __init__(self, hidden_dim: int, expansion: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        deep = self.fc1(x)
        deep = self.activation(deep)
        deep = self.dropout(deep)
        deep = self.fc2(deep)

        gate = torch.sigmoid(self.gate(x))
        out = residual + gate * deep
        return out


class CrossFeatureBlock(nn.Module):
    """Cross network style block to capture multiplicative feature interactions."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x0 * (x W) + b + x
        xw = torch.matmul(x, self.weight)  # (batch, 1)
        return x0 * xw + self.bias + x


class RankingNet(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        embedding_info: List[Tuple[int, int]],
        hidden_dim: int = 128,
        num_residual_blocks: int = 3,
        cross_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=dim)
            for size, dim in embedding_info
        ])

        embedding_dim_total = sum(dim for _, dim in embedding_info)
        numeric_projection_dim = max(32, numeric_dim * 2)
        self.numeric_projector = nn.Sequential(
            nn.Linear(numeric_dim, numeric_projection_dim),
            nn.GELU(),
            nn.LayerNorm(numeric_projection_dim),
        )

        self.cross_blocks = nn.ModuleList(
            CrossFeatureBlock(numeric_projection_dim + embedding_dim_total)
            for _ in range(cross_layers)
        )

        fusion_dim = numeric_projection_dim + embedding_dim_total
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
        )

        residuals = []
        for _ in range(num_residual_blocks):
            residuals.append(ResidualGatedBlock(hidden_dim, expansion=2, dropout=dropout))
        self.residual_stack = nn.Sequential(*residuals)

        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        numeric_proj = self.numeric_projector(numeric)

        embedded = []
        for emb, values in zip(self.embeddings, categorical.T):
            embedded.append(emb(values))
        if embedded:
            cat_features = torch.cat(embedded, dim=1)
            fused = torch.cat([numeric_proj, cat_features], dim=1)
        else:
            fused = numeric_proj

        x0 = fused
        x = fused
        for block in self.cross_blocks:
            x = block(x0, x)

        x = self.fusion_layer(x)
        x = self.residual_stack(x)
        out = self.reg_head(x)
        return out
