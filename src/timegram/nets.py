"""Neural estimators. The default is a residual MLP: the best real-data accuracy among the
architectures we tried, small and fast.

A network maps (x [B, L], m [B, L]) -> [B, out]: x is the normalized context,
right-aligned with zeros at missing / padded positions, m is 1 where x is observed. It
knows nothing about time series structure; the estimand decides what `out` means.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """h + W2 act(W1 LayerNorm(h))  (pre-norm residual MLP block)."""

    def __init__(self, width: int, expansion: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * expansion)
        self.down = nn.Linear(width * expansion, width)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)

    def forward(self, h):
        return h + self.down(self.drop(self.act(self.up(self.norm(h)))))


class ResMLPNet(nn.Module):
    def __init__(self, context: int, out: int, width: int, depth: int, expansion: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(2 * context, width)
        self.blocks = nn.ModuleList(ResidualBlock(width, expansion, dropout) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out)

    def forward(self, x, m):
        h = self.inp(torch.cat([x, m], 1))
        for block in self.blocks:
            h = block(h)
        return self.head(self.out_norm(h))

    def hidden_matrices(self):
        """The 2-D weights inside the residual blocks (what Muon optimizes)."""
        return [p for p in self.blocks.parameters() if p.ndim == 2]


@dataclass(frozen=True)
class ResMLP:
    """Network spec (a value, so it can be part of a library key)."""

    width: int = 384
    depth: int = 4
    expansion: int = 2
    dropout: float = 0.0

    def build(self, context: int, out: int) -> ResMLPNet:
        return ResMLPNet(context, out, self.width, self.depth, self.expansion, self.dropout)

    def to_dict(self):
        return {"family": "resmlp", **asdict(self)}
