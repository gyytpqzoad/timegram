"""Optimizers. The default is Muon on the hidden matrices with AdamW elsewhere (in our
experiments about 1.6x faster to equal accuracy than AdamW).

Muon (Jordan et al. 2024): SGD momentum whose update is orthogonalized by a few
Newton-Schulz iterations, for 2-D weight matrices. Embedding, head, biases and norms
use AdamW.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonal factor of each matrix in a stack [k, m, n]."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transpose = X.shape[1] > X.shape[2]
    if transpose:
        X = X.transpose(1, 2)
    X = X / (X.flatten(1).norm(dim=1)[:, None, None] + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        X = a * X + (b * A + c * A @ A) @ X
    return X.transpose(1, 2) if transpose else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.95, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            by_shape: dict[tuple, list] = {}
            for p in g["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("m", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                by_shape.setdefault(tuple(p.shape), []).append((p, p.grad.add(buf, alpha=g["momentum"])))
            for shape, items in by_shape.items():          # same-shape matrices in one batch
                update = newton_schulz(torch.stack([u for _, u in items]), g["ns_steps"])
                update *= max(1.0, shape[0] / shape[1]) ** 0.5
                for (p, _), u in zip(items, update):
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(u.to(p.dtype), alpha=-g["lr"])


@dataclass(frozen=True)
class MuonAdamW:
    """Muon (lr `muon_lr`) on the network's hidden matrices, AdamW (lr `lr`) on the rest."""

    lr: float = 1e-3
    muon_lr: float = 0.01
    momentum: float = 0.95
    weight_decay: float = 0.01

    def build(self, net) -> list[torch.optim.Optimizer]:
        hidden = {id(p) for p in net.hidden_matrices()}
        rest = [p for p in net.parameters() if id(p) not in hidden]
        fused = next(net.parameters()).is_cuda
        opts = [Muon(net.hidden_matrices(), lr=self.muon_lr, momentum=self.momentum,
                     weight_decay=self.weight_decay),
                torch.optim.AdamW(rest, lr=self.lr, weight_decay=self.weight_decay, fused=fused)]
        for o in opts:
            for g in o.param_groups:
                g["base_lr"] = g["lr"]
        return opts

    def to_dict(self):
        return {"family": "muon_adamw", **asdict(self)}


@dataclass(frozen=True)
class AdamW:
    lr: float = 1e-3
    weight_decay: float = 0.01

    def build(self, net):
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                fused=next(net.parameters()).is_cuda)
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
        return [opt]

    def to_dict(self):
        return {"family": "adamw", **asdict(self)}


def cosine_with_warmup(seen: int, total: int, warmup: int) -> float:
    """Learning-rate multiplier after `seen` training samples."""
    if seen < warmup:
        return seen / warmup
    return 0.5 * (1 + math.cos(math.pi * (seen - warmup) / max(total - warmup, 1)))
