"""Fractional filters with explicit finite-history semantics.

These blocks extend the copied package; they do not change integer integration.
An untruncated zero-start filter is exact on its finite simulated history. A
finite prehistory approximates an infinite-past process, without certifying it.

Ported from the grammar_v2_20261028 prototype; the fast path computes the weights with
one cumulative product instead of a Python loop over lags.
"""
from __future__ import annotations

import numpy as np
import torch
import warnings

from .dists import Fixed, as_dist
from .model import Transform


class FractionalBoundaryWarning(UserWarning):
    """The order is at or extremely close to an ordinary integration boundary."""


def fractional_weights(d: float, terms: int) -> np.ndarray:
    """Coefficients of (1-B)**(-d), including the coefficient at lag zero."""
    if not np.isfinite(d):
        raise ValueError("fractional order must be finite")
    if not isinstance(terms, (int, np.integer)) or terms < 1:
        raise ValueError("terms must be a positive integer")
    weights = np.ones(terms, dtype=float)
    for j in range(1, terms):
        weights[j] = weights[j - 1] * (j - 1 + d) / j
    return weights


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class FractionalFilter(Transform):
    """y = (1-B**period)**(-d) x, with optional finite kernel and prehistory.

    memory counts coefficients, including lag zero, rather than time steps.
    presample is the number of time steps before observation 1. With presample=0,
    the filter starts at observation 1, even if another block requests burn-in.
    """
    name = "fractional_integrate"

    def __init__(self, d, period=1, memory=None, presample=0):
        super().__init__()
        self.d = as_dist(d)
        if self.d.shape:
            raise ValueError("fractional order must be scalar")
        if isinstance(self.d, Fixed) and not np.isfinite(self.d.value):
            raise ValueError("fractional order must be finite")
        self.period = _integer(period, "period", 1)
        self.memory = None if memory is None else _integer(memory, "memory", 1)
        self.history = _integer(presample, "presample", 0)
        if isinstance(self.d, Fixed):
            value = float(self.d.value)
            if abs(value - round(value)) <= 1e-12:
                ordinary = "the identity (no integration)" if round(value) == 0 else (
                    "ordinary integration" if value > 0 else "ordinary differencing")
                warnings.warn(f"Fractional order {value:g} is at or near an integer: consider {ordinary}. "
                              "Kernel truncation and initial history must match; no rewrite was performed.",
                              FractionalBoundaryWarning, stacklevel=2)

    def label(self):
        return (f"{self.name}(d={self.d.describe()}, period={self.period}, "
                f"memory={self.memory}, presample={self.history})")

    def params(self):
        return {"d": self.d}

    def canonical_fields(self):
        return {"period": self.period, "memory": self.memory, "presample": self.history}

    def own_presample(self, s):
        return self.history

    def own_presample_batch(self, b):
        return self.history

    def _terms(self, width):
        available = (width - 1) // self.period + 1
        return available if self.memory is None else min(self.memory, available)

    def apply(self, s, x):
        x = np.where(s.t >= 1 - self.history, x, 0.0)
        weights = fractional_weights(float(s.draw(self.d)), self._terms(len(x)))
        kernel = np.zeros((len(weights) - 1) * self.period + 1)
        kernel[::self.period] = weights
        return np.convolve(x, kernel)[:len(x)]

    def apply_batch(self, b, x):
        x = torch.where(b.t >= 1 - self.history, x, torch.zeros_like(x))
        d = b.draw(self.d).reshape(b.n)
        if not bool(torch.isfinite(d).all()):
            raise ValueError("fractional order must be finite")
        width, terms = x.shape[1], self._terms(x.shape[1])
        j = torch.arange(1, terms, dtype=x.dtype, device=x.device)
        ratios = (j.unsqueeze(0) - 1 + d.unsqueeze(1)) / j          # psi_j / psi_{j-1} = (j - 1 + d) / j
        weights = torch.cat([torch.ones((b.n, 1), dtype=x.dtype, device=x.device),
                             torch.cumprod(ratios, 1)], 1)
        kernel = x.new_zeros((b.n, (terms - 1) * self.period + 1))
        kernel[:, ::self.period] = weights
        fft_length = 1 << (width + kernel.shape[1] - 2).bit_length()
        return torch.fft.irfft(torch.fft.rfft(x, fft_length) *
                               torch.fft.rfft(kernel, fft_length), fft_length)[:, :width]

    def diagnostics(self):
        return {
            "initialization": "zero_start" if self.history == 0 else "finite_prehistory",
            "kernel": "full_available_history" if self.memory is None else "truncated",
            "infinite_past_stationarity": "not_certified",
            "note": "A fixed truncated kernel has finite memory, even for positive d.",
        }


def fractional_integrate(d, period=1, memory=None, presample=0):
    return FractionalFilter(d, period, memory, presample)


def fractional_difference(d, period=1, memory=None, presample=0):
    """Apply (1-B**period)**d, using the same core with integration order -d."""
    order = as_dist(d)
    inverse = -float(order.value) if isinstance(order, Fixed) else -order
    result = FractionalFilter(inverse, period, memory, presample)
    result.name = "fractional_difference"
    return result
