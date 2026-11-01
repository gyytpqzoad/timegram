"""Closed ARCH/GARCH feedback filters, composed with the ordinary grammar.

Timegram follows the addendum's Bollerslev order: p variance lags, q shock
lags. The arch package reverses these argument names. No arch code is used
by these simulators. Internal parameter sampling remains private.

Ported from the grammar_v2_20261028 prototype (parameters drawn with the ordinary
`ctx.draw`). A GARCH(1,1) joint prior inside the persistence region is
`priors.Garch11`. To polish: joint priors for higher orders and asymmetric variants, and a
faster batched path (the time loop costs ~0.6 s per 2048 series at burn-in 1000).
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .dists import Fixed, as_dist
from .model import Transform


def _order(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


class GARCH(Transform):
    """sigma**power = omega + alpha*lagged |eps|**power
    + gamma*lagged |eps|**power*I(eps<0) + beta*lagged sigma**power.

    Feed centered standardized IID shocks, independent of the fixed/per-series
    parameters. The declared absolute_moment is E|z|**power; for asymmetric
    terms z must be symmetric. It defaults to one for power=2 and the normal
    moment otherwise. Different shock laws at other powers need their moment
    explicitly supplied. This is an input contract, not inferred from a path.

    Coefficients are nonnegative, omega positive, and powered-moment persistence
    is below one. For power=2 with standardized shocks this gives finite variance;
    at other powers it only gives the corresponding powered moment. The first
    max(p,q) states use initial (sigma**power), defaulting to its mean under that
    contract. Finite burn-in reduces transients, not exact stationary sampling.
    """
    name = "garch"

    def __init__(self, p=1, q=1, *, omega=.1, alpha=None, beta=None, gamma=None,
                 power=2., burn_in=1000, initial=None, absolute_moment=None):
        super().__init__()
        self.p, self.q = _order(p, "p"), _order(q, "q")
        self.burn_in = _order(burn_in, "burn_in")
        self.power = float(power)
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be finite and positive")
        if absolute_moment is None:
            absolute_moment = 2 ** (self.power / 2) * math.gamma((self.power + 1) / 2) / math.sqrt(math.pi)
            if self.power == 2:
                absolute_moment = 1.
        self.absolute_moment = float(absolute_moment)
        if not np.isfinite(self.absolute_moment) or self.absolute_moment <= 0:
            raise ValueError("absolute_moment must be finite and positive")
        self.omega = as_dist(omega)
        self.alpha = as_dist(np.full(self.q, .1 / self.q) if alpha is None and self.q else
                             [] if alpha is None else alpha)
        self.beta = as_dist(np.full(self.p, .8 / self.p) if beta is None and self.p else
                            [] if beta is None else beta)
        self.has_asymmetry = gamma is not None
        self.gamma = as_dist(np.zeros(self.q) if gamma is None else gamma)
        self.initial = None if initial is None else as_dist(initial)
        for name in ("omega", "initial"):
            value = getattr(self, name)
            if value is not None and value.shape:
                raise ValueError(f"{name} must be scalar")
        for name, order in (("alpha", self.q), ("beta", self.p), ("gamma", self.q)):
            shape = getattr(self, name).shape
            if shape != (order,) and not (shape == () and order == 1):
                raise ValueError(f"{name} needs {order} coefficients")
        values = (self.omega, self.alpha, self.beta, self.gamma)
        if all(isinstance(d, Fixed) for d in values) and (
                self.initial is None or isinstance(self.initial, Fixed)):
            self._checked(*(np.asarray(d.value, dtype=float) for d in values),
                          None if self.initial is None else np.asarray(self.initial.value, dtype=float))

    def label(self):
        prefix = "ARCH" if self.p == 0 else "GARCH"
        orders = str(self.q) if self.p == 0 else f"{self.p},{self.q}"
        extra = ", asymmetric" if self.has_asymmetry else ""
        return f"{prefix}({orders}, power={self.power:g}{extra})"

    def params(self):
        result = {"omega": self.omega}
        if self.q: result["alpha"] = self.alpha
        if self.p: result["beta"] = self.beta
        if self.has_asymmetry: result["gamma"] = self.gamma
        if self.initial is not None: result["initial"] = self.initial
        return result

    def canonical_fields(self):
        return {"p": self.p, "q": self.q, "power": self.power,
                "burn_in": self.burn_in, "absolute_moment": self.absolute_moment,
                "asymmetric": self.has_asymmetry, "initial": self.initial}

    def own_presample(self, s): return self.burn_in
    def own_presample_batch(self, b): return self.burn_in

    def _checked(self, omega, alpha, beta, gamma, initial):
        alpha, beta, gamma = map(np.atleast_1d, (alpha, beta, gamma))
        if any(not np.isfinite(v).all() for v in (omega, alpha, beta, gamma)):
            raise ValueError("volatility parameters must be finite")
        if omega <= 0 or any((v < 0).any() for v in (alpha, beta, gamma)):
            raise ValueError("omega must be positive and volatility coefficients nonnegative")
        persistence = self.absolute_moment * (alpha + .5 * gamma).sum() + beta.sum()
        if persistence >= 1:
            raise ValueError("powered-moment persistence must be below one")
        initial = float(omega / (1 - persistence)) if initial is None else float(initial)
        if not np.isfinite(initial) or initial <= 0:
            raise ValueError("initial sigma**power must be finite and positive")
        return float(omega), alpha, beta, gamma, initial

    def _parameters(self, ctx):
        values = [ctx.draw(d) for d in (self.omega, self.alpha, self.beta, self.gamma)]
        initial = None if self.initial is None else ctx.draw(self.initial)
        if not ctx._batched:
            return self._checked(*values, initial)
        omega = values[0].reshape(ctx.n)
        alpha, beta, gamma = (v.reshape(ctx.n, k) for v, k in zip(values[1:], (self.q, self.p, self.q)))
        if any(not bool(torch.isfinite(v).all()) for v in (omega, alpha, beta, gamma)):
            raise ValueError("volatility parameters must be finite")
        if bool((omega <= 0).any()) or any(bool((v < 0).any()) for v in (alpha, beta, gamma)):
            raise ValueError("omega must be positive and volatility coefficients nonnegative")
        persistence = self.absolute_moment * (alpha + .5 * gamma).sum(1) + beta.sum(1)
        if bool((persistence >= 1).any()):
            raise ValueError("powered-moment persistence must be below one")
        initial = omega / (1 - persistence) if initial is None else initial.reshape(ctx.n)
        if not bool(torch.isfinite(initial).all()) or bool((initial <= 0).any()):
            raise ValueError("initial sigma**power must be finite and positive")
        return omega, alpha, beta, gamma, initial

    def _path(self, ctx, shocks):
        """Reference disturbances and conditional variances; no external simulator."""
        shocks = np.asarray(shocks, dtype=float)
        if shocks.shape != ctx.t.shape or not np.isfinite(shocks).all():
            raise ValueError("volatility input must be finite on the simulation time grid")
        omega, alpha, beta, gamma, initial = self._parameters(ctx)
        width, warm = len(shocks), max(self.p, self.q)
        state, eps, magnitude = np.empty(width), np.empty(width), np.empty(width)
        negative = np.empty(width)
        for t in range(width):
            if t < warm:
                value = initial
            else:
                past = slice(t-self.q, t)
                value = (omega + np.dot(alpha, magnitude[past][::-1])
                         + np.dot(gamma, negative[past][::-1])
                         + np.dot(beta, state[t-self.p:t][::-1]))
            state[t] = value
            eps[t] = value ** (1 / self.power) * shocks[t]
            magnitude[t] = abs(eps[t]) ** self.power
            negative[t] = magnitude[t] if eps[t] < 0 else 0.
        variance = state ** (2 / self.power)
        if not np.isfinite(eps).all() or not np.isfinite(variance).all():
            raise ValueError("volatility path became nonfinite")
        return eps, variance

    def _path_batch(self, ctx, shocks):
        if shocks.shape != ctx.t.shape or not bool(torch.isfinite(shocks).all()):
            raise ValueError("volatility input must be finite on the simulation time grid")
        omega, alpha, beta, gamma, initial = self._parameters(ctx)
        width, warm = shocks.shape[1], max(self.p, self.q)
        state, eps, magnitude, negative = (torch.empty_like(shocks) for _ in range(4))
        for t in range(width):
            if t < warm:
                value = initial
            else:
                value = omega.clone()
                for j in range(self.q):
                    value += alpha[:, j] * magnitude[:, t-j-1] + gamma[:, j] * negative[:, t-j-1]
                for j in range(self.p):
                    value += beta[:, j] * state[:, t-j-1]
            state[:, t] = value
            eps[:, t] = value ** (1 / self.power) * shocks[:, t]
            magnitude[:, t] = eps[:, t].abs() ** self.power
            negative[:, t] = torch.where(eps[:, t] < 0, magnitude[:, t], 0.)
        variance = state ** (2 / self.power)
        if not bool(torch.isfinite(eps).all()) or not bool(torch.isfinite(variance).all()):
            raise ValueError("volatility path became nonfinite")
        return eps, variance

    def apply(self, s, x): return self._path(s, x)[0]
    def apply_batch(self, b, x): return self._path_batch(b, x)[0]

    def diagnostics(self):
        return {"feedback": "closed_primitive", "order_convention": "p variance lags, q shock lags",
                "stationary_initialization": False, "burn_in": self.burn_in,
                "moment_condition": "conditional on declared IID symmetric shock moment",
                "power": self.power, "absolute_moment": self.absolute_moment}


def garch(p=1, q=1, **kwargs):
    return GARCH(p, q, **kwargs)


def arch(q=1, **kwargs):
    return GARCH(0, q, **kwargs)
