"""Sources and atomic primitives.

Each block has a reference implementation (`run` / `apply`: one series, a plain loop
that mirrors the definition) and a fast one (`run_batch` / `apply_batch`: a batch, in
torch). The reference is the definition; the fast path must agree with it.

Time: `s.t` holds the time of each simulated step (t = 1 is the first observation).
Deterministic sources and integrators start at t = 1; everything before is zero. Their
starting states enter explicitly, as `constant`, `trend` and `pattern` sources. Noise exists at every step, so filters (ar, ma) run through
the pre-sample period and reach their steady state by t = 1.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .dists import (Choice, Distribution, Fixed, Invertible, JointKey, Laplace, Normal, Stationary,
                    StudentT, Uniform, as_dist, count)
from .model import Source, Transform


def _on(t):
    """Where a block that starts at the first observation is switched on: t >= 1."""
    return t >= 1


def _coefficients(dist, k: int, name: str) -> Distribution:
    """A coefficient vector parameter of length k (a scalar is accepted for k = 1). A joint
    entry may declare a shape; its actual draws are still checked when used."""
    dist = as_dist(dist)
    if not isinstance(dist, JointKey) and dist.shape not in ((k,), () if k == 1 else (k,)):
        raise ValueError(f"{name} needs {k} coefficients, not shape {dist.shape}")
    if isinstance(dist, Fixed) and not np.isfinite(np.asarray(dist.value, float)).all():
        raise ValueError(f"{name} must be finite")
    return dist


def _drawn_coefficients(v, k: int, name: str):
    """A drawn coefficient vector, one series [k] or a batch [n, k]. A shape check only
    (no device synchronization)."""
    if torch.is_tensor(v):
        if v.dim() not in (1, 2):
            raise ValueError(f"{name}: drawn shape {tuple(v.shape)}, expected (series, {k})")
        n = v.shape[0]
        if tuple(v.shape) not in ((n, k), (n,) if k == 1 else (n, k)):
            raise ValueError(f"{name}: drawn shape {tuple(v.shape)}, expected ({n}, {k})")
        return v.reshape(n, k)
    v = np.atleast_1d(v)
    if v.shape != (k,):
        raise ValueError(f"{name}: drawn shape {v.shape}, expected ({k},)")
    return v


def _order_dist(d, name: str) -> Distribution:
    """A random or fixed nonnegative integer order. Continuous laws are refused here;
    other draws are checked when they are used."""
    d = as_dist(d)
    if isinstance(d, Fixed):
        count(d.value, name)
    elif isinstance(d, Choice):
        for v in d.values:
            count(v, name)
    elif isinstance(d, (Uniform, Normal, StudentT, Laplace)):
        point = _point_mass(d)
        if point is None:
            raise ValueError(f"{name} must be integer-valued (use tg.integers or tg.choice), not {d.describe()}")
        count(point, name)
    return d


def _point_mass(d):
    """The value of a continuous-family law with zero spread (e.g. uniform(1, 1)), else None."""
    if isinstance(d, Uniform) and d.low == d.high:
        return d.low
    if isinstance(d, Normal) and d.sd == 0:
        return d.mu
    if isinstance(d, Laplace) and d.scale == 0:
        return d.loc
    return None


def _batch_scalar(v):
    """A per-series parameter [n] broadcast against time [n, W]."""
    return v.unsqueeze(-1) if torch.is_tensor(v) and v.dim() == 1 else v


# ---------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------
class Noise(Source):
    """IID draws e_t = scale * z_t, z_t ~ shape."""

    def own_type_rules(self, context):
        return context.noise(self.shape, self.scale)


    name = "noise"

    def __init__(self, shape: Distribution | None = None, scale=1.0):
        self.shape = shape if shape is not None else Normal(0, 1)
        self.scale = as_dist(scale)

    def label(self):
        sc = "" if isinstance(self.scale, Fixed) and self.scale.value == 1.0 else f", scale={self.scale.describe()}"
        return f"noise({self.shape.describe()}{sc})"

    def params(self):
        return {"scale": self.scale}

    def run(self, s):
        return s.draw(self.scale) * self.shape.sample_iid(s.rng, len(s.t))

    def run_batch(self, b):
        z = self.shape.sample_iid_batch(b.gen, tuple(b.t.shape), b.device)
        return _batch_scalar(b.draw(self.scale)) * z


class Input(Source):
    """Placeholder for the input of a compound block (bound when the compound runs)."""

    def own_type_rules(self, context):
        return context.input()


    name = "input"

    def run(self, s):
        return s.bound(self)

    def run_batch(self, b):
        return b.bound(self)


class Constant(Source):
    """Deterministic level y_t = c, from t = 1."""

    def own_type_rules(self, context):
        return context.deterministic({"c": (self.c, ((),))})


    name = "constant"

    def __init__(self, c=None):
        self.c = as_dist(Uniform(-10, 10) if c is None else c)

    def label(self):
        return f"constant({self.c.describe()})"

    def params(self):
        return {"c": self.c}

    def run(self, s):
        return s.draw(self.c) * _on(s.t)

    def run_batch(self, b):
        return _batch_scalar(b.draw(self.c)) * _on(b.t).float()


class Trend(Source):
    """Deterministic linear trend y_t = b * t, t = 1, 2, ..."""

    def own_type_rules(self, context):
        return context.deterministic({"b": (self.b, ((),))})


    name = "trend"

    def __init__(self, b=None):
        self.b = as_dist(Uniform(-1, 1) if b is None else b)

    def label(self):
        return f"trend({self.b.describe()})"

    def params(self):
        return {"b": self.b}

    def run(self, s):
        return s.draw(self.b) * np.where(s.t >= 1, s.t, 0)

    def run_batch(self, b):
        return _batch_scalar(b.draw(self.b)) * b.t.clamp_min(0)


class DampedTrend(Source):
    """Deterministic damped trend y_t = b (phi + phi^2 + ... + phi^t), t = 1, 2, ..."""

    def own_type_rules(self, context):
        return context.deterministic({"b": (self.b, ((),)), "phi": (self.phi, ((),))})


    name = "damped_trend"

    def __init__(self, b=None, phi=0.9):
        self.b = as_dist(Uniform(-1, 1) if b is None else b)
        self.phi = as_dist(phi)

    def label(self):
        return f"damped_trend({self.b.describe()}, phi={self.phi.describe()})"

    def params(self):
        return {"b": self.b, "phi": self.phi}

    def run(self, s):
        b, phi = s.draw(self.b), s.draw(self.phi)
        steps = np.maximum(s.t, 0).astype(int)                # t <= 0: before the start
        prefix = np.r_[0.0, np.cumsum(phi ** np.arange(1, int(steps.max(initial=0)) + 1))]
        return b * prefix[steps]                              # b (phi + ... + phi^t)

    def run_batch(self, b):
        slope, phi = _batch_scalar(b.draw(self.b)), _batch_scalar(b.draw(self.phi))
        t = b.t.clamp_min(0)
        geometric = torch.where(phi < 1, phi * (1 - phi ** t) / (1 - phi).clamp_min(1e-12), t)
        return slope * geometric


class Centered(Distribution):
    """k i.i.d. draws from `dist`, minus their mean: a zero-sum seasonal pattern."""

    def own_support(self, context):
        context.visit(self.dist)
        return context.region(self.shape, "centered_vector")


    def __init__(self, dist: Distribution, k: int):
        self.dist, self.k = dist, int(k)
        self.shape = (self.k,)

    def sample(self, rng, draw):
        v = self.dist.sample_iid(rng, self.k)
        return v - v.mean()

    def sample_batch(self, gen, n, device, draw):
        v = self.dist.sample_iid_batch(gen, (n, self.k), device)
        return v - v.mean(1, keepdim=True)

    def describe(self):
        return f"centered({self.dist.describe()}, {self.k})"


class Pattern(Source):
    """Fixed seasonal pattern, zero-sum over one period: y_t = p[(t - 1) mod period]."""

    def own_type_rules(self, context):
        return context.deterministic({"values": (self.values, ((self.period,),))}, period=self.period)


    name = "pattern"

    def __init__(self, period: int, values=None):
        self.period = count(period, "period", least=1)
        self.values = as_dist(Centered(Uniform(-1, 1), period) if values is None else values)

    def label(self):
        return f"pattern({self.period})"

    def params(self):
        return {"values": self.values}

    def run(self, s):
        p = np.asarray(s.draw(self.values))
        phase = (s.t - 1) % self.period
        return np.where(s.t >= 1, p[phase], 0.0)

    def run_batch(self, b):
        p = b.draw(self.values)                                    # [n, period]
        phase = torch.remainder(b.t - 1, self.period).long()
        return torch.where(b.t >= 1, p.gather(1, phase), torch.zeros_like(b.t))


# ---------------------------------------------------------------------------------------
# atomic primitives
# ---------------------------------------------------------------------------------------
STEADY_STATE = 100          # pre-sample steps for a filter's transient to die out
STEADY_STATE_SEASONS = 40   # extra seasons for a seasonal AR (its memory is period-times longer)


def _shift(x: torch.Tensor, k: int) -> torch.Tensor:
    """x_{t-k} along the last axis, zeros shifted in."""
    if k == 0:
        return x
    return torch.nn.functional.pad(x, (k, 0))[..., :x.shape[-1]]


class AR(Transform):
    """y_t = sum_i phi_i y_{t - i*period} + x_t.

    init="steady" (default): the filter runs through the pre-sample period, so it is in its
    steady state at t = 1. init="zero": it starts at t = 1 from zero, like an integrator
    (used where a recursion has an explicit starting state, e.g. the damped ETS slope)."""

    def own_type_rules(self, context):
        return context.coefficients(self.source, self.phi, self.p, self.period, "phi")


    name = "ar"

    def __init__(self, p: int, period: int = 1, phi=None, init: str = "steady"):
        super().__init__()
        if init not in ("steady", "zero"):
            raise ValueError('init must be "steady" or "zero"')
        self.p, self.period, self.init = count(p, "p"), count(period, "period", least=1), init
        self.phi = _coefficients(Stationary(self.p) if phi is None else phi, self.p, "ar phi")

    def label(self):
        return f"ar({self.p}" + (f", period={self.period})" if self.period > 1 else ")")

    def params(self):
        return {"phi": self.phi}

    def own_presample(self, s):
        if self.init == "zero":
            return 0
        return STEADY_STATE + (STEADY_STATE_SEASONS * self.period if self.period > 1 else 0)

    def own_presample_batch(self, b):
        return self.own_presample(None)

    def apply(self, s, x):
        phi = _drawn_coefficients(s.draw(self.phi), self.p, "ar phi")
        if self.init == "zero":
            x = np.where(_on(s.t), x, 0.0)
        y = np.zeros(len(x))
        for t in range(len(x)):
            y[t] = x[t] + sum(phi[i] * y[t - (i + 1) * self.period]
                              for i in range(self.p) if t - (i + 1) * self.period >= 0)
        return y

    def apply_batch(self, b, x):
        """The same recursion as a convolution with the filter's impulse response,
        y_t = sum_{k <= t} h_k x_{t-k} (zero initial conditions). Only h_0 .. h_{W-1} are
        needed (`_impulse_response`, in float64); the convolution is an FFT of length >= 2W,
        so nothing wraps around. For any stationary coefficients, near-unit and repeated
        roots included (float32 agreement with `apply` ~1e-5 relative to the peak). An
        explosive AR (a root inside the unit circle) is simulated correctly only by the
        reference backend."""
        phi = _drawn_coefficients(b.draw(self.phi), self.p, "ar phi")
        if self.init == "zero":
            x = torch.where(_on(b.t), x, torch.zeros_like(x))
        W = x.shape[-1]
        h = _impulse_response(phi, self.period, W)
        N = 1 << math.ceil(math.log2(2 * W))
        return torch.fft.irfft(torch.fft.rfft(x, N) * torch.fft.rfft(h, N), N)[:, :W]


def _impulse_response(phi: torch.Tensor, period: int, W: int) -> torch.Tensor:
    """h_0 .. h_{W-1} of 1 / phi(B), phi(B) = 1 - sum_i phi_i B^(i*period), by exponential
    windowing: g_k = rho^k h_k has the DFT 1 / phi(rho z) on the unit circle. A stationary
    filter's poles lie outside the unit circle, so 1 / phi(rho z) has no pole nearby, and
    g decays at least like rho^k, so a DFT of length N recovers it without wrap-around
    (rho^N = 1e-12). Then h_k = g_k / rho^k.

    Computed in float64: near a repeated root phi(rho z) is a sum of large coefficients
    cancelling to a tiny value (condition number ~1e8 for five roots at 0.95), which
    float32 cannot represent. phi(rho z) is evaluated term by term at the DFT frequencies,
    and with float64's precision N >= 2W suffices (the 1 / rho^k factor is at most 1e6)."""
    n, p = phi.shape
    N = 1 << math.ceil(math.log2(2 * W))
    rho = 10.0 ** (-12.0 / N)
    lags = torch.arange(1, p + 1, device=phi.device, dtype=torch.float64) * period
    omega = 2 * math.pi * torch.arange(N // 2 + 1, device=phi.device, dtype=torch.float64) / N
    terms = torch.polar(rho ** lags[:, None], -lags[:, None] * omega)          # rho^l e^{-i omega l}: [p, F]
    poly = 1 - phi.double().to(torch.complex128) @ terms                         # phi(rho e^{i omega}): [n, F]
    g = torch.fft.irfft(1 / poly, N)[:, :W]
    k = torch.arange(W, device=phi.device, dtype=torch.float64)
    return (g * rho ** -k).to(phi.dtype)


class MA(Transform):
    """y_t = x_t + sum_j theta_j x_{t - j*period}."""

    def own_type_rules(self, context):
        return context.coefficients(self.source, self.theta, self.q, self.period, "theta")


    name = "ma"

    def __init__(self, q: int, period: int = 1, theta=None):
        super().__init__()
        self.q, self.period = count(q, "q"), count(period, "period", least=1)
        self.theta = _coefficients(Invertible(self.q) if theta is None else theta, self.q, "ma theta")

    def label(self):
        return f"ma({self.q}" + (f", period={self.period})" if self.period > 1 else ")")

    def params(self):
        return {"theta": self.theta}

    def own_presample(self, s):
        return self.q * self.period

    def own_presample_batch(self, b):
        return self.q * self.period

    def apply(self, s, x):
        theta = _drawn_coefficients(s.draw(self.theta), self.q, "ma theta")
        y = np.array(x, dtype=float)
        for t in range(len(x)):
            for j in range(self.q):
                if t - (j + 1) * self.period >= 0:
                    y[t] += theta[j] * x[t - (j + 1) * self.period]
        return y

    def apply_batch(self, b, x):
        theta = _drawn_coefficients(b.draw(self.theta), self.q, "ma theta")
        return x + sum(theta[:, j:j + 1] * _shift(x, (j + 1) * self.period) for j in range(self.q))


class Integrate(Transform):
    """Apply y_t = y_{t - period} + x_t, d times, starting from zero at t = 1. d may be
    random (an integer-valued distribution). A nonzero starting state is added explicitly:
    `(x >> integrate()) + constant(l0)`, or `+ pattern(m, s0)` for period m."""

    def own_type_rules(self, context):
        return context.integrate(self.source, self.d, self.period)


    name = "integrate"

    def __init__(self, d=1, period: int = 1):
        super().__init__()
        self.d, self.period = _order_dist(d, "integrate d"), count(period, "period", least=1)

    def label(self):
        d = self.d.describe()
        return f"integrate({d}" + (f", period={self.period})" if self.period > 1 else ")")

    def params(self):
        return {} if isinstance(self.d, Fixed) else {"d": self.d}

    def apply(self, s, x):
        y = np.where(_on(s.t), x, 0.0)
        for _ in range(count(s.draw(self.d), "drawn integrate d")):
            out = np.zeros(len(y))
            for t in range(len(y)):
                out[t] = y[t] + (out[t - self.period] if t - self.period >= 0 else 0.0)
            y = out
        return y

    def apply_batch(self, b, x):
        d = b.draw(self.d).reshape(b.n)
        d_max, invalid = torch.stack([d.max(), ((d != d.round()) | (d < 0)).any().float()]).tolist()
        if invalid:                                  # one transfer for both (the loop needs d_max)
            raise ValueError("drawn integrate d must be integers >= 0")
        y = torch.where(_on(b.t), x, torch.zeros_like(x))
        out = torch.where((d == 0).unsqueeze(1), y, torch.zeros_like(y))
        for k in range(1, int(d_max) + 1):
            y = self._integrate_once(y)
            out = torch.where((d == k).unsqueeze(1), y, out)
        return out

    def _integrate_once(self, y):
        if self.period == 1:
            return torch.cumsum(y, 1)
        n, W = y.shape
        pad = (-W) % self.period
        z = torch.nn.functional.pad(y, (0, pad)).reshape(n, -1, self.period)
        return torch.cumsum(z, 1).reshape(n, -1)[:, :W]


class Normalize(Transform):
    """Location-scale normalization by statistics of the OBSERVED window (t = 1..n),
    applied to the whole series, future included. Causal in the forecasting sense: the
    statistics never use the future. kind: "minmax" (min, max - min), "std" (mean,
    standard deviation) or "robust" (median, 1.4826 * median absolute deviation). Puts
    operands on a common scale before they are combined."""

    def own_type_rules(self, context):
        return context.filter(self.source)


    name = "normalize"
    KINDS = ("minmax", "std", "robust")

    def __init__(self, kind: str = "minmax"):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"normalize kind must be one of {self.KINDS}")
        self.kind = kind

    def label(self):
        return f"normalize({self.kind})"

    def apply(self, s, x):
        v = np.asarray(x)[(s.t >= 1) & (s.t <= s.t[-1] - s.horizon)]
        if self.kind == "minmax":
            loc, scale = v.min(), v.max() - v.min()
        elif self.kind == "std":
            loc, scale = v.mean(), v.std()
        else:
            loc = np.median(v)
            scale = 1.4826 * np.median(np.abs(v - loc))
        return (x - loc) / max(scale, 1e-8)

    def apply_batch(self, b, x):
        observed = (b.t >= 1) & (b.t <= b.t[:, -1:] - b.horizon)
        if self.kind == "minmax":
            lo = torch.where(observed, x, torch.full_like(x, float("inf"))).amin(1, keepdim=True)
            hi = torch.where(observed, x, torch.full_like(x, float("-inf"))).amax(1, keepdim=True)
            loc, scale = lo, hi - lo
        elif self.kind == "std":
            n = observed.sum(1, keepdim=True).clamp_min(1)
            loc = torch.where(observed, x, torch.zeros_like(x)).sum(1, keepdim=True) / n
            scale = (torch.where(observed, (x - loc) ** 2, torch.zeros_like(x)).sum(1, keepdim=True) / n).sqrt()
        else:
            v = torch.where(observed, x, torch.full_like(x, float("nan")))
            loc = torch.nanquantile(v, 0.5, dim=1, keepdim=True)          # numpy-style median
            scale = 1.4826 * torch.nanquantile((v - loc).abs(), 0.5, dim=1, keepdim=True)
        return (x - loc) / scale.clamp_min(1e-8)


class Difference(Transform):
    """Apply y_t = x_t - x_{t - period}, d times."""

    def own_type_rules(self, context):
        return context.difference(self.source, self.d, self.period)


    name = "difference"

    def __init__(self, d: int = 1, period: int = 1):
        super().__init__()
        self.d, self.period = count(d, "difference d"), count(period, "period", least=1)

    def label(self):
        return f"difference({self.d}" + (f", period={self.period})" if self.period > 1 else ")")

    def apply(self, s, x):
        y = np.asarray(x, float)
        for _ in range(self.d):
            y = y - np.concatenate([np.zeros(self.period), y[:-self.period]])[:len(y)]
        return y

    def apply_batch(self, b, x):
        for _ in range(self.d):
            x = x - _shift(x, self.period)
        return x


class Exp(Transform):
    """y_t = exp(rate * x_t): turns an additive world into a positive one whose dynamics are
    multiplicative on the original scale (the additive world on the log scale). Exponentiated
    additive ETS is log-scale ETS, not the classical multiplicative ETS families: for SES,
    Y_t = L_{t-1} exp(e_t) and L_t = L_{t-1} exp(alpha e_t), whereas ETS(M,N,N) has
    Y_t = L_{t-1} (1 + u_t) and L_t = L_{t-1} (1 + alpha u_t); no choice of u_t matches both.

    The input's scale matters: integrated worlds reach values in the thousands. Standardize
    first, `world >> tg.normalize("std") >> tg.exp()`; the rate is then the spread of the
    series on the log scale over the observed window (how multiplicative it is), U(0.05, 0.7)
    by default, which covers the growth of seasonal swings in the M and Tourism data."""

    def own_type_rules(self, context):
        context.shape(self.rate, ((),), "rate")
        return context.filter(self.source)


    name = "exp"

    def __init__(self, rate=None):
        super().__init__()
        self.rate = as_dist(Uniform(0.05, 0.7) if rate is None else rate)

    def label(self):
        return f"exp(rate={self.rate.describe()})"

    def params(self):
        return {"rate": self.rate}

    def apply(self, s, x):
        return np.exp(float(s.draw(self.rate)) * np.asarray(x, float))

    def apply_batch(self, b, x):
        return torch.exp(_batch_scalar(b.draw(self.rate)) * x)


class Indicator(Transform):
    """y_t = 1 if x_t > 0 else 0: an occurrence process from a latent one. A persistent latent
    process gives occurrence that drifts over time; a seasonal one, occurrence with a season."""

    def own_type_rules(self, context):
        return context.filter(self.source)

    name = "indicator"

    def label(self):
        return "indicator"

    def params(self):
        return {}

    def apply(self, s, x):
        return (np.asarray(x, float) > 0).astype(float)

    def apply_batch(self, b, x):
        return (x > 0).to(x.dtype)


class Counts(Transform):
    """y_t = max(round(x_t), 0): non-negative integer counts, such as units sold."""

    def own_type_rules(self, context):
        return context.filter(self.source)

    name = "counts"

    def label(self):
        return "counts"

    def params(self):
        return {}

    def apply(self, s, x):
        return np.maximum(np.round(np.asarray(x, float)), 0.0)

    def apply_batch(self, b, x):
        return torch.clamp(torch.round(x), min=0.0)


class Lag(Transform):
    """y_t = x_{t - k}."""

    def own_type_rules(self, context):
        return context.filter(self.source, preserve=True)


    name = "lag"

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = count(k, "lag k")

    def label(self):
        return f"lag({self.k})"

    def apply(self, s, x):
        return np.concatenate([np.zeros(self.k), np.asarray(x, float)])[:len(x)]

    def apply_batch(self, b, x):
        return _shift(x, self.k)


# ---------------------------------------------------------------------------------------
# constructors (the public names)
# ---------------------------------------------------------------------------------------
def noise(shape=None, scale=1.0):
    return Noise(shape, scale)


def constant(c=None):
    return Constant(c)


def trend(b=None):
    return Trend(b)


def damped_trend(b=None, phi=0.9):
    return DampedTrend(b, phi)


def pattern(period, values=None):
    return Pattern(period, values)


def ar(p, period=1, phi=None, init="steady"):
    return AR(p, period, phi, init)


def ma(q, period=1, theta=None):
    return MA(q, period, theta)


def integrate(d=1, period=1):
    return Integrate(d, period)


def normalize(kind="minmax"):
    return Normalize(kind)


def difference(d=1, period=1):
    return Difference(d, period)


def lag(k=1):
    return Lag(k)


def exp(rate=None):
    return Exp(rate)


def indicator():
    return Indicator()


def counts():
    return Counts()
