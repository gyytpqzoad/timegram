"""Distributions: noise shapes and parameter distributions.

One class serves both roles.

* As a **noise shape** (`tg.noise(tg.student_t(3))`) a distribution produces i.i.d.
  draws, one per time step.
* As a **parameter** (`tg.ar(1, phi=tg.uniform(-0.5, 0.5))`) it is a random variable:
  it is drawn once per series, and every place that uses the *same object* sees the
  same draw. This is the parameter counterpart of shared blocks: in
  `w * a + (1 - w) * b` both terms see one `w`.

Arithmetic on distributions builds derived random variables (`Expr`), so `1 - w` and
`alpha - 1` are parameters too.

Every distribution has a plain NumPy sampler for one series (`sample`, the reference
backend) and, optionally, a batched torch sampler (`sample_batch`, the fast backend).
`draw` is a callback that returns the (memoized) value of another random variable; it
is how derived and joint variables see their parents.
"""
from __future__ import annotations

import math
import operator
from typing import Callable

import numpy as np
import torch


class Distribution:

    def own_support(self, context):
        context.claim("parameter_region", "unverified", evidence="no supported local support rule")
        return context.parameter(self.shape)

    shape: tuple[int, ...] = ()          # event shape: () scalar, (k,) vector

    # -- reference backend: one series ------------------------------------------------
    def sample(self, rng: np.random.Generator, draw: Callable):
        return self.sample_iid(rng, None)

    def sample_iid(self, rng: np.random.Generator, size):
        raise NotImplementedError(f"{self} cannot be used as a noise shape")

    # -- fast backend: n series at once -----------------------------------------------
    def sample_batch(self, gen: torch.Generator, n: int, device, draw: Callable) -> torch.Tensor:
        return self.sample_iid_batch(gen, (n,), device)

    def sample_iid_batch(self, gen: torch.Generator, size: tuple, device) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no batched sampler (fast backend)")

    # -- structure ----------------------------------------------------------------------
    def parents(self) -> tuple["Distribution", ...]:
        return ()

    def __repr__(self) -> str:
        return self.describe()

    def describe(self) -> str:
        return type(self).__name__.lower()

    # -- arithmetic: derived random variables --------------------------------------------
    # (operator functions rather than lambdas, so models can be pickled)
    def __add__(self, o):
        return Expr(operator.add, (self, o), "+")

    def __radd__(self, o):
        return Expr(operator.add, (o, self), "+")

    def __sub__(self, o):
        return Expr(operator.sub, (self, o), "-")

    def __rsub__(self, o):
        return Expr(operator.sub, (o, self), "-")

    def __mul__(self, o):
        from .model import Block                        # w * block is scaling, not arithmetic
        if isinstance(o, Block):
            return NotImplemented
        return Expr(operator.mul, (self, o), "*")

    def __rmul__(self, o):
        return Expr(operator.mul, (o, self), "*")

    def __truediv__(self, o):
        return Expr(operator.truediv, (self, o), "/")

    def __neg__(self):
        return Expr(operator.neg, (self,), "neg")


def _require(ok, message: str):
    if not ok:
        raise ValueError(message)


def _finite(*values) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def count(value, name: str, least: int | None = 0) -> int:
    """A structural integer (an order, a period): exactly an integer, at least `least`
    (no bound if None)."""
    ok = not isinstance(value, bool) and _finite(value) and float(value) == int(value)
    _require(ok and (least is None or int(value) >= least),
             f"{name} must be an integer" + ("" if least is None else f" >= {least}") + f", not {value!r}")
    return int(value)


def _fmt(values) -> str:
    return "[" + ", ".join(f"{float(v):.3g}" for v in values) + "]"


def numpy_rng(gen: torch.Generator) -> np.random.Generator:
    """A NumPy generator seeded from a torch generator (for samplers torch lacks)."""
    return np.random.default_rng(int(torch.randint(0, 2**62, (1,), generator=gen, device=gen.device)))


def as_dist(x) -> Distribution:
    """Numbers (and arrays) become fixed values; distributions pass through."""
    return x if isinstance(x, Distribution) else Fixed(x)


# ---------------------------------------------------------------------------------------
# scalar distributions
# ---------------------------------------------------------------------------------------
class Fixed(Distribution):

    def own_support(self, context):
        return context.fixed(self.value, self.shape)

    def __init__(self, value):
        self.value = value
        self.shape = np.shape(value)

    def sample(self, rng, draw):
        return np.array(self.value, dtype=float) if self.shape else float(self.value)

    def sample_batch(self, gen, n, device, draw):
        v = torch.as_tensor(self.value, dtype=torch.float32, device=device)
        return v.expand(n, *v.shape).clone()

    def describe(self):
        return f"{self.value}"


class Uniform(Distribution):

    def own_support(self, context):
        return context.real(interval=(self.low, self.high),
                            integer=self.low == self.high and float(self.low).is_integer())

    def __init__(self, low=0.0, high=1.0):
        _require(_finite(low, high) and low <= high, f"uniform needs finite low <= high, not ({low}, {high})")
        self.low, self.high = low, high

    def sample_iid(self, rng, size):
        return rng.uniform(self.low, self.high, size)

    def sample_iid_batch(self, gen, size, device):
        return self.low + (self.high - self.low) * torch.rand(size, generator=gen, device=device)

    def describe(self):
        return f"uniform({self.low}, {self.high})"


class Normal(Distribution):

    def own_support(self, context):
        return context.real(interval=(self.mu, self.mu) if self.sd == 0 else None,
                            integer=self.sd == 0 and float(self.mu).is_integer())

    def __init__(self, mu=0.0, sd=1.0):
        _require(_finite(mu, sd) and sd >= 0, f"normal needs a finite mean and sd >= 0, not ({mu}, {sd})")
        self.mu, self.sd = mu, sd

    def sample_iid(self, rng, size):
        return rng.normal(self.mu, self.sd, size)

    def sample_iid_batch(self, gen, size, device):
        return self.mu + self.sd * torch.randn(size, generator=gen, device=device)

    def describe(self):
        return f"normal({self.mu}, {self.sd})"


class StudentT(Distribution):

    def own_support(self, context):
        return context.real(finite_variance=self.df > 2)

    def __init__(self, df):
        _require(_finite(df) and df > 0, f"student_t needs finite df > 0, not {df}")
        self.df = df

    def sample_iid(self, rng, size):
        return rng.standard_t(self.df, size)

    def sample_iid_batch(self, gen, size, device):
        # torch's chi-square sampler takes no Generator; seed NumPy from `gen` instead
        return torch.as_tensor(self.sample_iid(numpy_rng(gen), size), dtype=torch.float32, device=device)

    def describe(self):
        return f"student_t({self.df})"


class Laplace(Distribution):

    def own_support(self, context):
        return context.real(interval=(self.loc, self.loc) if self.scale == 0 else None,
                            integer=self.scale == 0 and float(self.loc).is_integer())

    def __init__(self, loc=0.0, scale=1.0):
        _require(_finite(loc, scale) and scale >= 0, f"laplace needs a finite loc and scale >= 0, not ({loc}, {scale})")
        self.loc, self.scale = loc, scale

    def sample_iid(self, rng, size):
        return rng.laplace(self.loc, self.scale, size)

    def sample_iid_batch(self, gen, size, device):
        u = torch.rand(size, generator=gen, device=device) - 0.5
        # rand can be exactly zero. Move only the resulting singular endpoint;
        # every finite legacy draw and the generator's consumption stay unchanged.
        inner = torch.nextafter(u.new_tensor(-0.5), u.new_tensor(0.0))
        u = torch.where(u == -0.5, inner, u)
        return self.loc - self.scale * torch.sign(u) * torch.log1p(-2 * u.abs())

    def describe(self):
        return f"laplace({self.loc}, {self.scale})"


class Bernoulli(Distribution):
    """1 with probability p, else 0: an occurrence law. `tg.noise(tg.bernoulli(p)) * size` is
    intermittent demand (Croston's occurrence times a positive size process)."""

    def own_support(self, context):
        return context.real(interval=(0.0, 1.0) if 0 < self.p < 1 else (self.p, self.p), integer=True)

    def __init__(self, p=0.5):
        _require(_finite(p) and 0 <= p <= 1, f"bernoulli needs 0 <= p <= 1, not {p}")
        self.p = p

    def sample_iid(self, rng, size):
        return (rng.random(size) < self.p).astype(float) if size is not None else float(rng.random() < self.p)

    def sample_iid_batch(self, gen, size, device):
        return (torch.rand(size, generator=gen, device=device) < self.p).float()

    def describe(self):
        return f"bernoulli({self.p})"


class Mixture(Distribution):
    """Each draw comes from component k with probability weights[k]."""

    def __init__(self, *components, weights=None):
        if not components or not all(isinstance(c, Distribution) and not c.shape for c in components):
            raise ValueError("mixture needs at least one scalar distribution component")
        self.components = components
        w = np.ones(len(components)) if weights is None else np.asarray(weights, float)
        if w.shape != (len(components),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("mixture weights: one finite non-negative weight per component, positive total")
        self.weights = w / w.sum()

    def sample_iid(self, rng, size):
        k = rng.choice(len(self.components), size=size, p=self.weights)
        draws = [c.sample_iid(rng, size) for c in self.components]
        return np.choose(k, draws) if size is not None else draws[k]

    def sample_iid_batch(self, gen, size, device):
        w = torch.as_tensor(self.weights, dtype=torch.float32, device=device)
        k = torch.multinomial(w, math.prod(size), replacement=True, generator=gen).reshape(size)
        draws = torch.stack([c.sample_iid_batch(gen, size, device) for c in self.components])
        return draws.gather(0, k.unsqueeze(0)).squeeze(0)

    def describe(self):
        inner = ", ".join(c.describe() for c in self.components)
        return f"mixture({inner}, weights={_fmt(self.weights)})"


class Integers(Distribution):
    """Uniform on the integers low..high (both included)."""

    def own_support(self, context):
        return context.real(interval=(self.low, self.high), integer=True)


    def __init__(self, low, high):
        self.low, self.high = count(low, "integers low", least=None), count(high, "integers high", least=None)
        _require(self.low <= self.high, f"integers needs low <= high, not ({low}, {high})")

    def sample_iid(self, rng, size):
        return rng.integers(self.low, self.high + 1, size)

    def sample_iid_batch(self, gen, size, device):
        return torch.randint(self.low, self.high + 1, size, generator=gen, device=device).float()

    def describe(self):
        return f"integers({self.low}, {self.high})"


class Choice(Distribution):
    """One of `values`, with probabilities `probs` (uniform by default)."""

    def own_support(self, context):
        return context.finite_choice(self.values)


    def __init__(self, values, probs=None):
        self.values = list(values)
        p = np.ones(len(self.values)) if probs is None else np.asarray(probs, float)
        _require(self.values and p.shape == (len(self.values),) and np.isfinite(p).all()
                 and (p >= 0).all() and p.sum() > 0,
                 "choice needs values and one finite non-negative probability each, positive total")
        self.probs = p / p.sum()

    def sample_iid(self, rng, size):
        idx = rng.choice(len(self.values), size=size, p=self.probs)
        return np.asarray(self.values)[idx]

    def sample_iid_batch(self, gen, size, device):
        p = torch.as_tensor(self.probs, dtype=torch.float32, device=device)
        idx = torch.multinomial(p, math.prod(size), replacement=True, generator=gen).reshape(size)
        return torch.as_tensor(self.values, dtype=torch.float32, device=device)[idx]

    def describe(self):
        return f"choice({self.values})"


# ---------------------------------------------------------------------------------------
# vector distributions for AR / MA coefficients
# ---------------------------------------------------------------------------------------
def pacf_to_coefs(pacf: np.ndarray) -> np.ndarray:
    """Durbin-Levinson: partial autocorrelations in (-1, 1)^k -> stationary AR(k)
    coefficients (for y_t = sum phi_i y_{t-i} + e_t). The same map gives invertible MA
    coefficients. Trailing zero partial autocorrelations give trailing zero coefficients."""
    phi = np.zeros(0)
    for a in pacf:
        phi = np.concatenate([phi - a * phi[::-1], [a]])
    return phi


def pacf_to_coefs_batch(pacf: torch.Tensor) -> torch.Tensor:
    """`pacf_to_coefs` for a batch [n, k]."""
    n, k = pacf.shape
    phi = pacf.new_zeros(n, 0)
    for j in range(k):
        a = pacf[:, j:j + 1]
        phi = torch.cat([phi - a * phi.flip(1), a], 1)
    return phi


class Stationary(Distribution):
    """Uniform over the stationary region of an AR(k), y_t = sum phi_i y_{t-i} + x_t, in
    partial-autocorrelation coordinates: each partial autocorrelation is uniform on
    (-pacf_max, pacf_max). The default for `tg.ar`."""

    def own_support(self, context):
        return context.region(self.shape, "stationary")


    def __init__(self, k: int, pacf_max: float = 1.0):
        _require(_finite(pacf_max) and 0 < pacf_max <= 1, f"pacf_max must lie in (0, 1], not {pacf_max}")
        self.k, self.pacf_max = count(k, "k"), pacf_max
        self.shape = (self.k,)

    def sample(self, rng, draw):
        return pacf_to_coefs(rng.uniform(-self.pacf_max, self.pacf_max, self.k))

    def sample_batch(self, gen, n, device, draw):
        u = torch.rand(n, self.k, generator=gen, device=device)
        return pacf_to_coefs_batch(self.pacf_max * (2 * u - 1))

    def describe(self):
        return f"stationary({self.k}, pacf_max={self.pacf_max})"


class StationaryOrder(Stationary):
    """Stationary AR(k) coefficients whose last partial autocorrelation is at least
    `last_min` in magnitude, so the process is not effectively of lower order (needed when
    the order itself is the estimand). The others are uniform on (-pacf_max, pacf_max)."""

    def __init__(self, k: int, last_min: float, pacf_max: float = 1.0):
        super().__init__(k, pacf_max)
        _require(self.k >= 1, "last_min needs an order k >= 1: an AR(0) has no last partial autocorrelation")
        _require(_finite(last_min) and 0 < last_min < pacf_max,
                 f"last_min must lie in (0, pacf_max), not {last_min}")
        self.last_min = last_min

    def _last(self, u, sign):
        return sign * (self.last_min + (self.pacf_max - self.last_min) * u)

    def sample(self, rng, draw):
        pacf = rng.uniform(-self.pacf_max, self.pacf_max, self.k)
        pacf[-1] = self._last(rng.uniform(), 1.0 if rng.uniform() < 0.5 else -1.0)
        return pacf_to_coefs(pacf)

    def sample_batch(self, gen, n, device, draw):
        pacf = self.pacf_max * (2 * torch.rand(n, self.k, generator=gen, device=device) - 1)
        u = torch.rand(n, generator=gen, device=device)
        sign = torch.where(torch.rand(n, generator=gen, device=device) < 0.5, 1.0, -1.0)
        pacf[:, -1] = self._last(u, sign)
        return pacf_to_coefs_batch(pacf)

    def describe(self):
        return f"stationary({self.k}, pacf_max={self.pacf_max}, last_min={self.last_min})"


class Invertible(Stationary):
    """Uniform over the invertible region of an MA(k) in timegram's convention,
    y_t = x_t + sum theta_j x_{t-j}, in partial-autocorrelation coordinates. The MA
    polynomial 1 + sum theta_j B^j has the AR form 1 - sum c_j B^j with c = -theta, so
    theta is minus a stationary AR draw. The default for `tg.ma`."""

    def own_support(self, context):
        return context.region(self.shape, "invertible")


    def sample(self, rng, draw):
        return -super().sample(rng, draw)

    def sample_batch(self, gen, n, device, draw):
        return -super().sample_batch(gen, n, device, draw)

    def describe(self):
        return f"invertible({self.k}, pacf_max={self.pacf_max})"


class L1Ball(Distribution):
    """Coefficients inside the L1 ball sum |c_i| < 1 (a sufficient condition for
    stationarity / invertibility): a uniform direction on the ball's surface, scaled by a
    uniform radius in (0, 1)."""

    def own_support(self, context):
        return context.region(self.shape, "l1")


    def __init__(self, k: int):
        self.k = count(k, "k")
        self.shape = (self.k,)

    def sample(self, rng, draw):
        d = rng.uniform(-1, 1, self.k)
        return d / np.abs(d).sum() * rng.uniform()

    def sample_batch(self, gen, n, device, draw):
        d = 2 * torch.rand(n, self.k, generator=gen, device=device) - 1
        r = torch.rand(n, 1, generator=gen, device=device)
        return d / d.abs().sum(1, keepdim=True).clamp_min(1e-12) * r

    def describe(self):
        return f"l1_ball({self.k})"


class Dirichlet(Distribution):
    """Probability vector of length k; concentration 1 is uniform on the simplex."""

    def own_support(self, context):
        return context.region(self.shape, "simplex")


    def __init__(self, k: int, concentration: float = 1.0):
        _require(_finite(concentration) and concentration > 0, f"concentration must be > 0, not {concentration}")
        self.k, self.concentration = count(k, "k", least=1), concentration
        self.shape = (self.k,)

    def sample(self, rng, draw):
        return rng.dirichlet(np.full(self.k, self.concentration))

    def sample_batch(self, gen, n, device, draw):
        if self.concentration == 1.0:                 # normalized exponentials
            e = -torch.log(torch.rand(n, self.k, generator=gen, device=device).clamp_min(1e-12))
            return e / e.sum(1, keepdim=True)
        return torch.as_tensor(numpy_rng(gen).dirichlet(np.full(self.k, self.concentration), n),
                               dtype=torch.float32, device=device)

    def describe(self):
        return f"dirichlet({self.k})"


# ---------------------------------------------------------------------------------------
# derived and joint random variables
# ---------------------------------------------------------------------------------------
class Expr(Distribution):
    """A deterministic function of other random variables (and constants)."""

    def own_support(self, context):
        return context.expression(self.args, self.shape)


    def __init__(self, fn, args, label):
        self.fn, self.args, self.label = fn, tuple(as_dist(a) for a in args), label
        self.shape = max((a.shape for a in self.args), key=len)

    def parents(self):
        return self.args

    def sample(self, rng, draw):
        return self.fn(*(draw(a) for a in self.args))

    def sample_batch(self, gen, n, device, draw):
        """Per-series values carry a leading batch axis: a scalar is [n], a vector [n, k].
        Scalars get a trailing axis so that they broadcast against vectors, as they do
        for one series."""
        values = [draw(a) for a in self.args]
        rank = max((v.dim() for v in values if torch.is_tensor(v)), default=0)
        values = [v.reshape(*v.shape, *[1] * (rank - v.dim())) if torch.is_tensor(v) and v.dim() else v
                  for v in values]
        return self.fn(*values)

    def describe(self):
        a = [x.describe() for x in self.args]
        return f"-{a[0]}" if self.label == "neg" else f"({a[0]} {self.label} {a[1]})"


class Joint(Distribution):
    """Several parameters drawn together (constraints across parameters, e.g. beta < alpha
    in ETS, or the structure of a curated ARIMA prior). `sample` returns a dict; each
    entry is exposed as a parameter through `joint[key]`."""

    def own_support(self, context):
        return context.joint()


    keys: tuple[str, ...] = ()

    def shape_for(self, key: str) -> tuple[int, ...]:
        """Event shape of an entry; scalar unless a joint prior declares otherwise."""
        return ()

    def __getitem__(self, key: str) -> "JointKey":
        cache = self.__dict__.setdefault("_keys", {})
        if key not in cache:
            cache[key] = JointKey(self, key)
        return cache[key]


class JointKey(Distribution):

    def own_support(self, context):
        return context.joint_key(self.joint, self.key)

    def __init__(self, joint: Joint, key: str):
        self.joint, self.key = joint, key

    @property
    def shape(self) -> tuple[int, ...]:
        # Derived metadata also works for old pickles, without adding serialized
        # fields that would change forecast-only library keys.
        return self.joint.shape_for(self.key)

    def parents(self):
        return (self.joint,)

    def sample(self, rng, draw):
        return draw(self.joint)[self.key]

    def sample_batch(self, gen, n, device, draw):
        return draw(self.joint)[self.key]

    def describe(self):
        return f"{self.joint.describe()}.{self.key}"


# ---------------------------------------------------------------------------------------
# constructors (the public names)
# ---------------------------------------------------------------------------------------
def uniform(low=0.0, high=1.0):
    return Uniform(low, high)


def normal(mu=0.0, sd=1.0):
    return Normal(mu, sd)


def student_t(df):
    return StudentT(df)


def laplace(loc=0.0, scale=1.0):
    return Laplace(loc, scale)


def bernoulli(p=0.5):
    return Bernoulli(p)


def mixture(*components, weights=None):
    return Mixture(*components, weights=weights)


def integers(low, high):
    return Integers(low, high)


def choice(values, probs=None):
    return Choice(values, probs)


def dirichlet(k, concentration=1.0):
    return Dirichlet(k, concentration)


def stationary(k, pacf_max=1.0, last_min=0.0):
    """Uniform over the stationary region (partial autocorrelations); last_min > 0 keeps the
    last one away from zero, so the order is distinguishable."""
    return StationaryOrder(k, last_min, pacf_max) if last_min > 0 else Stationary(k, pacf_max)


def invertible(k, pacf_max=1.0):
    return Invertible(k, pacf_max)


def l1_ball(k):
    return L1Ball(k)
