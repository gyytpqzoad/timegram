"""Named compounds, each defined by its hand-built tree of atomic blocks. A compound *is* its tree: `apply` runs the tree with the compound's input in
place of the tree's `Input` placeholder. What a compound adds is a name, readable
parameter names, the usual parameter ranges, and an identifiability floor
(`min_length`).

    ets      one shared noise drives level, slope and season (single source of error)
    arima    noise >> ma >> seasonal ma >> ar >> seasonal ar (+ mean) >> integrate(s)
    local_level, local_trend, seasonal    unobserved components: one noise each
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .blocks import Centered, Input, ar, constant, damped_trend, integrate, lag, ma, noise, pattern, trend
from .dists import Distribution, Fixed, Invertible, Joint, Stationary, Uniform, as_dist
from .model import Transform, render_tree


class Compound(Transform):
    """A transform defined by a tree over an `Input` placeholder."""

    def own_type_rules(self, context):
        return context.compound(self.source, self.input, self.tree)


    def __init__(self):
        super().__init__()
        self.input = Input()
        self.tree = self.build(self.input)

    def build(self, e):
        raise NotImplementedError

    def apply(self, s, x):
        return s.scope({id(self.input): x}).value(self.tree)

    def apply_batch(self, b, x):
        return b.scope({id(self.input): x}).value(self.tree)

    def own_presample(self, s):
        return s.presample(self.tree)

    def own_presample_batch(self, b):
        return b.presample(self.tree)

    def floor(self, s) -> int:
        """Identifiability floor of the realized structure (override)."""
        return 1

    def own_min_length(self, s):
        return max(self.floor(s), s.min_length(self.tree))

    def own_min_length_batch(self, b):
        return torch.maximum(self.floor_batch(b), b.min_length(self.tree))

    def floor_batch(self, b) -> torch.Tensor:
        return torch.full((b.n,), float(self.floor(None)), device=b.device)

    def expanded(self) -> str:
        """The hand-built tree this compound stands for."""
        return render_tree(self.tree)


class Composite(Compound):
    """A chain of transforms with an open input, `a >> b >> c`: the tree
    input >> a >> b >> c. Its default source is the first stage's default noise."""

    name = "compose"

    def __init__(self, *stages):
        self.stages = tuple(stages)
        super().__init__()
        self.source = self.stages[0].source

    def build(self, e):
        from .model import Feed
        x = e
        for stage in self.stages:
            x = Feed(x, stage)
        return x

    def label(self):
        return "compose(" + ", ".join(s.label() for s in self.stages) + ")"

    def params(self):
        out, seen = {}, {}
        for stage in self.stages:
            prefix = stage.name if stage.name not in seen else f"{stage.name}{seen[stage.name]}"
            seen[stage.name] = seen.get(stage.name, 1) + 1
            out |= {f"{prefix}.{k}": v for k, v in stage.params().items()}
        return out


# ---------------------------------------------------------------------------------------
# exponential smoothing
# ---------------------------------------------------------------------------------------
def _ets_matrices(trend: bool, seasonal: bool, m: int | None):
    """F and w of the innovations state space x_t = F x_{t-1} + g e_t, y_t = w' x_{t-1} + e_t,
    with x = (level, [slope], [s_t, ..., s_{t-m+1}]); a damping factor multiplies the slope
    entries of F and w (filled in per draw)."""
    dim = 1 + trend + (m if seasonal else 0)
    F, w = np.zeros((dim, dim)), np.zeros(dim)
    F[0, 0], w[0] = 1.0, 1.0
    if trend:
        F[0, 1] = F[1, 1] = w[1] = 1.0
    if seasonal:
        k = 1 + trend
        F[k, dim - 1] = 1.0                            # s_t = s_{t-m} (+ gamma e_t)
        for j in range(1, m):
            F[k + j, k + j - 1] = 1.0
        w[dim - 1] = 1.0
    return F, w


def ets_second_modulus(alpha, beta, gamma, phi, trend, seasonal, m):
    """Largest eigenvalue modulus of D = F - g w' (reference: eigenvalues), excluding the
    structural unit eigenvalue of seasonal models (level / season redundancy)."""
    F, w = _ets_matrices(trend, seasonal, m)
    n, dim = len(alpha), len(w)
    Fs, ws = np.repeat(F[None], n, 0), np.repeat(w[None], n, 0)
    if trend:
        Fs[:, 0, 1] = Fs[:, 1, 1] = ws[:, 1] = phi
    g = np.zeros((n, dim))
    g[:, 0] = alpha
    if trend:
        g[:, 1] = beta
    if seasonal:
        g[:, 1 + trend] = gamma
    moduli = np.sort(np.abs(np.linalg.eigvals(Fs - g[:, :, None] * ws[:, None, :])), axis=1)[:, ::-1]
    return moduli[:, 1] if seasonal else moduli[:, 0]


def is_admissible_ets(alpha, beta, gamma, phi, trend, seasonal, m) -> bool:
    """Admissibility (Hyndman, Akram & Archibald 2008): every eigenvalue of F - g w'
    (except the structural one of seasonal models) inside the unit circle."""
    v = lambda x: np.array([x], float)                              # noqa: E731
    return bool(ets_second_modulus(v(alpha), v(beta), v(gamma), v(phi), trend, seasonal, m)[0] < 1 - 1e-9)


def ets_admissible_batch(alpha, beta, gamma, phi, trend, seasonal, m) -> torch.Tensor:
    """`is_admissible_ets` for a batch, without eigenvalues (fast backend).

    By the matrix determinant lemma det(I - D B) = det(I - F B) Psi(B), with
    det(I - F B) = (1 - B) [(1 - phi B)] [(1 - B^m)] and Psi the impulse response
    psi_0 = 1, psi_j = alpha + beta (phi + ... + phi^j) + gamma [m | j]. The eigenvalues
    of D are the roots of its reversal; after dividing out the structural root z = 1 of
    seasonal models, a Schur-Cohn test checks that all others lie inside the unit circle."""
    dev = alpha.device
    alpha, beta, gamma, phi = (x.double() for x in (alpha, beta, gamma, phi))
    n, deg = len(alpha), 1 + trend + (m if seasonal else 0)
    j = torch.arange(deg + 1, dtype=torch.float64, device=dev)
    psi = alpha[:, None].expand(n, deg + 1).clone()
    if trend:
        psi = psi + beta[:, None] * (torch.cumsum(phi[:, None] ** j, 1) - 1.0)   # phi + ... + phi^j
    if seasonal:
        psi = psi + gamma[:, None] * ((j % m == 0) & (j >= m)).double()
    psi[:, 0] = 1.0

    def times(p, coefs):
        out = torch.zeros(n, p.shape[1] + coefs.shape[1] - 1, dtype=torch.float64, device=dev)
        for i in range(coefs.shape[1]):
            out[:, i:i + p.shape[1]] += coefs[:, i:i + 1] * p
        return out

    one = torch.ones(n, 1, dtype=torch.float64, device=dev)
    poly = torch.cat([one, -one], 1)                                     # det(I - F B)
    if trend:
        poly = times(poly, torch.cat([one, -phi[:, None]], 1))
    if seasonal:
        poly = times(poly, torch.cat([one, torch.zeros(n, m - 1, dtype=torch.float64, device=dev), -one], 1))
    c = torch.stack([(poly[:, :k + 1].flip(1) * psi[:, :k + 1]).sum(1) for k in range(deg + 1)], 1)
    # c: characteristic polynomial of D, leading coefficient first; deflate z = 1 if seasonal
    p = (torch.cumsum(c[:, :deg], 1) if seasonal else c).flip(1)        # ascending powers
    ok = torch.ones(n, dtype=torch.bool, device=dev)
    for _ in range(p.shape[1] - 1):                                     # Schur-Cohn reduction
        a0, an = p[:, 0], p[:, -1]
        ok &= a0.abs() < an.abs() * (1 - 1e-9)
        p = an[:, None] * p[:, 1:] - a0[:, None] * p[:, :-1].flip(1)
        p = p / p[:, -1:].abs().clamp_min(1e-300)
    return ok


def hw_admissible_batch(alpha, beta, gamma, m) -> torch.Tensor:
    """ETS(A,A,A) case of `ets_admissible_batch`."""
    return ets_admissible_batch(alpha, beta, gamma, torch.ones_like(alpha), True, True, m)


def _hw_second_modulus(alpha, beta, gamma, period):
    return ets_second_modulus(alpha, beta, gamma, np.ones_like(alpha), True, True, period)


class EtsSmoothing(Joint):
    """The usual ETS parameter ranges: alpha ~ U(0.05, 0.95); beta = U(0.02, 0.98) * alpha;
    gamma = U(0.02, 0.98) * (1 - alpha); damping phi ~ U(0.8, 0.98). (beta, gamma, phi) are
    redrawn until the model is admissible."""

    def own_support(self, context):
        # A declared sampler, not proof that independently overridden tuples are admissible.
        return context.joint({"phi": {"kind": "real", "interval": self.phi_range if self.damped else (1., 1.)}})


    keys = ("alpha", "beta", "gamma", "phi")

    def __init__(self, trend: bool, seasonal: bool, period: int | None, damped: bool = False,
                 alpha_range=(0.05, 0.95), frac_range=(0.02, 0.98), phi_range=(0.8, 0.98)):
        self.trend, self.seasonal, self.period, self.damped = trend, seasonal, period, damped
        self.alpha_range, self.frac_range, self.phi_range = alpha_range, frac_range, phi_range

    def describe(self):
        return "ets_smoothing"

    def _needs_check(self):
        return self.seasonal or self.damped

    def sample(self, rng, draw):
        a = rng.uniform(*self.alpha_range)
        for _ in range(500):
            b = rng.uniform(*self.frac_range) * a if self.trend else 0.0
            g = rng.uniform(*self.frac_range) * (1 - a) if self.seasonal else 0.0
            ph = rng.uniform(*self.phi_range) if self.damped else 1.0
            if not self._needs_check() or is_admissible_ets(a, b, g, ph, self.trend, self.seasonal, self.period):
                break
        return {"alpha": a, "beta": b, "gamma": g, "phi": ph}

    def _propose(self, gen, a):
        k, dev = len(a), a.device
        lo, hi = self.frac_range
        r = lambda: torch.rand(k, generator=gen, device=dev)          # noqa: E731
        b = (lo + (hi - lo) * r()) * a if self.trend else torch.zeros(k, device=dev)
        g = (lo + (hi - lo) * r()) * (1 - a) if self.seasonal else torch.zeros(k, device=dev)
        ph = (self.phi_range[0] + (self.phi_range[1] - self.phi_range[0]) * r() if self.damped
              else torch.ones(k, device=dev))
        return b, g, ph

    def sample_batch(self, gen, n, device, draw):
        a = self.alpha_range[0] + (self.alpha_range[1] - self.alpha_range[0]) * torch.rand(
            n, generator=gen, device=device)
        b, g, ph = self._propose(gen, a)
        if self._needs_check():
            # rejection, as in `sample`, vectorized: each rejected row gets k proposals at
            # once and keeps the first admissible one (the same distribution as proposing
            # one at a time; for some alpha at period 12 acceptance is rare)
            def check(a_, b_, g_, p_):
                return ets_admissible_batch(a_, b_, g_, p_, self.trend, self.seasonal, self.period)
            todo, k = torch.arange(n, device=device), 64
            for _ in range(200):
                todo = todo[~check(a[todo], b[todo], g[todo], ph[todo])]
                if not len(todo):
                    break
                ar_ = a[todo].repeat_interleave(k)
                pb, pg, pph = self._propose(gen, ar_)
                ok = check(ar_, pb, pg, pph).view(-1, k)
                pick = torch.arange(len(todo), device=device) * k + ok.float().argmax(1)
                b[todo], g[todo], ph[todo] = pb[pick], pg[pick], pph[pick]   # failures are rechecked
        return {"alpha": a, "beta": b, "gamma": g, "phi": ph}


ETS_SPECS = ("ANN", "AAN", "AADN", "ANA", "AAA", "AADA")


class ETS(Compound):
    """Additive-error exponential smoothing: spec "ANN", "AAN", "AAdN", "ANA", "AAA" or
    "AAdA" (error, trend [d = damped], season). Built from one shared noise:

        y_t = l0 + e_t + alpha c1_{t-1}                               level
              + b0 t + beta c2_{t-1}                                  trend
              + b0 (phi + ... + phi^t) + beta phi cumsum(a)_{t-1}     damped trend
              + s0(t) + gamma sc_{t-m}                                season

    with c1 = cumsum(e), c2 = cumsum(c1), sc the seasonal cumulative sum and a the AR(1)
    filter of e with coefficient phi, started at t = 1. Equals the usual recursion
    l_t = l_{t-1} + phi b_{t-1} + alpha e_t, b_t = phi b_{t-1} + beta e_t,
    s_t = s_{t-m} + gamma e_t, y_t = l_{t-1} + phi b_{t-1} + s_{t-m} + e_t.
    """

    name = "ets"

    TRENDS = {None: "N", "none": "N", "additive": "A", "damped": "Ad"}

    def __init__(self, spec: str | None = None, period: int | None = None, *, trend=None, season=None,
                 alpha=None, beta=None, gamma=None, phi=None, level0=None, slope0=None, season0=None,
                 min_length: int | None = None):
        """Either a letter code, `ets("AAdA", period=4)`, or keywords,
        `ets(trend="damped", season=4)`: trend None | "additive" | "damped", season None or
        the period. Starting states: level0, slope0, season0 (a zero-sum pattern)."""
        if spec is None:
            if trend not in self.TRENDS:
                raise ValueError(f"trend must be one of {list(self.TRENDS)}")
            if season is not None and period is not None and season != period:
                raise ValueError("season and period disagree")
            period = season if season is not None else period
            spec = "A" + self.TRENDS[trend] + ("A" if period else "N")
        elif trend is not None or season is not None:
            raise ValueError("give either a letter code or trend= / season=, not both")
        key = spec.upper()
        if key not in ETS_SPECS:
            raise NotImplementedError(f"ets({spec!r}): choose from ANN, AAN, AAdN, ANA, AAA, AAdA")
        self.has_season = key[-1] == "A"
        if self.has_season and not period:
            raise ValueError(f"ets({spec!r}) needs a period")
        self.spec, self.period = spec, (period if self.has_season else None)
        self.has_trend, self.damped = key[1] == "A", key.startswith("AAD")
        joint = EtsSmoothing(self.has_trend, self.has_season, self.period, self.damped)
        self.alpha = as_dist(joint["alpha"] if alpha is None else alpha)
        self.beta = as_dist(joint["beta"] if beta is None else beta)
        self.gamma = as_dist(joint["gamma"] if gamma is None else gamma)
        self.phi = as_dist(joint["phi"] if phi is None else phi)
        self.level0 = as_dist(Uniform(-10, 10) if level0 is None else level0)
        self.slope0 = as_dist(Uniform(-1, 1) if slope0 is None else slope0)
        self.season0 = as_dist(Centered(Uniform(-5, 5), period) if (season0 is None and self.has_season)
                               else season0 if season0 is not None else 0.0)
        self._floor = min_length if min_length is not None else (
            math.ceil(2.5 * period) if self.has_season else 9)
        super().__init__()

    def build(self, e):
        y = constant(self.level0) + e + self.alpha * (e >> integrate() >> lag(1))
        if self.has_trend and not self.damped:
            y = y + trend(self.slope0) + self.beta * (e >> integrate(2) >> lag(1))
        if self.damped:
            y = (y + damped_trend(self.slope0, self.phi)
                 + (self.beta * self.phi) * (e >> ar(1, phi=self.phi, init="zero") >> integrate() >> lag(1)))
        if self.has_season:
            m = self.period
            y = y + pattern(m, self.season0) + self.gamma * (e >> integrate(period=m) >> lag(m))
        return y

    def label(self):
        return f"ets({self.spec}" + (f", period={self.period})" if self.period else ")")

    def params(self):
        out = {"alpha": self.alpha, "level0": self.level0}
        if self.has_trend:
            out |= {"beta": self.beta, "slope0": self.slope0}
        if self.damped:
            out |= {"phi": self.phi}
        if self.has_season:
            out |= {"gamma": self.gamma, "season0": self.season0}
        return out

    def floor(self, s):
        return self._floor


# ---------------------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------------------
class LogitNormalEts(EtsSmoothing):
    """ETS smoothing parameters from a multivariate normal on the logit scale of
    (alpha, beta/alpha, gamma/(1-alpha), (phi-0.8)/0.18), restricted to the coordinates the
    form uses; admissibility is enforced by rejection like the default sampler, and rows still
    inadmissible after the proposals fall back to it. Used for empirical-prior experiments
    (experiments/accuracy/empirical_priors.py); the default ETS prior is EtsSmoothing."""

    def __init__(self, trend, seasonal, period, damped, mean, cov):
        super().__init__(trend, seasonal, period, damped)
        use = [True, trend, seasonal, damped]
        self.mean = [m for m, u in zip(mean, use) if u]
        self.cov = [[c for c, u in zip(row, use) if u] for row, ru in zip(cov, use) if ru]

    def describe(self):
        return "logit_normal_ets"

    def _draw(self, gen, n, device):
        k = len(self.mean)
        L = torch.linalg.cholesky(torch.tensor(self.cov, dtype=torch.float64))
        z = torch.randn(n, k, generator=gen, device=device, dtype=torch.float64) @ L.T.to(device)
        u = torch.sigmoid(z + torch.tensor(self.mean, dtype=torch.float64, device=device)).float()
        cols, i = {}, 0
        for name, used in (("alpha", True), ("bfrac", self.trend), ("gfrac", self.seasonal), ("pfrac", self.damped)):
            if used:
                cols[name] = u[:, i]
                i += 1
        a = cols["alpha"].clamp(0.001, 0.999)
        b = cols["bfrac"] * a if self.trend else torch.zeros(n, device=device)
        g = cols["gfrac"] * (1 - a) if self.seasonal else torch.zeros(n, device=device)
        ph = 0.8 + 0.18 * cols["pfrac"] if self.damped else torch.ones(n, device=device)
        return a, b, g, ph

    def sample_batch(self, gen, n, device, draw):
        a, b, g, ph = self._draw(gen, n, device)
        if self._needs_check():
            bad = ~ets_admissible_batch(a, b, g, ph, self.trend, self.seasonal, self.period)
            for _ in range(50):
                if not bad.any():
                    break
                idx = bad.nonzero().ravel()
                a2, b2, g2, p2 = self._draw(gen, len(idx), device)
                a[idx], b[idx], g[idx], ph[idx] = a2, b2, g2, p2
                bad = ~ets_admissible_batch(a, b, g, ph, self.trend, self.seasonal, self.period)
            if bad.any():                                       # rare: default sampler
                idx = bad.nonzero().ravel()
                d = super().sample_batch(gen, len(idx), device, draw)
                a[idx], b[idx], g[idx], ph[idx] = d["alpha"], d["beta"], d["gamma"], d["phi"]
        return {"alpha": a, "beta": b, "gamma": g, "phi": ph}

    def sample(self, rng, draw):
        gen = torch.Generator().manual_seed(int(rng.integers(2 ** 62)))
        out = self.sample_batch(gen, 1, "cpu", draw)
        return {k: float(v[0]) for k, v in out.items()}


def arima_min_length(p, q, d, P=0, Q=0, D=0, s=1):
    """Shortest context that identifies ARIMA(p,d,q)(P,D,Q)_s: the 2k + 1 rule (k free
    parameters, constant included) after what differencing consumes; a seasonal model
    also needs two full seasons after seasonal differencing. Works elementwise."""
    k = p + q + P + Q + 1
    base = d + D * s + 2 * k + 1
    if s > 1:
        seasonal = (P > 0) | (Q > 0) | (D > 0)
        base = np.where(seasonal, np.maximum(base, d + D * s + 2 * s), base) if not torch.is_tensor(base) \
            else torch.where(seasonal, torch.maximum(base, d + D * s + 2 * s), base)
    return base


class ARIMA(Compound):
    """ARIMA(p,d,q)(P,D,Q)_period, built as

        noise >> ma(q) >> ma(Q, period) >> ar(p) >> ar(P, period)  [+ mean]
              >> integrate(d) >> integrate(D, period)

    The mean (if any) is the stationary mean of the differenced series. With `prior`
    (a joint distribution, e.g. `tg.priors.sarima_tree`), the orders given here are
    maxima and the prior draws the realized orders, coefficients, differencing and mean
    per series: an ARIMA with random orders is a selection across ARIMA orders,
    simulated in one pass.
    """

    name = "arima"

    def __init__(self, p=0, d=0, q=0, seasonal=(0, 0, 0), period=1, phi=None, theta=None,
                 seasonal_phi=None, seasonal_theta=None, mean=None, level0=None, season0=None,
                 prior: Joint | None = None, min_length: int | None = None):
        self.p, self.d, self.q = int(p), d, int(q)
        self.P, self.D, self.Q = int(seasonal[0]), seasonal[1], int(seasonal[2])
        self.period, self.prior, self._floor = int(period), prior, min_length or 1
        get = (lambda key, default: prior[key]) if prior is not None else (lambda key, default: default)
        self.phi = as_dist(get("phi", Stationary(self.p) if phi is None else phi))
        self.theta = as_dist(get("theta", Invertible(self.q) if theta is None else theta))
        self.seasonal_phi = as_dist(get("Phi", Stationary(self.P) if seasonal_phi is None else seasonal_phi))
        self.seasonal_theta = as_dist(get("Theta", Invertible(self.Q) if seasonal_theta is None else seasonal_theta))
        self.d_param = as_dist(get("d", self.d))
        self.D_param = as_dist(get("D", self.D))
        self.mean = None if (mean is None and prior is None) else as_dist(get("mean", mean))
        # explicit starting states of the integrators: level0 enters between the
        # ordinary and the seasonal integrator, season0 (a zero-sum pattern) after both
        self.level0 = None if (level0 is None and prior is None) else as_dist(get("level0", level0))
        self.season0 = None if (season0 is None and (prior is None or self.period == 1)) \
            else as_dist(get("season0", season0))
        super().__init__()

    def build(self, e):
        x, s = e, self.period
        if self.q:
            x = x >> ma(self.q, theta=self.theta)
        if self.Q:
            x = x >> ma(self.Q, period=s, theta=self.seasonal_theta)
        if self.p:
            x = x >> ar(self.p, phi=self.phi)
        if self.P:
            x = x >> ar(self.P, period=s, phi=self.seasonal_phi)
        if self.mean is not None:
            x = x + constant(self.mean)
        if self.prior is not None or self.d:
            x = x >> integrate(self.d_param)
        if self.level0 is not None:
            x = x + constant(self.level0)
        if self.prior is not None or self.D:
            x = x >> integrate(self.D_param, period=s)
        if self.season0 is not None:
            x = x + pattern(s, self.season0)
        return x

    def label(self):
        orders = f"{self.p}, {self.d}, {self.q}"
        seas = f", seasonal=({self.P}, {self.D}, {self.Q}), period={self.period}" if self.period > 1 else ""
        pr = f", prior={self.prior.describe()}" if self.prior is not None else ""
        return f"arima({orders}{seas}{pr})"

    def params(self):
        if self.prior is not None:
            return {k: self.prior[k] for k in self.prior.keys}
        out = {}
        for name, order, dist in (("phi", self.p, self.phi), ("theta", self.q, self.theta),
                                  ("seasonal_phi", self.P, self.seasonal_phi),
                                  ("seasonal_theta", self.Q, self.seasonal_theta)):
            if order:
                out[name] = dist
        if self.mean is not None:
            out["mean"] = self.mean
        if self.level0 is not None:
            out["level0"] = self.level0
        if self.season0 is not None:
            out["season0"] = self.season0
        return out

    def _orders(self, draw):
        if self.prior is None:
            return self.p, self.q, int(self.d), self.P, self.Q, int(self.D)
        o = draw(self.prior)
        return o["p"], o["q"], o["d"], o["P"], o["Q"], o["D"]

    def floor(self, s):
        p, q, d, P, Q, D = self._orders(s.draw)
        return max(self._floor, int(arima_min_length(p, q, d, P, Q, D, self.period)))

    def floor_batch(self, b):
        p, q, d, P, Q, D = self._orders(b.draw)
        if self.prior is None:
            return torch.full((b.n,), float(self.floor(b)), device=b.device)
        return arima_min_length(p, q, d, P, Q, D, self.period).float().clamp_min(self._floor)


# ---------------------------------------------------------------------------------------
# unobserved components (Harvey): one independent noise per component
# ---------------------------------------------------------------------------------------
def local_level(level=None):
    """irregular + random-walk level, independent noises."""
    return constant(level) + noise(scale=Uniform(0, 1)) + (noise(scale=Uniform(0, 1)) >> integrate())


def local_trend(level=None, slope=None):
    """irregular + random-walk level + random-walk slope, independent noises."""
    return (constant(level) + trend(slope) + noise(scale=Uniform(0, 1))
            + (noise(scale=Uniform(0, 1)) >> integrate())
            + (noise(scale=Uniform(0, 1)) >> integrate(2) >> lag(1)))


def seasonal(period, pattern_values=None):
    """Zero-sum (dummy) seasonal: noise, differenced, then seasonally integrated, plus a
    starting pattern."""
    from .blocks import difference
    return pattern(period, pattern_values) + (noise(scale=Uniform(0, 1)) >> difference()
                                              >> integrate(period=period))


def ets(spec=None, period=None, **kw):
    return ETS(spec, period, **kw)


def arima(p=0, d=0, q=0, seasonal=(0, 0, 0), period=1, **kw):
    return ARIMA(p, d, q, seasonal, period, **kw)
